// gram newton-schulz variant. assumes muon.cu was already #include'd
// so CUDA_CHECK / CUBLAS_CHECK / mom_update / scale_matrix / muon_update_kernel
// are already in scope. only the new bits live here.
//
// modes:
//   GNS_QUINTIC              -- naive scalar a, b, c (a=3.4445, b=-4.7750, c=2.0315)
//   GNS_POLAR                -- per-iter polar express coeffs (5-step schedule)
//   GNS_POLAR_RESTART        -- polar + restart at k=2 (reform G from Y=X*R)
//   GNS_POLAR_RESTART_SYRK   -- + cublasSsyrk for X^T X / Y^T Y, mirror lower→upper
//
// fp16 path is fixed to polar+restart (the regime where stability matters).

#include <cuda_fp16.h>

enum GNSMode { GNS_QUINTIC, GNS_POLAR, GNS_POLAR_RESTART, GNS_POLAR_RESTART_SYRK };

// ---- fp32 kernels -----------------------------------------------------------

__global__ void set_identity(float* __restrict__ R, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*N){
    int row = idx / N;
    int col = idx % N;
    R[idx] = (row == col) ? 1.0f : 0.0f;
  }
}

__global__ void build_M(float* __restrict__ G, float* __restrict__ GG, float* __restrict__ M, int N, float a, float b, float c){
  // M = a*I + b*G + c*GG, fused into one launch (fp32 is fine)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*N){
    int row = idx / N;
    int col = idx % N;
    float val = b * G[idx] + c * GG[idx];
    if(row == col) val += a;
    M[idx] = val;
  }
}

__global__ void mirror_lower_to_upper(float* __restrict__ G, int N){
  // column-major: G[i + j*N]. ssyrk fills lower (i>=j); mirror to upper.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*N){
    int j = idx / N;
    int i = idx % N;
    if(i < j) G[i + j*N] = G[j + i*N];
  }
}

// ---- fp32 launch ------------------------------------------------------------

void gram_newton_schulz_launch(float* __restrict__ d_input, float* __restrict__ d_output, int N, int M, int iterations, GNSMode mode, cublasHandle_t handle){
  // polar express coeffs (5-step schedule). clamp index if iterations > 5.
  const float pe_a[5] = {8.123737f, 4.026529f, 3.870284f, 3.253351f, 2.300652f};
  const float pe_b[5] = {-22.232240f, -2.776323f, -2.739120f, -2.343223f, -1.668904f};
  const float pe_c[5] = {16.373715f, 0.514551f, 0.520999f, 0.481420f, 0.418807f};

  bool use_syrk    = (mode == GNS_POLAR_RESTART_SYRK);
  bool use_restart = (mode == GNS_POLAR_RESTART || mode == GNS_POLAR_RESTART_SYRK);
  bool use_polar   = (mode != GNS_QUINTIC);

  float alpha = 1.0f, beta = 0.0f;
  int orig_N = N, orig_M = M;
  int len = N * M;

  // normalize by frobenius norm
  float norm;
  CUBLAS_CHECK(cublasSnrm2(handle, len, d_input, 1, &norm));

  const float eps = 1e-6f;
  float inv_scale = 1.0f / (norm + eps);
  int threads = 256;
  int blocks = (len + threads - 1)/threads;
  scale_matrix<<<blocks, threads>>>(d_input, len, inv_scale);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ensure X is `working_M × working_N` col-major with working_M >= working_N (lda = working_M).
  // Tall input (N >= M) is already in this layout, no transpose needed. Wide input (N < M) is
  // transposed so that rows become the larger dim.
  bool transposed = false;
  int working_M, working_N;
  float* d_input_transposed = nullptr;

  if(N >= M){
    working_M = N;
    working_N = M;
  } else {
    CUDA_CHECK(cudaMalloc((void**)&d_input_transposed, (size_t)orig_N * orig_M * sizeof(float)));
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_M, orig_N, &alpha, d_input, orig_N, &beta, d_input, orig_N, d_input_transposed, orig_M));
    working_M = M;
    working_N = N;
    transposed = true;
  }
  float* X = transposed ? d_input_transposed : d_input;

  // small N×N buffers for the iteration
  float *d_G, *d_GG, *d_M, *d_R, *d_R_new, *d_MG, *d_G_new;
  size_t size_NN = working_N * working_N * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_G,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_GG,    size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_M,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_R,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_R_new, size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_MG,    size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_G_new, size_NN));

  // rect scratch (working_M × working_N) -- restart's Y_temp + final Y_working
  float* d_rect;
  CUDA_CHECK(cudaMalloc((void**)&d_rect, working_M * working_N * sizeof(float)));

  // R = I, then G = X^T X (sgemm or ssyrk depending on mode)
  int blocks_NN = (working_N * working_N + threads - 1) / threads;
  set_identity<<<blocks_NN, threads>>>(d_R, working_N);
  CUDA_CHECK(cudaDeviceSynchronize());

  if(use_syrk){
    // ssyrk on ada can be slower than sgemm for skinny shapes -- benchmark to see
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                              working_N, working_M,
                              &alpha, X, working_M,
                              &beta, d_G, working_N));
    mirror_lower_to_upper<<<blocks_NN, threads>>>(d_G, working_N);
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              working_N, working_N, working_M,
                              &alpha, X, working_M,
                              X, working_M,
                              &beta, d_G, working_N));
  }

  // iteration loop -- N×N from here, except the restart at k=2
  for(int k = 0; k < iterations; k++){
    // pick coefficients
    float a, b, c;
    if(use_polar){
      int i = k < 5 ? k : 4;  // polar express is a 5-step schedule
      a = pe_a[i]; b = pe_b[i]; c = pe_c[i];
    } else {
      a = 3.4445f; b = -4.7750f; c = 2.0315f;
    }

    // GG = G @ G
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_G, working_N, d_G, working_N, &beta, d_GG, working_N));

    // M = a*I + b*G + c*GG
    build_M<<<blocks_NN, threads>>>(d_G, d_GG, d_M, working_N, a, b, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    // R_new = R @ M
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_R, working_N, d_M, working_N, &beta, d_R_new, working_N));
    std::swap(d_R, d_R_new);

    if(use_restart && k == 2){
      // restart: reform gram matrix from current Y = X * R, reset R = I
      // step 1: Y_temp (working_M × working_N) = X @ R
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                working_M, working_N, working_N,
                                &alpha, X, working_M,
                                d_R, working_N,
                                &beta, d_rect, working_M));
      // step 2: G = Y_temp^T @ Y_temp (symmetric -- syrk path optional)
      if(use_syrk){
        CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                  working_N, working_M,
                                  &alpha, d_rect, working_M,
                                  &beta, d_G, working_N));
        mirror_lower_to_upper<<<blocks_NN, threads>>>(d_G, working_N);
        CUDA_CHECK(cudaDeviceSynchronize());
      } else {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  working_N, working_N, working_M,
                                  &alpha, d_rect, working_M,
                                  d_rect, working_M,
                                  &beta, d_G, working_N));
      }
      // step 3: R = I
      set_identity<<<blocks_NN, threads>>>(d_R, working_N);
      CUDA_CHECK(cudaDeviceSynchronize());
    } else if(k < iterations - 1){
      // normal G update path (only when no restart this step)
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_M, working_N, d_G, working_N, &beta, d_MG, working_N));
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_MG, working_N, d_M, working_N, &beta, d_G_new, working_N));
      std::swap(d_G, d_G_new);
    }
  }

  // Y = X @ R (the second and final rectangular matmul). reuse d_rect when transposed.
  float* Y_working = transposed ? d_rect : d_output;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_M, working_N, working_N, &alpha, X, working_M, d_R, working_N, &beta, Y_working, working_M));

  if(transposed){
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_N, orig_M, &alpha, Y_working, working_M, &beta, Y_working, working_M, d_output, orig_N));
    CUDA_CHECK(cudaFree(d_input_transposed));
  }

  CUDA_CHECK(cudaFree(d_rect));
  CUDA_CHECK(cudaFree(d_G));
  CUDA_CHECK(cudaFree(d_GG));
  CUDA_CHECK(cudaFree(d_M));
  CUDA_CHECK(cudaFree(d_R));
  CUDA_CHECK(cudaFree(d_R_new));
  CUDA_CHECK(cudaFree(d_MG));
  CUDA_CHECK(cudaFree(d_G_new));
}

void gns_muon_step(float* __restrict__ d_W, float* __restrict__ d_G, float* __restrict__ d_M, float* __restrict__ d_U, int N, int M, float lr, float weight_decay, GNSMode mode){
  int size = N * M;
  int threads = 256;
  int blocks = (size + threads - 1)/threads;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  mom_update<<<blocks, threads>>>(d_M, d_G, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  int ns_iterations = 5;
  gram_newton_schulz_launch(d_M, d_U, N, M, ns_iterations, mode, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  // performing W←(1−ηλ)W−ηU
  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}

// ---- fp16 kernels -----------------------------------------------------------

__global__ void set_identity_fp16(__half* __restrict__ R, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*N){
    int row = idx / N;
    int col = idx % N;
    R[idx] = __float2half((row == col) ? 1.0f : 0.0f);
  }
}

__global__ void build_M_fp16(__half* __restrict__ G, __half* __restrict__ GG, __half* __restrict__ M, int N, float a, float b, float c){
  // load fp16, compute fp32, store fp16 -- avoids fp16 accum drift in build_M
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*N){
    int row = idx / N;
    int col = idx % N;
    float val = b * __half2float(G[idx]) + c * __half2float(GG[idx]);
    if(row == col) val += a;
    M[idx] = __float2half(val);
  }
}

__global__ void scale_matrix_fp16(__half* __restrict__ X, int n, float scale){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n) X[idx] = __float2half(__half2float(X[idx]) * scale);
}

__global__ void half_to_float_kernel(const __half* __restrict__ in, float* __restrict__ out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n) out[idx] = __half2float(in[idx]);
}

__global__ void float_to_half_kernel(const float* __restrict__ in, __half* __restrict__ out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n) out[idx] = __float2half(in[idx]);
}

// ---- fp16 launch ------------------------------------------------------------
// always polar express + restart -- fp16 needs the stability.
// no cublasHsyrk -- always use cublasGemmEx for the gram formation.

static inline void gemm_fp16(cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, float alpha, const __half* A, int lda, const __half* B, int ldb, float beta, __half* C, int ldc){
  // fp16 storage, fp32 accum -- standard stable path on ada tensor cores
  CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k,
                            &alpha, A, CUDA_R_16F, lda,
                                    B, CUDA_R_16F, ldb,
                            &beta,  C, CUDA_R_16F, ldc,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void gram_newton_schulz_launch_fp16(__half* __restrict__ d_input, __half* __restrict__ d_output, int N, int M, int iterations, cublasHandle_t handle){
  const float pe_a[5] = {8.123737f, 4.026529f, 3.870284f, 3.253351f, 2.300652f};
  const float pe_b[5] = {-22.232240f, -2.776323f, -2.739120f, -2.343223f, -1.668904f};
  const float pe_c[5] = {16.373715f, 0.514551f, 0.520999f, 0.481420f, 0.418807f};

  int orig_N = N, orig_M = M;
  int len = N * M;

  // norm via fp32 copy (no cublasHnrm2). one-shot per call.
  float* d_in_fp32_tmp;
  CUDA_CHECK(cudaMalloc((void**)&d_in_fp32_tmp, len * sizeof(float)));
  int threads = 256;
  int blocks = (len + threads - 1)/threads;
  half_to_float_kernel<<<blocks, threads>>>(d_input, d_in_fp32_tmp, len);
  CUDA_CHECK(cudaDeviceSynchronize());

  float norm;
  CUBLAS_CHECK(cublasSnrm2(handle, len, d_in_fp32_tmp, 1, &norm));
  CUDA_CHECK(cudaFree(d_in_fp32_tmp));

  const float eps = 1e-6f;
  float inv_scale = 1.0f / (norm + eps);
  scale_matrix_fp16<<<blocks, threads>>>(d_input, len, inv_scale);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ensure X is `working_M × working_N` col-major with working_M >= working_N (lda = working_M).
  // Tall input (N >= M) needs no transpose; wide (N < M) is transposed (fp32 round-trip since cublas has no Hgeam).
  bool transposed = false;
  int working_M, working_N;
  __half* d_input_transposed = nullptr;
  float alpha_f = 1.0f, beta_f = 0.0f;

  if(N >= M){
    working_M = N;
    working_N = M;
  } else {
    CUDA_CHECK(cudaMalloc((void**)&d_input_transposed, (size_t)orig_N * orig_M * sizeof(__half)));
    float *d_tmp_in_f, *d_tmp_out_f;
    CUDA_CHECK(cudaMalloc((void**)&d_tmp_in_f,  (size_t)orig_N * orig_M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_tmp_out_f, (size_t)orig_N * orig_M * sizeof(float)));
    half_to_float_kernel<<<blocks, threads>>>(d_input, d_tmp_in_f, orig_N * orig_M);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_M, orig_N, &alpha_f, d_tmp_in_f, orig_N, &beta_f, d_tmp_in_f, orig_N, d_tmp_out_f, orig_M));
    int blocks2 = (orig_N * orig_M + threads - 1) / threads;
    float_to_half_kernel<<<blocks2, threads>>>(d_tmp_out_f, d_input_transposed, orig_N * orig_M);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_tmp_in_f));
    CUDA_CHECK(cudaFree(d_tmp_out_f));
    working_M = M;
    working_N = N;
    transposed = true;
  }
  __half* X = transposed ? d_input_transposed : d_input;

  // small N×N buffers (fp16)
  __half *d_G, *d_GG, *d_M, *d_R, *d_R_new, *d_MG, *d_G_new;
  size_t size_NN = working_N * working_N * sizeof(__half);
  CUDA_CHECK(cudaMalloc((void**)&d_G,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_GG,    size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_M,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_R,     size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_R_new, size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_MG,    size_NN));
  CUDA_CHECK(cudaMalloc((void**)&d_G_new, size_NN));

  __half* d_rect;
  CUDA_CHECK(cudaMalloc((void**)&d_rect, working_M * working_N * sizeof(__half)));

  int blocks_NN = (working_N * working_N + threads - 1) / threads;
  set_identity_fp16<<<blocks_NN, threads>>>(d_R, working_N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // G = X^T X via gemmEx (fp32 accum)
  gemm_fp16(handle, CUBLAS_OP_T, CUBLAS_OP_N, working_N, working_N, working_M, 1.0f, X, working_M, X, working_M, 0.0f, d_G, working_N);

  for(int k = 0; k < iterations; k++){
    int i = k < 5 ? k : 4;
    float a = pe_a[i], b = pe_b[i], c = pe_c[i];

    gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, 1.0f, d_G, working_N, d_G, working_N, 0.0f, d_GG, working_N);

    build_M_fp16<<<blocks_NN, threads>>>(d_G, d_GG, d_M, working_N, a, b, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, 1.0f, d_R, working_N, d_M, working_N, 0.0f, d_R_new, working_N);
    std::swap(d_R, d_R_new);

    if(k == 2){
      // restart
      gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_M, working_N, working_N, 1.0f, X, working_M, d_R, working_N, 0.0f, d_rect, working_M);
      gemm_fp16(handle, CUBLAS_OP_T, CUBLAS_OP_N, working_N, working_N, working_M, 1.0f, d_rect, working_M, d_rect, working_M, 0.0f, d_G, working_N);
      set_identity_fp16<<<blocks_NN, threads>>>(d_R, working_N);
      CUDA_CHECK(cudaDeviceSynchronize());
    } else if(k < iterations - 1){
      gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, 1.0f, d_M, working_N, d_G, working_N, 0.0f, d_MG, working_N);
      gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, 1.0f, d_MG, working_N, d_M, working_N, 0.0f, d_G_new, working_N);
      std::swap(d_G, d_G_new);
    }
  }

  // Y = X @ R
  __half* Y_working = transposed ? d_rect : d_output;
  gemm_fp16(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_M, working_N, working_N, 1.0f, X, working_M, d_R, working_N, 0.0f, Y_working, working_M);

  if(transposed){
    // un-transpose via fp32 sgeam
    float *d_y_f, *d_o_f;
    CUDA_CHECK(cudaMalloc((void**)&d_y_f, working_M * working_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_o_f, orig_N * orig_M * sizeof(float)));
    int blocks_y = (working_M * working_N + threads - 1) / threads;
    half_to_float_kernel<<<blocks_y, threads>>>(Y_working, d_y_f, working_M * working_N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_N, orig_M, &alpha_f, d_y_f, working_M, &beta_f, d_y_f, working_M, d_o_f, orig_N));
    int blocks_o = (orig_N * orig_M + threads - 1) / threads;
    float_to_half_kernel<<<blocks_o, threads>>>(d_o_f, d_output, orig_N * orig_M);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_y_f));
    CUDA_CHECK(cudaFree(d_o_f));
    CUDA_CHECK(cudaFree(d_input_transposed));
  }

  CUDA_CHECK(cudaFree(d_rect));
  CUDA_CHECK(cudaFree(d_G));
  CUDA_CHECK(cudaFree(d_GG));
  CUDA_CHECK(cudaFree(d_M));
  CUDA_CHECK(cudaFree(d_R));
  CUDA_CHECK(cudaFree(d_R_new));
  CUDA_CHECK(cudaFree(d_MG));
  CUDA_CHECK(cudaFree(d_G_new));
}

