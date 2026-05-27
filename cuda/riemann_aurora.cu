// riemannian aurora variant. standalone entry that pulls in the base muon primitives.

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"

__global__ void riem_momentum_update_kernel(
    float* __restrict__ momentum,
    const float* __restrict__ grad,
    float* __restrict__ update,
    int size,
    float mu,
    bool nesterov
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size) return;

  float g = grad[idx];
  float m = mu * momentum[idx] + (1.0f - mu) * g;
  momentum[idx] = m;
  update[idx] = nesterov ? ((1.0f - mu) * g + mu * m) : m;
}

__global__ void riem_fill_kernel(float* __restrict__ x, int size, float value){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) x[idx] = value;
}

__global__ void riem_symmetrize_kernel(float* __restrict__ A, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = n * n;
  if(idx >= size) return;

  int row = idx % n;
  int col = idx / n;
  if(row < col){
    float a = A[row + col * n];
    float b = A[col + row * n];
    float s = 0.5f * (a + b);
    A[row + col * n] = s;
    A[col + row * n] = s;
  }
}

__global__ void riem_row_dot_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    int rows,
    int cols
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float acc = 0.0f;
  for(int col = 0; col < cols; col++){
    acc += A[row + col * rows] * B[row + col * rows];
  }
  out[row] = acc;
}

__global__ void riem_subtract_vec_mean_kernel(float* __restrict__ x, int n, float mean){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) x[idx] -= mean;
}

__global__ void riem_scale_rows_kernel(
    const float* __restrict__ X,
    const float* __restrict__ row_scale,
    float* __restrict__ Y,
    int rows,
    int cols
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = rows * cols;
  if(idx >= size) return;

  int row = idx % rows;
  Y[idx] = row_scale[row] * X[idx];
}

__global__ void riem_axpby_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int n,
    float a,
    float b
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) y[idx] = a * x[idx] + b * y[idx];
}

__global__ void riem_vec_add_scaled_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int n,
    float a
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) y[idx] += a * x[idx];
}

__global__ void riem_vec_copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) dst[idx] = src[idx];
}

__global__ void riem_build_S_kernel(
    const float* __restrict__ B,
    const float* __restrict__ UtDU,
    float* __restrict__ S,
    int n
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n * n) S[idx] = B[idx] - UtDU[idx];
}

__global__ void riem_build_Z_kernel(
    const float* __restrict__ G,
    const float* __restrict__ US,
    const float* __restrict__ lambda,
    const float* __restrict__ U,
    float* __restrict__ Z,
    int rows,
    int cols
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = rows * cols;
  if(idx >= size) return;

  int row = idx % rows;
  Z[idx] = G[idx] - US[idx] - lambda[row] * U[idx];
}

__global__ void riem_build_Y_kernel(
    const float* __restrict__ U,
    const float* __restrict__ Z,
    float* __restrict__ Y,
    int size,
    float eta
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) Y[idx] = U[idx] + eta * Z[idx];
}

__global__ void riem_row_norm_to_scale_kernel(
    float* __restrict__ X,
    int rows,
    int cols,
    float target,
    float eps
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float acc = 0.0f;
  for(int col = 0; col < cols; col++){
    float x = X[row + col * rows];
    acc += x * x;
  }

  float scale = target / fmaxf(sqrtf(acc), eps);
  for(int col = 0; col < cols; col++){
    X[row + col * rows] *= scale;
  }
}

__global__ void riem_cg_matvec_kernel(
    const float* __restrict__ UTU_weighted,
    const float* __restrict__ U,
    const float* __restrict__ v,
    float* __restrict__ Ap,
    int rows,
    int cols,
    float r_eff
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float acc = 0.0f;
  for(int col = 0; col < cols; col++){
    float t = 0.0f;
    for(int j = 0; j < cols; j++){
      t += U[row + j * rows] * UTU_weighted[j + col * cols];
    }
    acc += t * U[row + col * rows];
  }
  Ap[row] = r_eff * v[row] - acc;
}

float riem_vec_sum(float* d_x, int n, cublasHandle_t handle){
  float* d_ones;
  CUDA_CHECK(cudaMalloc((void**)&d_ones, n * sizeof(float)));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  riem_fill_kernel<<<blocks, threads>>>(d_ones, n, 1.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  float sum;
  CUBLAS_CHECK(cublasSdot(handle, n, d_x, 1, d_ones, 1, &sum));
  CUDA_CHECK(cudaFree(d_ones));
  return sum;
}

void riem_solve_row_norm_multipliers(
    float* __restrict__ d_U,
    float* __restrict__ d_b,
    float* __restrict__ d_lambda,
    int rows,
    int cols,
    int cg_steps,
    float r,
    cublasHandle_t handle
){
  int threads = 256;
  int row_blocks = (rows + threads - 1) / threads;
  int elem_blocks = (rows * cols + threads - 1) / threads;
  float r_eff = r + 1e-3f;
  float alpha = 1.0f;
  float beta = 0.0f;

  float *d_res, *d_p, *d_Ap, *d_vU, *d_T;
  CUDA_CHECK(cudaMalloc((void**)&d_res, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_p, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_Ap, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_vU, (size_t)rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_T, (size_t)cols * cols * sizeof(float)));

  riem_fill_kernel<<<row_blocks, threads>>>(d_lambda, rows, 0.0f);
  riem_vec_copy_kernel<<<row_blocks, threads>>>(d_res, d_b, rows);
  riem_vec_copy_kernel<<<row_blocks, threads>>>(d_p, d_b, rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  float rs_old;
  CUBLAS_CHECK(cublasSdot(handle, rows, d_res, 1, d_res, 1, &rs_old));
  if(rs_old <= 1e-30f){
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_vU));
    CUDA_CHECK(cudaFree(d_T));
    return;
  }

  for(int k = 0; k < cg_steps; k++){
    riem_scale_rows_kernel<<<elem_blocks, threads>>>(d_U, d_p, d_vU, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, rows, &alpha, d_U, rows, d_vU, rows, &beta, d_T, cols));
    riem_cg_matvec_kernel<<<row_blocks, threads>>>(d_T, d_U, d_p, d_Ap, rows, cols, r_eff);
    CUDA_CHECK(cudaDeviceSynchronize());

    float denom;
    CUBLAS_CHECK(cublasSdot(handle, rows, d_p, 1, d_Ap, 1, &denom));
    if(denom <= 1e-30f) break;

    float step = rs_old / denom;
    riem_vec_add_scaled_kernel<<<row_blocks, threads>>>(d_lambda, d_p, rows, step);
    riem_vec_add_scaled_kernel<<<row_blocks, threads>>>(d_res, d_Ap, rows, -step);
    CUDA_CHECK(cudaDeviceSynchronize());

    float rs_new;
    CUBLAS_CHECK(cublasSdot(handle, rows, d_res, 1, d_res, 1, &rs_new));
    if(rs_new <= 1e-16f) break;

    float p_beta = rs_new / fmaxf(rs_old, 1e-30f);
    riem_axpby_kernel<<<row_blocks, threads>>>(d_p, d_res, rows, 1.0f, p_beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    rs_old = rs_new;
  }

  CUDA_CHECK(cudaFree(d_res));
  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_Ap));
  CUDA_CHECK(cudaFree(d_vU));
  CUDA_CHECK(cudaFree(d_T));
}

void riemann_aurora_polar_launch(
    float* __restrict__ d_input,
    float* __restrict__ d_output,
    int N,
    int M,
    int outer_steps,
    int cg_steps,
    float riemannian_eta,
    int retraction_steps,
    float eps,
    cublasHandle_t handle
){
  int size = N * M;
  int threads = 256;
  int elem_blocks = (size + threads - 1) / threads;
  float alpha = 1.0f;
  float beta = 0.0f;

  if(N == M){
    float* d_square_input;
    CUDA_CHECK(cudaMalloc((void**)&d_square_input, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_square_input, d_input, size * sizeof(float), cudaMemcpyDeviceToDevice));
    newton_schulz_launch(d_square_input, d_output, N, M, 5, handle);
    CUDA_CHECK(cudaFree(d_square_input));
  } else {
    bool transposed = N < M;
    int rows = transposed ? M : N;
    int cols = transposed ? N : M;
    int row_blocks = (rows + threads - 1) / threads;
    int nn_blocks = (cols * cols + threads - 1) / threads;
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float *d_G, *d_U, *d_Y, *d_Y_polar, *d_B, *d_UB, *d_q, *d_lam, *d_lamU, *d_UtDU, *d_S, *d_US, *d_Z;
    CUDA_CHECK(cudaMalloc((void**)&d_G, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_U, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y_polar, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_UB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_q, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_lam, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_lamU, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_UtDU, (size_t)cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_S, (size_t)cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_US, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Z, bytes));

    if(transposed){
      CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha, d_input, N, &beta, d_G, M, d_G, M));
    } else {
      CUDA_CHECK(cudaMemcpy(d_G, d_input, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_Y, d_G, bytes, cudaMemcpyDeviceToDevice));
    newton_schulz_launch(d_Y, d_U, rows, cols, 5, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    float r = (float)cols / (float)rows;
    float target_row_norm = sqrtf(r);
    for(int step = 0; step < outer_steps; step++){
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, rows, &alpha, d_U, rows, d_G, rows, &beta, d_B, cols));
      riem_symmetrize_kernel<<<nn_blocks, threads>>>(d_B, cols);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, cols, &alpha, d_U, rows, d_B, cols, &beta, d_UB, rows));
      riem_row_dot_kernel<<<row_blocks, threads>>>(d_G, d_U, d_q, rows, cols);
      riem_row_dot_kernel<<<row_blocks, threads>>>(d_UB, d_U, d_lam, rows, cols);
      riem_vec_add_scaled_kernel<<<row_blocks, threads>>>(d_q, d_lam, rows, -1.0f);
      CUDA_CHECK(cudaDeviceSynchronize());

      float q_mean = riem_vec_sum(d_q, rows, handle) / (float)rows;
      riem_subtract_vec_mean_kernel<<<row_blocks, threads>>>(d_q, rows, q_mean);
      CUDA_CHECK(cudaDeviceSynchronize());

      riem_solve_row_norm_multipliers(d_U, d_q, d_lam, rows, cols, cg_steps, r, handle);
      float lam_mean = riem_vec_sum(d_lam, rows, handle) / (float)rows;
      riem_subtract_vec_mean_kernel<<<row_blocks, threads>>>(d_lam, rows, lam_mean);
      CUDA_CHECK(cudaDeviceSynchronize());

      riem_scale_rows_kernel<<<elem_blocks, threads>>>(d_U, d_lam, d_lamU, rows, cols);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, cols, rows, &alpha, d_U, rows, d_lamU, rows, &beta, d_UtDU, cols));
      riem_build_S_kernel<<<nn_blocks, threads>>>(d_B, d_UtDU, d_S, cols);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, cols, &alpha, d_U, rows, d_S, cols, &beta, d_US, rows));
      riem_build_Z_kernel<<<elem_blocks, threads>>>(d_G, d_US, d_lam, d_U, d_Z, rows, cols);
      riem_build_Y_kernel<<<elem_blocks, threads>>>(d_U, d_Z, d_Y, rows * cols, riemannian_eta);
      CUDA_CHECK(cudaDeviceSynchronize());

      for(int k = 0; k < retraction_steps; k++){
        riem_row_norm_to_scale_kernel<<<row_blocks, threads>>>(d_Y, rows, cols, target_row_norm, eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        newton_schulz_launch(d_Y, d_Y_polar, rows, cols, 5, handle);
        CUDA_CHECK(cudaDeviceSynchronize());
        float* tmp = d_Y;
        d_Y = d_Y_polar;
        d_Y_polar = tmp;
      }

      float* tmp = d_U;
      d_U = d_Y;
      d_Y = tmp;
    }

    if(transposed){
      CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_U, rows, &beta, d_output, N, d_output, N));
    } else {
      CUDA_CHECK(cudaMemcpy(d_output, d_U, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_G));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_Y_polar));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_UB));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_lam));
    CUDA_CHECK(cudaFree(d_lamU));
    CUDA_CHECK(cudaFree(d_UtDU));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_US));
    CUDA_CHECK(cudaFree(d_Z));
  }

  float aspect_scale = sqrtf(fmaxf(1.0f, (float)N / (float)M));
  scale_matrix<<<elem_blocks, threads>>>(d_output, size, aspect_scale);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void riemann_aurora_step(
    float* __restrict__ d_W,
    float* __restrict__ d_G,
    float* __restrict__ d_M,
    float* __restrict__ d_U,
    int N,
    int M,
    float lr,
    float weight_decay
){
  int size = N * M;
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  riem_momentum_update_kernel<<<blocks, threads>>>(d_M, d_G, d_U, size, 0.95f, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  // lighter than the python defaults so the laptop benchmark finishes.
  riemann_aurora_polar_launch(d_U, d_U, N, M, 2, 8, 0.1f, 1, 1e-7f, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}
