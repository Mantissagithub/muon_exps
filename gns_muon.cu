// gram newton-schulz variant. assumes muon.cu was already #include'd
// so CUDA_CHECK / CUBLAS_CHECK / mom_update / scale_matrix / muon_update_kernel
// are already in scope. only the new bits live here.

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

void gram_newton_schulz_launch(float* __restrict__ d_input, float* __restrict__ d_output, int N, int M, int iterations, cublasHandle_t handle){
  const float a = 3.4445f;
  const float b = -4.7750f;
  const float c = 2.0315f;
  float alpha = 1.0f, beta = 0.0f;

  int orig_N = N, orig_M = M;
  int len = N * M;

  // normalize by frobenius norm (same as v1)
  float norm;
  CUBLAS_CHECK(cublasSnrm2(handle, len, d_input, 1, &norm));

  const float eps = 1e-6f;
  float inv_scale = 1.0f / (norm + eps);
  int threads = 256;
  int blocks = (len + threads - 1)/threads;
  scale_matrix<<<blocks, threads>>>(d_input, len, inv_scale);
  CUDA_CHECK(cudaDeviceSynchronize());

  // transpose if N>M so we always have working_M >= working_N (gram on the small side)
  bool transposed = false;
  int working_N = N, working_M = M;
  float *d_input_transposed = nullptr;

  if(N > M){
    CUDA_CHECK(cudaMalloc((void**)&d_input_transposed, orig_N * orig_M * sizeof(float)));
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_M, orig_N, &alpha, d_input, orig_N, &beta, d_input, orig_N, d_input_transposed, orig_M));
    working_N = M;
    working_M = N;
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

  // R = I, G = X^T X (the one rectangular matmul per call besides Y=X R at the end)
  int blocks_NN = (working_N * working_N + threads - 1) / threads;
  set_identity<<<blocks_NN, threads>>>(d_R, working_N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, working_N, working_N, working_M, &alpha, X, working_M, X, working_M, &beta, d_G, working_N));

  // the iteration -- all N×N from here
  for(int k = 0; k < iterations; k++){
    // GG = G @ G
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_G, working_N, d_G, working_N, &beta, d_GG, working_N));

    // M = a*I + b*G + c*GG (one fused launch)
    build_M<<<blocks_NN, threads>>>(d_G, d_GG, d_M, working_N, a, b, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    // R_new = R @ M
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_R, working_N, d_M, working_N, &beta, d_R_new, working_N));
    std::swap(d_R, d_R_new);

    // skip G update on last iteration -- not needed
    if(k < iterations - 1){
      // MG = M @ G
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_M, working_N, d_G, working_N, &beta, d_MG, working_N));
      // G_new = MG @ M
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_MG, working_N, d_M, working_N, &beta, d_G_new, working_N));
      std::swap(d_G, d_G_new);
    }
  }

  // Y = X @ R (the second and final rectangular matmul)
  float* Y_working;
  if(transposed){
    CUDA_CHECK(cudaMalloc((void**)&Y_working, working_N * working_M * sizeof(float)));
  } else {
    Y_working = d_output;
  }
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_M, working_N, working_N, &alpha, X, working_M, d_R, working_N, &beta, Y_working, working_M));

  // un-transpose if needed
  if(transposed){
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_N, orig_M, &alpha, Y_working, working_M, &beta, Y_working, working_M, d_output, orig_N));
    CUDA_CHECK(cudaFree(Y_working));
    CUDA_CHECK(cudaFree(d_input_transposed));
  }

  CUDA_CHECK(cudaFree(d_G));
  CUDA_CHECK(cudaFree(d_GG));
  CUDA_CHECK(cudaFree(d_M));
  CUDA_CHECK(cudaFree(d_R));
  CUDA_CHECK(cudaFree(d_R_new));
  CUDA_CHECK(cudaFree(d_MG));
  CUDA_CHECK(cudaFree(d_G_new));
}

void gns_muon_step(float* __restrict__ d_W, float* __restrict__ d_G, float* __restrict__ d_M, float* __restrict__ d_U, int N, int M, float lr, float weight_decay){
  int size = N * M;
  int threads = 256;
  int blocks = (size + threads - 1)/threads;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  mom_update<<<blocks, threads>>>(d_M, d_G, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  int ns_iterations = 5;
  gram_newton_schulz_launch(d_M, d_U, N, M, ns_iterations, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  // performing W←(1−ηλ)W−ηU
  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}
