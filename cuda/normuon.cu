// row-normalized muon variant. assumes muon.cu was already #include'd
// so CUDA_CHECK / CUBLAS_CHECK / mom_update / newton_schulz_launch /
// muon_update_kernel are already in scope.

__global__ void row_sq_mean_kernel(
    const float* __restrict__ U,
    float* __restrict__ row_mean_sq,
    int N,
    int M
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= N) return;

  float acc = 0.0f;
  for(int col = 0; col < M; col++){
    float x = U[row + col * N];
    acc += x * x;
  }
  row_mean_sq[row] = acc / (float)M;
}

__global__ void row_ema_update_kernel(
    float* __restrict__ row_ema,
    const float* __restrict__ row_mean_sq,
    int N,
    float beta2
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < N){
    row_ema[row] = beta2 * row_ema[row] + (1.0f - beta2) * row_mean_sq[row];
  }
}

__global__ void apply_row_norm_kernel(
    float* __restrict__ U,
    const float* __restrict__ row_ema,
    int N,
    int M,
    float eps
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = N * M;
  if(idx >= size) return;

  int row = idx % N;
  float denom = sqrtf(row_ema[row] + eps);
  U[idx] /= denom;
}

void normuon_row_postprocess(
    float* __restrict__ d_U,
    float* __restrict__ d_row_ema,
    int N,
    int M,
    float beta2,
    float eps
){
  float* d_row_mean_sq;
  CUDA_CHECK(cudaMalloc((void**)&d_row_mean_sq, N * sizeof(float)));

  int threads = 256;
  int row_blocks = (N + threads - 1) / threads;
  int elem_blocks = (N * M + threads - 1) / threads;

  row_sq_mean_kernel<<<row_blocks, threads>>>(d_U, d_row_mean_sq, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  row_ema_update_kernel<<<row_blocks, threads>>>(d_row_ema, d_row_mean_sq, N, beta2);
  CUDA_CHECK(cudaDeviceSynchronize());

  apply_row_norm_kernel<<<elem_blocks, threads>>>(d_U, d_row_ema, N, M, eps);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_row_mean_sq));
}

void normuon_step(
    float* __restrict__ d_W,
    float* __restrict__ d_G,
    float* __restrict__ d_M,
    float* __restrict__ d_U,
    float* __restrict__ d_row_ema,
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

  mom_update<<<blocks, threads>>>(d_M, d_G, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  int ns_iterations = 5;
  newton_schulz_launch(d_M, d_U, N, M, ns_iterations, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  normuon_row_postprocess(d_U, d_row_ema, N, M, 0.999f, 1e-8f);
  CUDA_CHECK(cudaDeviceSynchronize());

  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}
