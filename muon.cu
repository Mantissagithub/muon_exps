#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

using namespace std;

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUBLAS_CHECK(err) cublasAssert((err), __FILE__, __LINE__)
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"CUBLASassert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

__global__ void mom_update(float* __restrict__ momentum, float* __restrict__ grad, int N, int M){
  const float beta = 0.95f;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N*M) momentum[idx] = beta * momentum[idx] + (1.0f - beta) * grad[idx];
}

__global__ void combine_matrices(float* __restrict__ A, float* __restrict__ AA, float* __restrict__ B, int size, float b, float c){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx<size) B[idx] = b*A[idx] + c*AA[idx];
}

__global__ void combine_matrices_v2(float* __restrict__ X, float* __restrict__ BX, float* __restrict__ C, int size, float a){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < size) C[idx] = a*X[idx] + BX[idx];
}

__global__ void scale_matrix(float* __restrict__ X, int size, float scale){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size) X[idx] = X[idx] * scale;
}

__global__ void muon_update_kernel(float* __restrict__ W, float* __restrict__ U, int size, float lr, float weight_decay){
  // performing W←(1−ηλ)W−ηU
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    float w = W[idx];
    w *= (1.0f - lr * weight_decay);
    w -= lr * U[idx];
    W[idx] = w;
  }
}

void newton_schulz_launch(float* __restrict__ d_input, float* __restrict__ d_output, int N, int M, int iterations, cublasHandle_t handle){
  const float a = 3.4445f;
  const float b = -4.7750f;
  const float c = 2.0315f;
  float *d_input_transpose, *d_xxt, *d_AA, *d_B, *d_BX;

  int orig_N = N;
  int orig_M = M;

  float alpha = 1.0f, beta = 0.0f;

  int len = N*M;
  float norm;
  CUBLAS_CHECK(cublasSnrm2(handle, len, d_input, 1, &norm));

  const float eps = 1e-6f;
  float inv_scale = 1.0f / (norm + eps);
  int threads = 256;
  int blocks = (len + threads - 1)/threads;
  scale_matrix<<<blocks, threads>>>(d_input, len, inv_scale);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool transposed = false;
  int working_N = N, working_M = M;

  size_t size_input = orig_N * orig_M * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_input_transpose, size_input));

  if(N > M){
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_M, orig_N, &alpha, d_input, orig_N, &beta, d_input, orig_N, d_input_transpose, orig_M));

    working_N = M;
    working_M = N;

    transposed = true;
  }

  size_t size_xxt = working_N * working_N * sizeof(float);
  size_t size_AA = working_N * working_N * sizeof(float);
  size_t size_B = working_N * working_N * sizeof(float);
  size_t size_working = working_N * working_M * sizeof(float);

  CUDA_CHECK(cudaMalloc((void**)&d_xxt, size_xxt));
  CUDA_CHECK(cudaMalloc((void**)&d_AA, size_AA));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void**)&d_BX, size_working));

  float* X = transposed ? d_input_transpose : d_input;
  float* Y = d_output;

  for(int i=0;i<iterations;i++){
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, working_M, working_N, &alpha, X, working_N, &beta, X, working_N, d_input_transpose, working_M));

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_M, &alpha, X, working_N, d_input_transpose, working_M, &beta, d_xxt, working_N));

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_N, working_N, &alpha, d_xxt, working_N, d_xxt, working_N, &beta, d_AA, working_N));

    int threads = 256;
    int blocks_A = (working_N * working_N + threads - 1) / threads;
    combine_matrices<<<blocks_A, threads>>>(d_xxt, d_AA, d_B, working_N * working_N, b, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, working_N, working_M, working_N, &alpha, d_B, working_N, X, working_N, &beta, d_BX, working_N));

    int blocks_X = (working_N * working_M + threads - 1) / threads;
    combine_matrices_v2<<<blocks_X, threads>>>(X, d_BX, Y, working_N * working_M, a);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(i<iterations-1){
      float* tmp = X;
      X = Y;
      Y = tmp;
    }
  }

  if(transposed){
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig_M, orig_N, &alpha, X, working_N, &beta, X, working_N, d_output, orig_N));
  } else {
    if(X != d_output){
      CUDA_CHECK(cudaMemcpy(d_output, X, size_input, cudaMemcpyDeviceToDevice));
    }
  }

  CUDA_CHECK(cudaFree(d_input_transpose));
  CUDA_CHECK(cudaFree(d_xxt));
  CUDA_CHECK(cudaFree(d_AA));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_BX));
}

void muon_step(float* __restrict__ d_W, float* __restrict__ d_G, float* __restrict__ d_M, float* __restrict__ d_U, int N, int M, float lr, float weight_decay){
  int size = N * M;
  int threads = 256;
  int blocks = (size + threads - 1)/threads;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  mom_update<<<blocks, threads>>>(d_M, d_G, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  int ns_iterations = 5;
  newton_schulz_launch(d_M, d_U, N, M, ns_iterations, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}