// u-normuon variant. standalone entry that pulls in the base muon primitives.

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"

__global__ void u_normuon_row_sq_mean_kernel(
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

__global__ void u_normuon_row_ema_update_kernel(
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

__global__ void u_normuon_apply_row_norm_kernel(
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

void u_normuon_row_postprocess(
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

  u_normuon_row_sq_mean_kernel<<<row_blocks, threads>>>(d_U, d_row_mean_sq, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  u_normuon_row_ema_update_kernel<<<row_blocks, threads>>>(d_row_ema, d_row_mean_sq, N, beta2);
  CUDA_CHECK(cudaDeviceSynchronize());

  u_normuon_apply_row_norm_kernel<<<elem_blocks, threads>>>(d_U, d_row_ema, N, M, eps);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_row_mean_sq));
}

void u_normuon_step(
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

  u_normuon_row_postprocess(d_U, d_row_ema, N, M, 0.999f, 1e-8f);
  CUDA_CHECK(cudaDeviceSynchronize());

  float u_norm;
  CUBLAS_CHECK(cublasSnrm2(handle, size, d_U, 1, &u_norm));
  float lr_hat = 0.2f * lr * u_norm / sqrtf((float)M);

  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr_hat, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}
