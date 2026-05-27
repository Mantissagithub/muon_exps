// practical aurora. standalone entry that pulls in the base muon primitives.

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"

__global__ void aurora_momentum_update_kernel(
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

__global__ void aurora_row_norm_kernel(
    const float* __restrict__ X,
    float* __restrict__ row_norm,
    int rows,
    int cols,
    float eps
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float acc = 0.0f;
  for(int col = 0; col < cols; col++){
    float x = X[row + col * rows];
    acc += x * x;
  }
  row_norm[row] = sqrtf(fmaxf(acc, eps * eps));
}

__global__ void aurora_row_sq_kernel(
    const float* __restrict__ X,
    float* __restrict__ row_sq,
    int rows,
    int cols,
    float eps
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float acc = 0.0f;
  for(int col = 0; col < cols; col++){
    float x = X[row + col * rows];
    acc += x * x;
  }
  row_sq[row] = fmaxf(acc, eps * eps);
}

__global__ void aurora_init_diag_kernel(
    float* __restrict__ D,
    const float* __restrict__ row_norm,
    int rows,
    float eps
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  D[row] = 1.0f / fmaxf(row_norm[row], eps);
}

__global__ void aurora_update_diag_kernel(
    float* __restrict__ D,
    const float* __restrict__ row_sq,
    int rows,
    float target_row_sq,
    float pp_beta,
    float eps
){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= rows) return;

  float rsq = fmaxf(row_sq[row], eps * eps);
  D[row] *= powf(target_row_sq / rsq, pp_beta);
}

__global__ void aurora_left_scale_rows_kernel(
    const float* __restrict__ X,
    const float* __restrict__ D,
    float* __restrict__ Y,
    int rows,
    int cols
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = rows * cols;
  if(idx >= size) return;

  int row = idx % rows;
  Y[idx] = D[row] * X[idx];
}

void aurora_polar_launch(
    float* __restrict__ d_input,
    float* __restrict__ d_output,
    int N,
    int M,
    int pp_iterations,
    float pp_beta,
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
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float *d_x0, *d_x_cur, *d_x_next, *d_x_tilde, *d_row_stat, *d_diag;
    CUDA_CHECK(cudaMalloc((void**)&d_x0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x_cur, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x_next, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x_tilde, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_row_stat, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_diag, rows * sizeof(float)));

    if(transposed){
      CUBLAS_CHECK(cublasSgeam(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          M,
          N,
          &alpha,
          d_input,
          N,
          &beta,
          d_x0,
          M,
          d_x0,
          M));
    } else {
      CUDA_CHECK(cudaMemcpy(d_x0, d_input, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_x_cur, d_x0, bytes, cudaMemcpyDeviceToDevice));

    aurora_row_norm_kernel<<<row_blocks, threads>>>(d_x_cur, d_row_stat, rows, cols, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    aurora_init_diag_kernel<<<row_blocks, threads>>>(d_diag, d_row_stat, rows, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    float target_row_sq = (float)cols / (float)rows;
    for(int k = 0; k < pp_iterations; k++){
      aurora_left_scale_rows_kernel<<<elem_blocks, threads>>>(d_x_cur, d_diag, d_x_tilde, rows, cols);
      CUDA_CHECK(cudaDeviceSynchronize());

      newton_schulz_launch(d_x_tilde, d_x_next, rows, cols, 5, handle);
      CUDA_CHECK(cudaDeviceSynchronize());

      if(k < pp_iterations - 1){
        aurora_row_sq_kernel<<<row_blocks, threads>>>(d_x_next, d_row_stat, rows, cols, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        aurora_update_diag_kernel<<<row_blocks, threads>>>(d_diag, d_row_stat, rows, target_row_sq, pp_beta, eps);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* tmp = d_x_cur;
        d_x_cur = d_x_next;
        d_x_next = tmp;
      }
    }

    if(transposed){
      CUBLAS_CHECK(cublasSgeam(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          N,
          M,
          &alpha,
          d_x_next,
          rows,
          &beta,
          d_output,
          N,
          d_output,
          N));
    } else {
      CUDA_CHECK(cudaMemcpy(d_output, d_x_next, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_x0));
    CUDA_CHECK(cudaFree(d_x_cur));
    CUDA_CHECK(cudaFree(d_x_next));
    CUDA_CHECK(cudaFree(d_x_tilde));
    CUDA_CHECK(cudaFree(d_row_stat));
    CUDA_CHECK(cudaFree(d_diag));
  }

  float aspect_scale = sqrtf(fmaxf(1.0f, (float)N / (float)M));
  scale_matrix<<<elem_blocks, threads>>>(d_output, size, aspect_scale);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void aurora_step(
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

  aurora_momentum_update_kernel<<<blocks, threads>>>(d_M, d_G, d_U, size, 0.95f, true);
  CUDA_CHECK(cudaDeviceSynchronize());

  aurora_polar_launch(d_U, d_U, N, M, 2, 0.5f, 1e-7f, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  muon_update_kernel<<<blocks, threads>>>(d_W, d_U, size, lr, weight_decay);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasDestroy(handle));
}
