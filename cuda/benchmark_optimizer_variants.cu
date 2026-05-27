// compile from repo root:
// nvcc -O3 -std=c++17 cuda/benchmark_optimizer_variants.cu -lcublas -o artifacts/bin/benchmark_optimizer_variants

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "muon.cu"
#include "normuon.cu"
#include "u_normuon.cu"
#include "aurora.cu"
#include "riemann_aurora.cu"

enum OptimizerKind { MUON, NORMUON, U_NORMUON, AURORA, RIEMANN_AURORA };

struct Shape {
  int N;
  int M;
};

struct Metrics {
  std::string optimizer;
  int N;
  int M;
  float avg_step_ms;
  float update_fro_norm;
  float row_norm_mean;
  float row_norm_std;
  float row_norm_cv;
  float dead_row_fraction;
  float orthogonality_defect;
  float gradient_alignment;
  float row_norm_min;
  float row_norm_max;
};

__global__ void fill_kernel(float* x, int size, float value){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) x[idx] = value;
}

__global__ void subtract_identity_kernel(float* G, int r){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = r * r;
  if(idx >= size) return;

  int row = idx % r;
  int col = idx / r;
  if(row == col) G[idx] -= 1.0f;
}

const char* optimizer_name(OptimizerKind kind){
  switch(kind){
    case MUON:       return "muon";
    case NORMUON:    return "normuon";
    case U_NORMUON:  return "u_normuon";
    case AURORA:     return "aurora";
    case RIEMANN_AURORA: return "riemann_aurora";
  }
  return "unknown";
}

void generate_anisotropic_gradient(std::vector<float>& h_G, int N, int M, unsigned seed){
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::bernoulli_distribution tiny_row(0.20);

  std::vector<float> row_scale(N);
  for(int row = 0; row < N; row++){
    row_scale[row] = tiny_row(rng) ? 1e-3f : 1.0f;
  }

  for(int col = 0; col < M; col++){
    for(int row = 0; row < N; row++){
      h_G[(size_t)row + (size_t)col * N] = row_scale[row] * normal(rng);
    }
  }
}

void fill_device(float* d_x, int size, float value){
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  fill_kernel<<<blocks, threads>>>(d_x, size, value);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void run_step(
    OptimizerKind kind,
    float* d_W,
    float* d_G,
    float* d_M,
    float* d_U,
    float* d_row_ema,
    int N,
    int M
){
  const float lr = 1e-3f;
  const float weight_decay = 0.1f;

  switch(kind){
    case MUON:
      muon_step(d_W, d_G, d_M, d_U, N, M, lr, weight_decay);
      break;
    case NORMUON:
      normuon_step(d_W, d_G, d_M, d_U, d_row_ema, N, M, lr, weight_decay);
      break;
    case U_NORMUON:
      u_normuon_step(d_W, d_G, d_M, d_U, d_row_ema, N, M, lr, weight_decay);
      break;
    case AURORA:
      aurora_step(d_W, d_G, d_M, d_U, N, M, lr, weight_decay);
      break;
    case RIEMANN_AURORA:
      riemann_aurora_step(d_W, d_G, d_M, d_U, N, M, lr, weight_decay);
      break;
  }
}

float compute_orthogonality_defect(float* d_U, int N, int M, cublasHandle_t handle){
  int r = std::min(N, M);
  float* d_gram;
  CUDA_CHECK(cudaMalloc((void**)&d_gram, (size_t)r * r * sizeof(float)));

  float alpha = 1.0f;
  float beta = 0.0f;
  if(N >= M){
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        M,
        M,
        N,
        &alpha,
        d_U,
        N,
        d_U,
        N,
        &beta,
        d_gram,
        M));
  } else {
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        N,
        N,
        M,
        &alpha,
        d_U,
        N,
        d_U,
        N,
        &beta,
        d_gram,
        N));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  int threads = 256;
  int blocks = (r * r + threads - 1) / threads;
  subtract_identity_kernel<<<blocks, threads>>>(d_gram, r);
  CUDA_CHECK(cudaDeviceSynchronize());

  float defect_norm;
  CUBLAS_CHECK(cublasSnrm2(handle, r * r, d_gram, 1, &defect_norm));
  CUDA_CHECK(cudaFree(d_gram));

  return defect_norm / std::sqrt((float)r);
}

Metrics compute_metrics(
    OptimizerKind kind,
    int N,
    int M,
    float avg_step_ms,
    const std::vector<float>& h_G,
    float* d_G,
    float* d_U,
    cublasHandle_t handle
){
  int size = N * M;
  size_t bytes = (size_t)size * sizeof(float);

  Metrics out{};
  out.optimizer = optimizer_name(kind);
  out.N = N;
  out.M = M;
  out.avg_step_ms = avg_step_ms;

  CUBLAS_CHECK(cublasSnrm2(handle, size, d_U, 1, &out.update_fro_norm));

  float grad_norm;
  float dot;
  CUBLAS_CHECK(cublasSnrm2(handle, size, d_G, 1, &grad_norm));
  CUBLAS_CHECK(cublasSdot(handle, size, d_G, 1, d_U, 1, &dot));
  out.gradient_alignment = dot / std::max(grad_norm * out.update_fro_norm, 1e-20f);

  std::vector<float> h_U(size);
  CUDA_CHECK(cudaMemcpy(h_U.data(), d_U, bytes, cudaMemcpyDeviceToHost));

  std::vector<double> row_norms(N);
  double sum = 0.0;
  double sumsq = 0.0;
  out.row_norm_min = std::numeric_limits<float>::infinity();
  out.row_norm_max = 0.0f;

  for(int row = 0; row < N; row++){
    double acc = 0.0;
    for(int col = 0; col < M; col++){
      double x = h_U[(size_t)row + (size_t)col * N];
      acc += x * x;
    }
    double rn = std::sqrt(acc);
    row_norms[row] = rn;
    sum += rn;
    sumsq += rn * rn;
    out.row_norm_min = std::min(out.row_norm_min, (float)rn);
    out.row_norm_max = std::max(out.row_norm_max, (float)rn);
  }

  double mean = sum / (double)N;
  double var = std::max(0.0, sumsq / (double)N - mean * mean);
  out.row_norm_mean = (float)mean;
  out.row_norm_std = (float)std::sqrt(var);
  out.row_norm_cv = out.row_norm_std / std::max(out.row_norm_mean, 1e-20f);

  int dead_rows = 0;
  double dead_thresh = 0.01 * mean;
  for(double rn : row_norms){
    if(rn < dead_thresh) dead_rows++;
  }
  out.dead_row_fraction = (float)dead_rows / (float)N;

  out.orthogonality_defect = compute_orthogonality_defect(d_U, N, M, handle);
  return out;
}

Metrics run_optimizer_once(OptimizerKind kind, int N, int M, const std::vector<float>& h_G){
  int size = N * M;
  size_t bytes = (size_t)size * sizeof(float);
  float *d_W, *d_G, *d_M, *d_U, *d_row_ema = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_W, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_G, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_M, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_U, bytes));

  if(kind == NORMUON || kind == U_NORMUON){
    CUDA_CHECK(cudaMalloc((void**)&d_row_ema, N * sizeof(float)));
  }

  CUDA_CHECK(cudaMemset(d_W, 0, bytes));
  CUDA_CHECK(cudaMemset(d_M, 0, bytes));
  CUDA_CHECK(cudaMemset(d_U, 0, bytes));
  CUDA_CHECK(cudaMemcpy(d_G, h_G.data(), bytes, cudaMemcpyHostToDevice));

  if(d_row_ema){
    // start at 1 to avoid first-step row-norm explosions from tiny EMA values.
    fill_device(d_row_ema, N, 1.0f);
  }

  int warmup_steps = 2;
  int timed_steps = 5;

  for(int i = 0; i < warmup_steps; i++){
    run_step(kind, d_W, d_G, d_M, d_U, d_row_ema, N, M);
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for(int i = 0; i < timed_steps; i++){
    run_step(kind, d_W, d_G, d_M, d_U, d_row_ema, N, M);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_step_ms = elapsed_ms / (float)timed_steps;

  cublasHandle_t metrics_handle;
  CUBLAS_CHECK(cublasCreate(&metrics_handle));
  Metrics metrics = compute_metrics(kind, N, M, avg_step_ms, h_G, d_G, d_U, metrics_handle);
  CUBLAS_CHECK(cublasDestroy(metrics_handle));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_G));
  CUDA_CHECK(cudaFree(d_M));
  CUDA_CHECK(cudaFree(d_U));
  if(d_row_ema) CUDA_CHECK(cudaFree(d_row_ema));

  return metrics;
}

std::vector<Metrics> benchmark_shape(int N, int M){
  std::vector<float> h_G((size_t)N * M);
  unsigned seed = 1337u + (unsigned)N * 17u + (unsigned)M * 31u;
  generate_anisotropic_gradient(h_G, N, M, seed);

  std::vector<OptimizerKind> optimizers = {MUON, NORMUON, U_NORMUON, AURORA, RIEMANN_AURORA};
  std::vector<Metrics> rows;
  for(OptimizerKind kind : optimizers){
    std::cerr << "PROGRESS," << optimizer_name(kind) << ',' << N << ',' << M << std::endl;
    rows.push_back(run_optimizer_once(kind, N, M, h_G));
  }
  return rows;
}

void write_header(std::ostream& out){
  out << "optimizer,N,M,avg_step_ms,update_fro_norm,row_norm_mean,row_norm_std,row_norm_cv,"
      << "dead_row_fraction,orthogonality_defect,gradient_alignment,row_norm_min,row_norm_max\n";
}

void write_row(std::ostream& out, const Metrics& m){
  out << m.optimizer << ','
      << m.N << ','
      << m.M << ','
      << m.avg_step_ms << ','
      << m.update_fro_norm << ','
      << m.row_norm_mean << ','
      << m.row_norm_std << ','
      << m.row_norm_cv << ','
      << m.dead_row_fraction << ','
      << m.orthogonality_defect << ','
      << m.gradient_alignment << ','
      << m.row_norm_min << ','
      << m.row_norm_max << '\n';
}

int main(){
  std::vector<Shape> shapes = {
    {512, 128},
    {1024, 256},
    {2048, 512},
    {1024, 1024},
    {512, 2048},
  };

  std::ofstream csv("benchmark_results.csv");
  if(!csv){
    std::fprintf(stderr, "failed to open benchmark_results.csv\n");
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6);
  csv << std::fixed << std::setprecision(6);

  write_header(std::cout);
  write_header(csv);

  for(const Shape& shape : shapes){
    std::vector<Metrics> rows = benchmark_shape(shape.N, shape.M);
    for(const Metrics& row : rows){
      write_row(std::cout, row);
      std::cout.flush();
      write_row(csv, row);
      csv.flush();
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
