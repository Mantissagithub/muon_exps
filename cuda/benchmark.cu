// #include <bits/stdc++.h> // got a problem running in a100
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"
#include "normuon.cu"
#include "gns_muon.cu"

// direct compile example:
// nvcc -o artifacts/bin/benchmark cuda/benchmark.cu -lcublas -arch=sm_89 && ./artifacts/bin/benchmark
// emits machine-parseable lines so scripts/benchmark_tui.py can render a rich tui on top.
// run via: uv run scripts/benchmark_tui.py

using namespace std;

static const char* mode_name(GNSMode m){
  switch(m){
    case GNS_QUINTIC:              return "quintic";
    case GNS_POLAR:                return "polar";
    case GNS_POLAR_RESTART:        return "polar_restart";
    case GNS_POLAR_RESTART_SYRK:   return "polar_restart_syrk";
  }
  return "?";
}

float run_muon_step(int N, int M, int iterations){
  float *d_W, *d_G, *d_Mom, *d_U;
  cudaMalloc(&d_W, N*M*4); cudaMalloc(&d_G, N*M*4);
  cudaMalloc(&d_Mom, N*M*4); cudaMalloc(&d_U, N*M*4);

  muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1);  // warmup

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(int i=0;i<iterations;i++) muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1);
  cudaEventRecord(stop); cudaEventSynchronize(stop);

  float ms; cudaEventElapsedTime(&ms, start, stop);
  cudaFree(d_W); cudaFree(d_G); cudaFree(d_Mom); cudaFree(d_U);
  return ms / iterations;
}

float run_normuon_step(int N, int M, int iterations){
  float *d_W, *d_G, *d_Mom, *d_U, *d_row_ema;
  cudaMalloc(&d_W, N*M*4); cudaMalloc(&d_G, N*M*4);
  cudaMalloc(&d_Mom, N*M*4); cudaMalloc(&d_U, N*M*4);
  cudaMalloc(&d_row_ema, N*4);
  cudaMemset(d_row_ema, 0, N*4);

  normuon_step(d_W, d_G, d_Mom, d_U, d_row_ema, N, M, 1e-3, 0.1);  // warmup

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(int i=0;i<iterations;i++) normuon_step(d_W, d_G, d_Mom, d_U, d_row_ema, N, M, 1e-3, 0.1);
  cudaEventRecord(stop); cudaEventSynchronize(stop);

  float ms; cudaEventElapsedTime(&ms, start, stop);
  cudaFree(d_W); cudaFree(d_G); cudaFree(d_Mom); cudaFree(d_U); cudaFree(d_row_ema);
  return ms / iterations;
}

float run_gns_muon_step(int N, int M, int iterations, GNSMode mode){
  float *d_W, *d_G, *d_Mom, *d_U;
  cudaMalloc(&d_W, N*M*4); cudaMalloc(&d_G, N*M*4);
  cudaMalloc(&d_Mom, N*M*4); cudaMalloc(&d_U, N*M*4);

  gns_muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1, mode);  // warmup

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(int i=0;i<iterations;i++) gns_muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1, mode);
  cudaEventRecord(stop); cudaEventSynchronize(stop);

  float ms; cudaEventElapsedTime(&ms, start, stop);
  cudaFree(d_W); cudaFree(d_G); cudaFree(d_Mom); cudaFree(d_U);
  return ms / iterations;
}

// ---- correctness checks ----------------------------------------------------
// Two probes:
//   * verify_against_v1 -- pointwise diff vs v1 NS. Only meaningful for GNS_QUINTIC
//     (which shares (a,b,c) with v1). Polar/restart variants use different per-iter
//     coefficients so their pointwise output diverges from v1 by design.
//   * verify_ortho_*   -- ‖Y^T Y − I_r‖_F / sqrt(r). Operationally meaningful for
//     every mode: tells us whether the launch actually orthogonalized.

// computes orthogonality residual ‖Y^T Y − I_r‖_F / sqrt(r) where Y is N×M col-major fp32
// and r = min(N, M). Reuses the caller's cublas handle.
float ortho_residual(float* d_Y, int N, int M, cublasHandle_t handle){
  int r = (N < M) ? N : M;
  float *d_G;
  cudaMalloc(&d_G, (size_t)r * r * sizeof(float));
  float alpha = 1.0f, beta = 0.0f;
  if(N >= M){
    // G = Y^T Y, M×M
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, M, N, &alpha, d_Y, N, d_Y, N, &beta, d_G, M);
  } else {
    // G = Y Y^T, N×N
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, M, &alpha, d_Y, N, d_Y, N, &beta, d_G, N);
  }
  cudaDeviceSynchronize();
  float* h_G = (float*)malloc((size_t)r * r * sizeof(float));
  cudaMemcpy(h_G, d_G, (size_t)r * r * sizeof(float), cudaMemcpyDeviceToHost);
  double sumsq = 0.0;
  for(int j = 0; j < r; j++){
    for(int i = 0; i < r; i++){
      double v = h_G[(size_t)i + (size_t)j * r];
      if(i == j) v -= 1.0;
      sumsq += v * v;
    }
  }
  free(h_G);
  cudaFree(d_G);
  return (float)sqrt(sumsq / (double)r);
}

void verify_against_v1(int N, int M, GNSMode mode, float threshold){
  size_t bytes = (size_t)N * M * sizeof(float);
  float* h_in       = (float*)malloc(bytes);
  float* h_out_v1   = (float*)malloc(bytes);
  float* h_out_gns  = (float*)malloc(bytes);
  for(int i=0;i<N*M;i++) h_in[i] = ((float)rand()/(float)RAND_MAX) * 2.0f - 1.0f;

  float *d_in_v1, *d_in_gns, *d_out_v1, *d_out_gns;
  cudaMalloc(&d_in_v1, bytes); cudaMalloc(&d_in_gns, bytes);
  cudaMalloc(&d_out_v1, bytes); cudaMalloc(&d_out_gns, bytes);
  cudaMemcpy(d_in_v1, h_in, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_gns, h_in, bytes, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  newton_schulz_launch(d_in_v1, d_out_v1, N, M, 5, handle);
  cudaDeviceSynchronize();
  gram_newton_schulz_launch(d_in_gns, d_out_gns, N, M, 5, mode, handle);
  cudaDeviceSynchronize();
  cublasDestroy(handle);

  cudaMemcpy(h_out_v1, d_out_v1, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_gns, d_out_gns, bytes, cudaMemcpyDeviceToHost);

  float max_diff = 0.0f;
  for(int i=0;i<N*M;i++){
    float d = fabsf(h_out_v1[i] - h_out_gns[i]);
    if(d > max_diff) max_diff = d;
  }

  const char* status = (max_diff < threshold) ? "ok" : "fail";
  printf("VERIFY|%d|%d|%s|%.6e|%s\n", N, M, mode_name(mode), max_diff, status);
  fflush(stdout);

  free(h_in); free(h_out_v1); free(h_out_gns);
  cudaFree(d_in_v1); cudaFree(d_in_gns); cudaFree(d_out_v1); cudaFree(d_out_gns);
}

// orthogonality probe for v1 NS
void verify_ortho_v1(int N, int M, float threshold){
  size_t bytes = (size_t)N * M * sizeof(float);
  float* h_in = (float*)malloc(bytes);
  for(int i=0;i<N*M;i++) h_in[i] = ((float)rand()/(float)RAND_MAX) * 2.0f - 1.0f;
  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
  cublasHandle_t handle; cublasCreate(&handle);
  newton_schulz_launch(d_in, d_out, N, M, 5, handle);
  cudaDeviceSynchronize();
  float res = ortho_residual(d_out, N, M, handle);
  cublasDestroy(handle);
  const char* status = (res < threshold) ? "ok" : "fail";
  printf("VERIFY|%d|%d|v1_ortho|%.6e|%s\n", N, M, res, status);
  fflush(stdout);
  free(h_in); cudaFree(d_in); cudaFree(d_out);
}

// orthogonality probe for each GNS fp32 mode
void verify_ortho_gns(int N, int M, GNSMode mode, float threshold){
  size_t bytes = (size_t)N * M * sizeof(float);
  float* h_in = (float*)malloc(bytes);
  for(int i=0;i<N*M;i++) h_in[i] = ((float)rand()/(float)RAND_MAX) * 2.0f - 1.0f;
  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
  cublasHandle_t handle; cublasCreate(&handle);
  gram_newton_schulz_launch(d_in, d_out, N, M, 5, mode, handle);
  cudaDeviceSynchronize();
  float res = ortho_residual(d_out, N, M, handle);
  cublasDestroy(handle);
  const char* status = (res < threshold) ? "ok" : "fail";
  char label[32]; snprintf(label, sizeof(label), "%s_ortho", mode_name(mode));
  printf("VERIFY|%d|%d|%s|%.6e|%s\n", N, M, label, res, status);
  fflush(stdout);
  free(h_in); cudaFree(d_in); cudaFree(d_out);
}

// orthogonality probe for the fp16 path -- runs the fp16 launch, casts output to fp32, then checks
void verify_ortho_fp16(int N, int M, float threshold){
  size_t bytes_h = (size_t)N * M * sizeof(__half);
  size_t bytes_f = (size_t)N * M * sizeof(float);
  float*  h_in   = (float*) malloc(bytes_f);
  __half* h_in_h = (__half*)malloc(bytes_h);
  for(int i=0;i<N*M;i++){
    h_in[i] = ((float)rand()/(float)RAND_MAX) * 2.0f - 1.0f;
    h_in_h[i] = __float2half(h_in[i]);
  }
  __half *d_in_h, *d_out_h;
  float* d_out_f;
  cudaMalloc(&d_in_h, bytes_h); cudaMalloc(&d_out_h, bytes_h); cudaMalloc(&d_out_f, bytes_f);
  cudaMemcpy(d_in_h, h_in_h, bytes_h, cudaMemcpyHostToDevice);
  cublasHandle_t handle; cublasCreate(&handle);
  gram_newton_schulz_launch_fp16(d_in_h, d_out_h, N, M, 5, handle);
  cudaDeviceSynchronize();
  int threads = 256;
  int blocks = (N*M + threads - 1) / threads;
  half_to_float_kernel<<<blocks, threads>>>(d_out_h, d_out_f, N*M);
  cudaDeviceSynchronize();
  float res = ortho_residual(d_out_f, N, M, handle);
  cublasDestroy(handle);
  const char* status = (res < threshold) ? "ok" : "fail";
  printf("VERIFY|%d|%d|fp16_ortho|%.6e|%s\n", N, M, res, status);
  fflush(stdout);
  free(h_in); free(h_in_h); cudaFree(d_in_h); cudaFree(d_out_h); cudaFree(d_out_f);
}

int main() {
  // chose this shapes from llama 3.1 8b model arch + a few tall ratios for the gram trick.
  // (4096,4096) and (8192,2048) are dropped here because at iters=1000 they take ~20+ minutes each
  // on the dev machine (RTX 4060 Laptop, 7.62 GiB). Bring them back when running on bigger silicon.
  // (2048,1024) replaces them as a cheap mid-rho check.
  vector<pair<int,int>> sizes = {
    {1024, 1024},  // ρ=1.0
    {2048, 2048},  // ρ=1.0
    {2048, 1024},  // ρ=2.0
    {4096, 1024},  // ρ=4.0
    {8192, 1024},  // ρ=8.0
  };

  const int iters = 1000;
  srand(42);

  // emit system info up front so the tui can render it
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int rt;
  cudaRuntimeGetVersion(&rt);
  double vram_gib = (double)p.totalGlobalMem / (1024.0*1024.0*1024.0);
  printf("SYS|%s|%d.%d|%.2f|%d|%d\n", p.name, p.major, p.minor, vram_gib, rt, iters);
  fflush(stdout);

  // ---- verify pass ---------------------------------------------------------
  // structural check for quintic + orthogonality residual for every mode
  printf("VERIFY_START\n"); fflush(stdout);
  vector<pair<int,int>> verify_shapes = { {1024,1024}, {2048,2048}, {4096,1024}, {8192,2048} };
  GNSMode modes[] = {GNS_QUINTIC, GNS_POLAR, GNS_POLAR_RESTART, GNS_POLAR_RESTART_SYRK};
  const float ortho_thresh = 5e-1f;  // loose -- 5 NS iters under Frobenius normalization don't fully saturate
  for(auto& sz : verify_shapes){
    verify_against_v1(sz.first, sz.second, GNS_QUINTIC, 1e-3f);  // structural: gram-form == direct-form when (a,b,c) match
    verify_ortho_v1(sz.first, sz.second, ortho_thresh);
    for(auto m : modes) verify_ortho_gns(sz.first, sz.second, m, ortho_thresh);
    verify_ortho_fp16(sz.first, sz.second, ortho_thresh);
  }
  printf("VERIFY_END\n"); fflush(stdout);

  // ---- benchmark grid ------------------------------------------------------
  printf("TOTAL|%d\n", (int)sizes.size());
  fflush(stdout);

  int idx = 1;
  for(auto& sz : sizes){
    int N = sz.first, M = sz.second;
    int big = N>M?N:M, small = N>M?M:N;
    float rho = (float)big / (float)small;

    printf("START|%d|%d|%d|%.2f\n", idx, N, M, rho);
    fflush(stdout);

    float v1_ms = run_muon_step(N, M, iters);
    printf("TIME|%d|v1_ns|%.4f\n", idx, v1_ms); fflush(stdout);

    float normuon_ms = run_normuon_step(N, M, iters);
    printf("TIME|%d|normuon|%.4f\n", idx, normuon_ms); fflush(stdout);

    for(auto m : modes){
      float ms = run_gns_muon_step(N, M, iters, m);
      printf("TIME|%d|%s|%.4f\n", idx, mode_name(m), ms); fflush(stdout);
    }

    printf("END|%d\n", idx); fflush(stdout);
    idx++;
  }

  printf("DONE\n");
  fflush(stdout);
  return 0;
}
