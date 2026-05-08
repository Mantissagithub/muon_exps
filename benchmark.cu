// #include <bits/stdc++.h> // got a problem running in a1000
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"
#include "gns_muon.cu"

// command to run: nvcc -o benchmark benchmark.cu -lcublas && ./benchmark
// emits machine-parseable lines so benchmark_tui.py can render a rich tui on top.
// run via: uv run benchmark_tui.py

using namespace std;

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

float run_gns_muon_step(int N, int M, int iterations){
  float *d_W, *d_G, *d_Mom, *d_U;
  cudaMalloc(&d_W, N*M*4); cudaMalloc(&d_G, N*M*4);
  cudaMalloc(&d_Mom, N*M*4); cudaMalloc(&d_U, N*M*4);

  gns_muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1);  // warmup

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(int i=0;i<iterations;i++) gns_muon_step(d_W, d_G, d_Mom, d_U, N, M, 1e-3, 0.1);
  cudaEventRecord(stop); cudaEventSynchronize(stop);

  float ms; cudaEventElapsedTime(&ms, start, stop);
  cudaFree(d_W); cudaFree(d_G); cudaFree(d_Mom); cudaFree(d_U);
  return ms / iterations;
}

int main() {
  // chose this shapes from llama 3.1 8b model arch + a few tall ratios for the gram trick
  vector<pair<int,int>> sizes = {
    {1024, 1024},  // ρ=1.0
    {2048, 2048},  // ρ=1.0
    {4096, 4096},  // ρ=1.0
    {4096, 1024},  // ρ=4.0
    {8192, 2048},  // ρ=4.0
    {8192, 1024},  // ρ=8.0
  };

  const int iters = 1000;

  // emit system info up front so the tui can render it
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int rt;
  cudaRuntimeGetVersion(&rt);
  double vram_gib = (double)p.totalGlobalMem / (1024.0*1024.0*1024.0);
  printf("SYS|%s|%d.%d|%.2f|%d|%d\n", p.name, p.major, p.minor, vram_gib, rt, iters);
  fflush(stdout);

  printf("TOTAL|%d\n", (int)sizes.size());
  fflush(stdout);

  int idx = 1;
  for(auto& sz : sizes){
    int N = sz.first, M = sz.second;
    int big = N>M?N:M, small = N>M?M:N;
    float rho = (float)big / (float)small;

    printf("START|%d|%d|%d|%.2f\n", idx, N, M, rho);
    fflush(stdout);

    float v1_ms  = run_muon_step(N, M, iters);
    printf("TIME|%d|v1_ns|%.4f\n", idx, v1_ms);
    fflush(stdout);

    float gns_ms = run_gns_muon_step(N, M, iters);
    printf("TIME|%d|gns|%.4f\n", idx, gns_ms);
    fflush(stdout);

    printf("END|%d\n", idx);
    fflush(stdout);

    idx++;
  }

  printf("DONE\n");
  fflush(stdout);
  return 0;
}
