#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "muon.cu"

// command to run: nvcc -o benchmark benchmark.cu -lcublas && ./benchmark

using namespace std;

void benchmark_muon_step(int N, int M, int iterations=1000) {
    float *d_W, *d_G, *d_M, *d_U;
    cudaMalloc(&d_W, N*M*4); cudaMalloc(&d_G, N*M*4);
    cudaMalloc(&d_M, N*M*4); cudaMalloc(&d_U, N*M*4);

    muon_step(d_W, d_G, d_M, d_U, N, M, 1e-3, 0.1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < iterations; i++) {
        muon_step(d_W, d_G, d_M, d_U, N, M, 1e-3, 0.1);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("Your Muon: %.2f ms/step (N=%d, M=%d)\n", ms/iterations, N, M);

    cudaFree(d_W); cudaFree(d_G); cudaFree(d_M); cudaFree(d_U);
}

int main() {
  // chose this shapes from llama 3.1 8b model arch
    vector<pair<int,int>> sizes = {
        {4096, 4096},
        {4096, 11008},
        {11008, 4096}
    };

    for(auto& sz : sizes) {
        benchmark_muon_step(sz.first, sz.second);
    }

    return 0;
}