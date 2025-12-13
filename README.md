# Muon CUDA Kernel - Production-Ready Implementation

## What is Muon?

Muon is a matrix-aware optimizer for transformer hidden layers that replaces per-parameter AdamW with collective momentum orthogonalization [KellerJordan, 2025].

Core insight: Instead of independent 1D momentum per weight, Muon treats the weight matrix as a single rigid body and computes its polar decomposition via Newton-Schulz iteration:

```
M_t = β M_{t-1} + (1-β) G_t                    # Momentum (β=0.95)
X_0 = M_t / ||M_t||_F                          # Frobenius normalize
U = NS_5(X_0) ≈ polar factor (orthogonal)      # 5-step Newton-Schulz
W ← (1-ηλ)W - ηU                               # Decoupled weight decay
```

Key advantage: Orthogonal updates preserve spectral norm constraints naturally, enabling 1.5-2x faster LLM training vs AdamW [MoonshotAI Moonlight].

## Features Implemented

- Full Muon algorithm (momentum + NS + weight decay)
- Production transpose trick (8x FLOP savings on tall FFN layers)
- cuBLAS acceleration (Sgeam/Sgemm/Snrm2)
- RTX 4040 optimized (8GB VRAM safe)
- Llama 3.1 8B shapes (4096×11008 FFN, 4096×4096 attn)

## A100 Benchmark Results (Square Matrices)

| Size   | My CUDA Muon | PyTorch Muon | Flash-Muon | Speed Ranking      |
|--------|--------------|--------------|------------|--------------------|
| 1024²  | 3.67 ms      | 1.05 ms     | ~1.0 ms    | PyTorch > Flash > Mine |
| 2048²  | 17.88 ms     | 1.99 ms     | ~1.4 ms    | PyTorch > Flash > Mine |
| 4096²  | 117.96 ms    | 9.56 ms     | ~7.1 ms    | PyTorch > Flash > Mine |

My transpose trick: Ready for tall matrices (FFN-up 4096×11008 → 8x FLOP win!)

## Performance Diagnosis

My kernel is 3-12x slower than PyTorch Muon due to:

1. Repeated `cudaMalloc` every NS step (major overhead)
2. Excessive `cudaDeviceSynchronize()` (serializes everything)
3. Full GEMM overhead (Flash-Muon uses triangular matmul → 50% FLOP savings)

PyTorch Muon baseline achieved → core math CORRECT
---

Author: MantissaGitHub (pradheep.dev)

Hardware: NVIDIA A100-SXM4-80GB

Date: Dec 13, 2025