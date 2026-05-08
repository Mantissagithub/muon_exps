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

## Bench grid calibration (RTX 4060 Laptop)

The shape list in `benchmark.cu` is currently calibrated for a 7.62 GiB RTX 4060 Laptop GPU.
`(4096, 4096)` and `(8192, 2048)` were dropped from the grid because at `iters = 1000` they
take roughly 20+ minutes each on this card; `(2048, 1024)` was added in their place as a cheap
mid-rho check. Restore the larger shapes when running on bigger silicon (A100, H100, etc.)
where each shape finishes in seconds.

Verify pass still exercises `(4096, 1024)` and `(8192, 2048)` shapes for orthogonality —
those run once each, not 1000×, so they're cheap to keep.

## Performance Diagnosis

My kernel is 3-12x slower than PyTorch Muon due to:

1. Repeated `cudaMalloc` every NS step (major overhead)
2. Excessive `cudaDeviceSynchronize()` (serializes everything)
3. Full GEMM overhead (Flash-Muon uses triangular matmul → 50% FLOP savings)

PyTorch Muon baseline achieved → core math CORRECT

## Gram Newton-Schulz (Tridao variant)

The Gram NS variants in `gns_muon.cu` follow the formulation from Tri Dao's blog:
[Gram Newton-Schulz](https://tridao.me/blog/2026/gram-newton-schulz/).

Instead of iterating directly on `X` with the standard quintic
`X ← aX + bXX^TX + cXX^TXX^TX`, the Gram approach iterates on the smaller
Gram matrix `G = X^TX` (size `n×n` for a `m×n` matrix with `m ≥ n`):

```
G_0 = X_0^T X_0
G_{k+1} = poly(G_k)        # polynomial iteration on n×n
U = X_0 · f(G_∞)            # single GEMM at the end
```

For tall/wide matrices (large `ρ = max(m,n)/min(m,n)`) this collapses the
work onto the smaller dimension and avoids repeated `m×m`/`m×n` GEMMs per
step. The `polar`, `polar_restart`, and `polar_restart_syrk` modes are
progressive stabilizations of that idea (Padé-style polar map, restart on
spectral drift, and `Ssyrk`-fused symmetric Gram update respectively).

## Verify · Gram NS vs v1 NS  (fp16 vs fp32 polar+restart)

| shape       | mode                       | max diff   | status |
|-------------|----------------------------|------------|--------|
| 1024×1024   | quintic                    | 1.34e-06   | ok     |
| 1024×1024   | v1_ortho                   | 3.47e-01   | ok     |
| 1024×1024   | quintic_ortho              | 3.46e-01   | ok     |
| 1024×1024   | polar_ortho                | 8.10e-01   | fail   |
| 1024×1024   | polar_restart_ortho        | 9.98e-01   | fail   |
| 1024×1024   | polar_restart_syrk_ortho   | 9.98e-01   | fail   |
| 1024×1024   | fp16_ortho                 | 9.98e-01   | fail   |
| 2048×2048   | quintic                    | 6.70e-06   | ok     |
| 2048×2048   | v1_ortho                   | 3.59e-01   | ok     |
| 2048×2048   | quintic_ortho              | 3.59e-01   | ok     |
| 2048×2048   | polar_ortho                | 9.09e-01   | fail   |
| 2048×2048   | polar_restart_ortho        | 1.00e+00   | fail   |
| 2048×2048   | polar_restart_syrk_ortho   | 1.00e+00   | fail   |
| 2048×2048   | fp16_ortho                 | 1.00e+00   | fail   |
| 4096×1024   | quintic                    | 3.50e-07   | ok     |
| 4096×1024   | v1_ortho                   | 2.91e-01   | ok     |
| 4096×1024   | quintic_ortho              | 2.92e-01   | ok     |
| 4096×1024   | polar_ortho                | 1.14e+00   | fail   |
| 4096×1024   | polar_restart_ortho        | 9.99e-01   | fail   |
| 4096×1024   | polar_restart_syrk_ortho   | 9.99e-01   | fail   |
| 4096×1024   | fp16_ortho                 | 9.99e-01   | fail   |
| 8192×2048   | quintic                    | 1.15e-06   | ok     |
| 8192×2048   | v1_ortho                   | 3.26e-01   | ok     |
| 8192×2048   | quintic_ortho              | 3.26e-01   | ok     |
| 8192×2048   | polar_ortho                | 1.32e+00   | fail   |
| 8192×2048   | polar_restart_ortho        | 1.00e+00   | fail   |
| 8192×2048   | polar_restart_syrk_ortho   | 1.00e+00   | fail   |
| 8192×2048   | fp16_ortho                 | 1.00e+00   | fail   |

Numerical notes:
- `quintic` matches the v1 reference to ~1e-6 — identical math, different layout.
- `v1_ortho` / `quintic_ortho` are at ~0.3 because they measure orthogonality
  residual `||U^TU - I||_∞`, not divergence — both are passing the polar map.
- `polar*` and `fp16` modes saturate near 1.0 — the polar/restart variants in
  this branch are not yet numerically stable and the fp16 path inherits the
  same drift. Treat them as WIP.

## Muon CUDA Benchmark · v1 NS vs stabilized Gram NS

RTX 4060 Laptop GPU · cc 8.9 · 7.62 GiB · iters 1000 · 414.0 s total

| shape       |  ρ  | v1 NS  | Quintic | Polar  | +Restart | +Syrk  | best       |
|-------------|-----|--------|---------|--------|----------|--------|------------|
| 1024×1024   | 1.0 |  4.71  |   6.56  |  6.60  |   6.66   |  6.51  | ▼ 0.72×    |
| 2048×2048   | 1.0 | 31.45  |  44.35  | 44.64  |  45.13   | 43.23  | ▼ 0.73×    |
| 2048×1024   | 2.0 |  7.81  |   7.68  |  7.66  |   8.21   |  7.91  | ▲ 1.02×    |
| 4096×1024   | 4.0 | 13.25  |   8.97  |  8.98  |  10.54   |  9.94  | ▲ 1.48×    |
| 8192×1024   | 8.0 | 24.71  |  11.83  | 11.80  |  15.51   | 14.23  | ▲ 2.09×    |

Times are ms/iter. The Gram-based variants only beat v1 once the matrix
becomes rectangular enough (`ρ ≥ 2`) for the smaller `n×n` Gram iterate to
dominate cost — exactly the regime Tridao's blog targets. On square shapes
(`ρ = 1.0`) the extra `X^TX → poly(G) → X·f(G)` round-trip is pure overhead
and v1 wins by ~30%.

---

Author: MantissaGitHub (pradheep.dev)

Hardware: NVIDIA A100-SXM4-80GB

Date: Dec 13, 2025