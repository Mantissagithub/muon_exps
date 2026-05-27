# muon experiments

just messing around with muon kernels, benchmark variants, and a couple of small training comparisons.

## layout

```text
.
├── cuda/                   # cuda kernels and benchmark driver
├── scripts/                # tui + quick benchmark entrypoints
├── experiments/
│   └── mnist/              # small training comparisons
├── artifacts/              # saved outputs / compiled benchmark binary
└── README.md
```

## run

- `uv run scripts/benchmark_tui.py`
- `uv run scripts/optimizer_variants_tui.py`
- `uv run scripts/pytorch_tui.py`
- `uv run scripts/pytorch_muon_benchmark.py`
- `uv run experiments/mnist/mnist_muon.py`
- `uv run experiments/mnist/mnist_with_adam.py`
- `uv run experiments/mnist/mnist_with_adamw.py`

## notes

- `cuda/benchmark.cu` is the main cuda benchmark driver.
- `cuda/benchmark_optimizer_variants.cu` compares the optimizer update rules on synthetic anisotropic gradients.
- `cuda/gns_muon.cu` has the gram newton-schulz experiments.
- the current benchmark grid is tuned down for a 4060 laptop gpu so it does not take forever.
- some gram-ns variants are still wip and not numerically stable yet.

## quick results

verify is still the same story: quintic matches v1, the polar / restart paths are still not stable in this branch.

| shape     | quintic | v1_ortho | quintic_ortho | polar_ortho | polar_restart_ortho | polar_restart_syrk_ortho | fp16_ortho |
|-----------|---------|----------|---------------|-------------|---------------------|---------------------------|------------|
| 1024×1024 | ok      | ok       | ok            | fail        | fail                | fail                      | fail       |
| 2048×2048 | ok      | ok       | ok            | fail        | fail                | fail                      | fail       |
| 4096×1024 | ok      | ok       | ok            | fail        | fail                | fail                      | fail       |
| 8192×2048 | ok      | ok       | ok            | fail        | fail                | fail                      | fail       |

square cases are still slower than pytorch / flash-muon, and normuon is basically flat with v1 there:

| size   | v1 ns   | normuon | pytorch muon | flash-muon |
|--------|---------|----------|--------------|------------|
| 1024²  | 4.62 ms | 4.57 ms  | 1.05 ms      | ~1.0 ms    |
| 2048²  | 30.59 ms| 30.75 ms | 1.99 ms      | ~1.4 ms    |

more rectangular shapes still favor the gram-ns variants, not normuon:

| shape     | v1 ns | normuon | quintic | polar | +restart | +syrk | best speedup |
|-----------|-------|----------|---------|-------|----------|-------|--------------|
| 2048×1024 | 7.47  | 7.42     | 7.29    | 7.29  | 7.85     | 7.52  | 1.02x        |
| 4096×1024 | 12.71 | 12.66    | 8.59    | 8.59  | 10.13    | 9.51  | 1.48x        |
| 8192×1024 | 23.98 | 24.06    | 11.41   | 11.41 | 14.93    | 13.69 | 2.10x        |

bench run for the table above:

- `NVIDIA GeForce RTX 4060 Laptop GPU`
- `cc 8.9`
- `7.62 GiB`
- `iters 1000`
- `480.5s total`

optimizer-variant benchmark on anisotropic synthetic gradients:

| shape | optimizer | ms/step | row cv | dead rows | ortho defect | alignment |
|-------|-----------|--------:|-------:|----------:|-------------:|----------:|
| 512×128 | muon | 0.410 | 0.496 | 0.195 | 0.385 | 0.913 |
| 512×128 | aurora | 0.737 | 1.780 | 0.000 | 2.183 | 0.030 |
| 512×128 | riemann_aurora | 7.594 | 0.025 | 0.000 | 1.368 | 0.866 |
| 2048×512 | muon | 2.363 | 0.494 | 0.196 | 0.278 | 0.944 |
| 2048×512 | aurora | 4.818 | 1.900 | 0.000 | 2.544 | 0.016 |
| 2048×512 | riemann_aurora | 59.591 | 0.064 | 0.000 | 2.082 | 0.898 |
| 8192×2048 | muon | 110.947 | 0.508 | 0.205 | 0.327 | 0.975 |
| 8192×2048 | aurora | 229.441 | 1.911 | 0.000 | 2.586 | 0.008 |
| 8192×2048 | riemann_aurora | 9139.112 | 0.010 | 0.000 | 2.303 | 0.877 |
| 2048×8192 | muon | 111.715 | 0.467 | 0.000 | 0.536 | 0.996 |
| 2048×8192 | aurora | 235.177 | 0.396 | 0.000 | 0.535 | 0.987 |
| 2048×8192 | riemann_aurora | 7106.821 | 0.019 | 0.000 | 0.260 | 0.878 |

full table is in `benchmark_results.csv`. riemann aurora does what it is supposed to do on row uniformity, but it is very expensive in this raw cuda version.
