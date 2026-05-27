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
- `uv run scripts/pytorch_tui.py`
- `uv run scripts/pytorch_muon_benchmark.py`
- `uv run experiments/mnist/mnist_muon.py`
- `uv run experiments/mnist/mnist_with_adam.py`
- `uv run experiments/mnist/mnist_with_adamw.py`

## notes

- `cuda/benchmark.cu` is the main cuda benchmark driver.
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
