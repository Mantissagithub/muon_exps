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

square cases are still slower than pytorch / flash-muon:

| size   | cuda muon | pytorch muon | flash-muon |
|--------|-----------|--------------|------------|
| 1024²  | 3.67 ms   | 1.05 ms      | ~1.0 ms    |
| 2048²  | 17.88 ms  | 1.99 ms      | ~1.4 ms    |
| 4096²  | 117.96 ms | 9.56 ms      | ~7.1 ms    |

more rectangular shapes look better with the gram-ns variants:

| shape     | v1 ns  | quintic | best speedup |
|-----------|--------|---------|--------------|
| 2048×1024 | 7.81   | 7.68    | 1.02x        |
| 4096×1024 | 13.25  | 8.97    | 1.48x        |
| 8192×1024 | 24.71  | 11.83   | 2.09x        |
