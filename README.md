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
