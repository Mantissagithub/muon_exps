# muon experiments

just messing around with muon kernels, optimizer variants, and a couple of small training comparisons.

## layout

```text
.
├── cuda/                   # cuda kernels, variant implementations, benchmark drivers
├── scripts/                # tui wrappers / quick benchmark entrypoints
├── experiments/
│   ├── mnist/              # older small training comparisons
│   └── char_lm/            # tinyshakespeare optimizer comparison
├── optimizers/             # pytorch reference optimizer wrappers
├── artifacts/              # saved outputs / compiled benchmark binaries
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
- `python experiments/char_lm/train_optimizer_variants.py`
- `python experiments/char_lm/plot_results.py`

## what's here

- `cuda/benchmark.cu`
  - the older main cuda benchmark driver.
- `cuda/benchmark_optimizer_variants.cu`
  - compares `muon`, `normuon`, `u_normuon`, `aurora`, and `riemann_aurora` on synthetic anisotropic gradients.
- `cuda/README.md`
  - writes out the math for the cuda-side variants in latex.
- `optimizers/muon_variants.py`
  - pytorch reference wrappers used for the char-lm run.
- the current synthetic benchmark grid is intentionally tighter because this repo is being run on a `RTX 4060 Laptop GPU`, and `riemann_aurora` gets expensive fast.
- some gram-ns / polar restart paths are still wip and not numerically stable in this branch.

## artifacts

- `artifacts/bin/benchmark`
  - compiled binary for the older gram-ns / muon kernel benchmark.
- `artifacts/bin/benchmark_optimizer_variants`
  - compiled binary for the synthetic optimizer-update benchmark.
- `benchmark_results.csv`
  - full csv output from `cuda/benchmark_optimizer_variants.cu`.
- `artifacts/char_lm/results.csv`
  - per-checkpoint train loss, val loss, tokens/sec, elapsed time, and running best val.
- `artifacts/char_lm/summary.md`
  - compact final summary for the saved tinyshakespeare run.
- `artifacts/char_lm/loss_curves.png`
  - the nicer plot with direct labels plus best-loss and wall-time side panels.
- `artifacts/mnist/results/`
  - the older mnist logs and loss-curve images.

## cuda benchmark

verify is still the same story: quintic matches v1, while the polar / restart paths are still not stable in this branch.

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

## optimizer-update benchmark

`cuda/benchmark_optimizer_variants.cu` does not train a model. it just compares the update rule itself on synthetic gradients with deliberately uneven row energy.

what the extra metrics mean:

- `row cv`
  - lower means the update mass is spread more evenly across rows / neurons.
- `dead rows`
  - fraction of rows with almost no update.
- `ortho defect`
  - how far the final update drifts from the polar geometry.
- `alignment`
  - cosine-style alignment with the original gradient.

current results:

| shape | optimizer | ms/step | row cv | dead rows | ortho defect | alignment |
|-------|-----------|--------:|-------:|----------:|-------------:|----------:|
| 512×128 | muon | 0.422 | 0.496 | 0.195 | 0.385 | 0.913 |
| 512×128 | aurora | 0.784 | 1.780 | 0.000 | 2.183 | 0.030 |
| 512×128 | riemann_aurora | 7.555 | 0.048 | 0.000 | 1.468 | 0.876 |
| 1024×256 | muon | 0.815 | 0.533 | 0.221 | 0.293 | 0.936 |
| 1024×256 | aurora | 1.527 | 1.770 | 0.000 | 2.372 | 0.013 |
| 1024×256 | riemann_aurora | 18.996 | 0.005 | 0.000 | 3.411 | 0.852 |
| 2048×512 | muon | 2.379 | 0.494 | 0.196 | 0.278 | 0.944 |
| 2048×512 | aurora | 4.590 | 1.900 | 0.000 | 2.544 | 0.016 |
| 2048×512 | riemann_aurora | 59.331 | 0.031 | 0.000 | 2.445 | 0.874 |
| 1024×1024 | muon | 4.720 | 0.107 | 0.000 | 0.419 | 0.825 |
| 1024×1024 | aurora | 5.361 | 0.500 | 0.011 | 0.530 | 0.884 |
| 1024×1024 | riemann_aurora | 5.359 | 0.500 | 0.011 | 0.530 | 0.884 |
| 512×2048 | muon | 2.217 | 0.022 | 0.000 | 0.300 | 0.865 |
| 512×2048 | aurora | 4.662 | 0.180 | 0.000 | 0.404 | 0.928 |
| 512×2048 | riemann_aurora | 76.498 | 0.067 | 0.000 | 0.313 | 0.905 |

rough read:

- `muon` is still the speed baseline and usually keeps the best gradient alignment.
- `normuon` / `u_normuon` are basically tiny perturbations of `muon` in this synthetic setup, not a new regime.
- `aurora` removes dead rows but gives up too much geometry and alignment here.
- `riemann_aurora` is the cleanest on row balance, but it is still expensive enough that the grid had to be tightened for the laptop.

full table is in `benchmark_results.csv`, and `scripts/optimizer_variants_tui.py` renders it in a more readable way.

## char lm training

there is also a tiny character-lm comparison over TinyShakespeare:

```bash
python experiments/char_lm/train_optimizer_variants.py
python experiments/char_lm/plot_results.py
```

it writes results to `artifacts/char_lm/`. `torch_muon` is skipped automatically if the active torch build does not expose `torch.optim.Muon`.

default run right now:

- `6000` train steps
- eval every `250` steps
- `40` eval batches each time
- `3` layers, `4` heads, `128` hidden dim
- `64` batch size
- `128` token context

current saved run from `artifacts/char_lm/summary.md`:

| optimizer | best val | final val | wall time |
|-----------|---------:|----------:|----------:|
| adamw | 1.5348 | 1.5803 | 164.9s |
| torch_muon | 1.5408 | 1.5695 | 176.5s |
| muon_like | 1.5410 | 1.5703 | 219.5s |
| normuon | 1.5545 | 1.6037 | 238.0s |
| u_normuon | 1.5064 | 1.5125 | 237.9s |
| aurora | 1.5288 | 1.5657 | 272.1s |
| riemann_aurora | 1.5312 | 1.5682 | 625.5s |

for this saved run:

- `u_normuon` is the best validation result currently checked into the repo.
- `adamw`, `torch_muon`, `muon_like`, `aurora`, and `riemann_aurora` are all in the same general band, but with different runtime costs.
- `normuon` is clearly behind on this workload.

`artifacts/char_lm/loss_curves.png` is the easiest way to read it quickly, since it shows the full validation trajectory plus the best-loss and wall-time rankings on the side.
