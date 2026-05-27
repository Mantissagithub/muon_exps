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
  - the static validation-loss plot with a late-run zoom inset.
- `artifacts/char_lm/loss_report.html`
  - the richer browser report for the same char-lm run.
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

current readout:

| shape | speed winner | row winner | geometry winner | direction winner |
|-------|--------------|------------|-----------------|------------------|
| 512×128 | muon (0.41 ms) | riemann_aurora (cv 0.048) | normuon (0.383) | muon (0.913) |
| 1024×256 | u_normuon (0.73 ms) | riemann_aurora (cv 0.005) | muon (0.293) | muon (0.936) |
| 2048×512 | u_normuon (2.04 ms) | riemann_aurora (cv 0.031) | normuon (0.278) | muon (0.944) |
| 1024×1024 | normuon (4.58 ms) | normuon (cv 0.107) | normuon (0.418) | aurora (0.884) |
| 512×2048 | u_normuon (2.30 ms) | muon (cv 0.022) | normuon (0.297) | aurora (0.928) |

scoreboard across the synthetic grid:

| optimizer | total wins | where it wins |
|-----------|-----------:|---------------|
| u_normuon | 14 | 1024×1024:balance, 1024×1024:dead, 1024×1024:geometry, 1024×256:speed, 1024×256:direction, 2048×512:speed, 2048×512:geometry, 2048×512:direction, ... |
| normuon | 12 | 1024×1024:speed, 1024×1024:balance, 1024×1024:dead, 1024×1024:geometry, 1024×256:direction, 2048×512:geometry, 2048×512:direction, 512×128:geometry, ... |
| muon | 9 | 1024×1024:balance, 1024×1024:dead, 1024×256:geometry, 1024×256:direction, 2048×512:direction, 512×128:speed, 512×128:direction, 512×2048:balance, ... |
| riemann_aurora | 8 | 1024×1024:direction, 1024×256:balance, 1024×256:dead, 2048×512:balance, 2048×512:dead, 512×128:balance, 512×128:dead, 512×2048:dead |
| aurora | 6 | 1024×1024:direction, 1024×256:dead, 2048×512:dead, 512×128:dead, 512×2048:dead, 512×2048:direction |

rough read:

- `u_normuon` and `normuon` now win the most cells overall on this tightened laptop-sized grid.
- `muon` is still the clean speed/alignment baseline, and it keeps the best direction score on most rectangular cases.
- `riemann_aurora` still dominates row balance, but it does that at a much higher runtime cost.
- `aurora` only really stands out on a couple of direction/dead-row cells and does not look competitive as a general winner here.

full table is in `benchmark_results.csv`, and `scripts/optimizer_variants_tui.py` renders it in a more readable way.

## char lm training

there is also a tiny character-lm comparison over TinyShakespeare:

```bash
python experiments/char_lm/train_optimizer_variants.py
python experiments/char_lm/plot_results.py
```

it writes results to `artifacts/char_lm/`. `torch_muon` is skipped automatically if the active torch build does not expose `torch.optim.Muon`.

current checked-in run:

- `15000` train steps
- eval every `500` steps
- `40` eval batches each time
- `3` layers, `4` heads, `128` hidden dim
- `64` batch size
- `128` token context

current saved run from `artifacts/char_lm/summary.md`:

| optimizer | best val | final val | wall time |
|-----------|---------:|----------:|----------:|
| adamw | 1.5348 | 1.6144 | 206.7s |
| torch_muon | 1.5408 | 1.5968 | 221.0s |
| muon_like | 1.5410 | 1.5998 | 274.5s |
| normuon | 1.5545 | 1.6502 | 298.8s |
| u_normuon | 1.5064 | 1.5387 | 296.3s |
| aurora | 1.5288 | 1.5938 | 338.4s |
| riemann_aurora | 1.5312 | 1.6037 | 770.6s |

for this saved run:

- `u_normuon` is the best validation result currently checked into the repo.
- `adamw`, `torch_muon`, `muon_like`, `aurora`, and `riemann_aurora` are all in the same general band, but with different runtime costs.
- `normuon` is clearly behind on this workload.

`artifacts/char_lm/loss_curves.png` is the quick read: one clean static curve with an inset for the late-run spread. `artifacts/char_lm/loss_report.html` keeps the fuller browser view.

![TinyShakespeare validation loss](artifacts/char_lm/loss_curves.png)
