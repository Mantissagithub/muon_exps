# Follow-up: fast2 low-step orthogonalization

Branch: `experiment/fast2-low-step-orthogonalization`

## What this branch changed

- `polar.py`: added an opt-in per-step coefficient schedule to `polar()` (new
  `coeffs=` arg; `coeffs=None` keeps the legacy fixed Keller-Jordan path byte-for-byte,
  so every existing optimizer is unchanged), plus the `MODDED_NANOGPT_NS_COEFFS`
  constant (the speedrun's 5-step quintic schedule).
- `optimizers/muon_variants.py`: threaded `polar_coeffs` through the AMUSE chain
  (mirroring `polar_dtype`) and switched `AMUSEAuroraFast2` to the quintic schedule at
  **2 steps in bf16** (was: fixed "simple" coeffs, fp16).
- `scripts/h100_orthogonalization_bench.py`, `scripts/h100_convergence_ab.py`: the
  H100 reproduction harness used below.

## What we measured on an H100 (SXM5, bf16, torch.compiled)

Context discovered during the run: **the current nanogpt-speedrun WR already uses
Polar Express at 5 steps** (not plain Muon), plus NorMuon + cautious weight decay +
Triton kernels. So the real question was: *does dropping the orthogonalization from
5 steps to 2 speed up the run without wrecking convergence?*

1. **Orthogonalization op (speed + quality)** — `h100_orthogonalization_bench.py`:
   - ours (quintic, 2-step) is **1.6x faster** than the WR's 5-step Polar Express.
   - but ours is **~5x worse** at orthogonalizing: distance to the true polar factor
     U Vᵀ ≈ **0.5 vs 0.10**. PE/CANS truncated to 2–3 steps overshoot and are no better.

2. **Convergence A/B on real FineWeb GPT-2 tokens** — `h100_convergence_ab.py`
   (small GPT, only the orthogonalization differs between runs):
   - First pass (no magnitude control): ours looked *slightly better* (−0.06 val).
   - **That was an effective-LR artifact.** After normalizing both modes to the same
     update magnitude (the only remaining difference is direction/quality), over 1000
     steps:
     ```
     wr   (5-step PE):  final val 5.846
     ours (2-step q):   final val 6.312   (+0.466 WORSE)
     steady per-step:   ~29.4 ms vs ~26.8 ms  (~1.10x faster)
     ```

## Verdict

**Not a world record.** The 2-step variant buys ~10% per optimizer step (and the
orthogonalization is a *smaller* fraction of step time at real speedrun scale, so the
real-world saving is smaller still), but it converges substantially worse. The WR's
choice of 5-step Polar Express is the right speed/quality trade. Reducing step count
is the wrong lever.

## Where an actual win could come from (next directions)

1. **Keep quality, cut cost of the 5-step iteration itself** — the promising lever.
   Look at Gram-Newton-Schulz / hardware-aware NS (Tri Dao, 2026:
   <https://tridao.me/blog/2026/gram-newton-schulz/>) and the WR's own `split_baddbmm`
   / fused Triton `XTX`/`XXT`/`ba_plus_cAA` kernels. Goal: same orthogonality at fewer
   FLOPs / better tensor-core utilization, not fewer steps.
2. **Coefficients optimized for the low-step regime specifically.** Off-the-shelf
   Polar Express / CANS are tuned for ~5+ steps and overshoot at 2–3. Run Remez over
   the *empirical* singular-value band of real momentum matrices to get a genuinely
   2–3-step-optimal schedule. (CPU experiments: the quintic schedule only modestly
   beats fixed coeffs at 2 steps; the bottleneck is fundamental, see below.)
3. **Spectral-norm normalization** (cheap power iteration) instead of Frobenius, so the
   coefficients operate in their designed range regardless of the matrix's effective
   rank. Local tests showed KJ-spectral ≈ 2x lower orthogonality error than KJ-Frobenius
   at 2 steps — possibly the cheapest reliability win, worth an H100 convergence A/B.
4. **Validate at real scale and target.** The 512-dim / 1000-step proxy may mis-state
   the gap. Re-run the A/B inside `modded-nanogpt/train_gpt.py` (world_size=1 is allowed)
   to val 3.28, or on multi-GPU, before trusting any ranking.
5. **Joint LR/momentum tuning.** A lower-quality orthogonalization may simply prefer a
   different LR/momentum; the fair A/B fixed magnitude but not the schedule. Sweep LR
   per mode before declaring a convergence penalty final.

## Note on fundamentals

On strongly decaying (spiked) spectra, *no* method orthogonalizes well in 2–3 steps —
the bottom singular values can't be lifted to 1 that fast (distance ~0.9 for everything).
This caps how good any low-step scheme can be; (1) above (cheaper full-quality steps)
is therefore more likely to pay off than (2) (better low-step coeffs).

## Reproduce

```bash
# rent 1x H100, then on the pod:
pip install torch --index-url https://download.pytorch.org/whl/cu124
python data/cached_fineweb10B.py 2     # in a modded-nanogpt clone
python scripts/h100_orthogonalization_bench.py
python scripts/h100_convergence_ab.py 1000
```
