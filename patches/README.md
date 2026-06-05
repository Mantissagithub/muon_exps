# modded-nanogpt patches

`modded_nanogpt_aurora_balance.patch` applies a small env-gated Aurora-style row-balance ablation to the current `KellerJordan/modded-nanogpt` `train_gpt.py`.

```bash
cd /path/to/modded-nanogpt
git apply /home/pradheep/muon_exps/patches/modded_nanogpt_aurora_balance.patch
SPEEDRUN_AURORA_BALANCE=1 ./run.sh
```

local probe result on the RTX 4060 Laptop GPU:

```bash
uv run python scripts/nanogpt_speedrun_aurora_probe.py --batch 8 --rows 768 --cols 768
```

The Aurora balance pass added about `0.0206 ms` to the isolated NorMuon variance-reduction section for an `8x768x768` bank and did not improve row CV in that synthetic probe. that means it should be treated as a convergence ablation, not a recommended wall-clock WR patch. the current upstream speedrun already has NorMuon row-energy normalization, so full Aurora-style balancing is mostly redundant there.
