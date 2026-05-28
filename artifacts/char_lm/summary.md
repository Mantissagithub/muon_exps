# char lm optimizer results

| optimizer | beta1 | rho | fixed_beta | best_val_loss | final_val_loss | step_of_best_val | wall_time | avg_update_cosine | final_zx_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `adamw` |  |  |  | 1.5348 | 1.6144 | 6000 | 206.0s | nan | nan |
| `amuse_muon_b0.4_r0.5` | 0.4 | 0.5 |  | 1.5331 | 1.5331 | 15000 | 416.0s | 0.999998 | 0.612224 |
| `amuse_muon_b0.4_r0.8` | 0.4 | 0.8 |  | 1.5731 | 1.5731 | 15000 | 423.9s | 0.999998 | 0.647462 |
| `amuse_muon_b0.6_r0.5` | 0.6 | 0.5 |  | 1.5523 | 1.5523 | 15000 | 449.1s | 0.999998 | 0.628307 |
| `amuse_muon_b0.6_r0.8` | 0.6 | 0.8 |  | 1.5822 | 1.5822 | 15000 | 429.7s | 0.999999 | 0.662113 |
| `muon_like` |  |  |  | 1.5410 | 1.5998 | 5000 | 276.5s | nan | nan |
| `sf_muon_fixed_beta_0.6` | 0.6 |  | 0.6 | 1.5238 | 1.5310 | 14000 | 413.6s | 0.999998 | 0.577564 |
| `sf_muon_fixed_beta_0.9` | 0.9 |  | 0.9 | 1.5436 | 1.5436 | 15000 | 399.3s | 0.999999 | 0.613608 |
| `torch_muon` |  |  |  | 1.5408 | 1.5968 | 5000 | 220.1s | nan | nan |
