# Rotor-drag offset in the smoothing input only (kept)

Mayank's diagnosis (PR #59) found a near-constant offset in the drone's rotor-drag readings (about -0.6 / -0.3 m/s, fitted on development flights, PR #61). Removing it from the whole filter improved checkpoints and dropouts but made blind worse, because the blind rows come from the causal trajectory. Here a second (shadow) velocity with the offset removed is carried in the same pass and integrated into the position the offline zero-phase smoothing uses; the causal rows, and so the blind window, keep the uncorrected velocity. Options `--zp-drag-off1 -0.6 --zp-drag-off2 -0.3` (0 turns it off). Tables from `drag_offset_smoothing.json` (`tools/exp_option.py --name drag_offset_smoothing tables`); "entry" is this branch, "main (no offset)" is main.


### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.004 | 1.861 | 1.499 | 1.906 | 0.606 | 3.23 / 1.94 | 1.85 / 1.91 | 0.92 / 0.64 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | main (no offset) | 2.041 | 1.910 | 1.519 | 1.951 | 0.605 | 3.41 / 1.99 | 1.79 / 1.93 | 0.92 / 0.64 | 6/6 |
| all 17, seed 42 | entry | 1.556 | 1.426 | 1.419 | 1.430 | 0.414 | 2.75 / 2.16 | 1.46 / 1.57 | 0.66 / 0.65 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | main (no offset) | 1.597 | 1.467 | 1.443 | 1.464 | 0.417 | 2.91 / 2.22 | 1.44 / 1.58 | 0.66 / 0.65 | 17/17 |
| all 17, seed 1 | entry | 1.588 | 1.460 | 1.603 | 1.462 | 0.419 | 2.82 / 2.65 | 1.49 / 1.72 | 0.66 / 0.61 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | main (no offset) | 1.617 | 1.493 | 1.623 | 1.484 | 0.419 | 2.95 / 2.71 | 1.46 / 1.73 | 0.66 / 0.61 | 17/17 |
| all 17, seed 7 | entry | 1.572 | 1.442 | 1.694 | 1.441 | 0.411 | 2.78 / 2.83 | 1.48 / 1.80 | 0.65 / 0.64 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | main (no offset) | 1.604 | 1.473 | 1.727 | 1.465 | 0.414 | 2.91 / 2.89 | 1.47 / 1.84 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | entry | 1.547 | 1.422 | 1.487 | 1.424 | 0.406 | 2.71 / 2.39 | 1.47 / 1.58 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | main (no offset) | 1.607 | 1.478 | 1.569 | 1.481 | 0.410 | 2.89 / 2.56 | 1.49 / 1.68 | 0.65 / 0.64 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.76 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.69 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.61 vs 1.35; s42_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s1_accel_first euroc:V1_03_difficult ate_rmse_m 1.61 vs 1.58; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.58 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.51 vs 1.41; s1_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.83 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.79 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.71 vs 1.42; s7_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.64 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.63 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.50 vs 1.34; s2026_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93
