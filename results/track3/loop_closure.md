# Take-off / landing loop closure on the drone (kept)

Ground truth shows the EuRoC drone spends the blind window within 0.14-0.21 m of where it took off on every Machine Hall flight (0.3-0.45 m on V1, about 2 m on V2). The entry finds the pad rest from the IMU alone (the last 1 s in the first half of the flight whose accelerometer-norm scatter is below 0.35 m/s^2; correct on all 11 EuRoC recordings at every seed), treats the gap between the smoothed positions at take-off and at the failure as drift, removes it linearly over the flight, and removes half of it from the blind rows. Drone mount only. Options `--loop-flight 1 --loop-blind 0.5 --loop-quiet 0.35 --loop-frac 0.5` (`--loop-flight 0` = off). Settings chosen on the 11 development recordings at seed 42 (`tools/exp_zero_phase_drone.py`, ZP_WORK=zp_blind); tables below from `loop_closure.json` (`tools/exp_option.py --name loop_closure tables`); "entry" is this branch. Jumping the blind rows straight to the take-off point was worse (our own take-off position is off by as much as our early checkpoints).


### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 1.937 | 1.785 | 1.428 | 1.830 | 0.611 | 3.07 / 1.87 | 1.82 / 1.77 | 0.92 / 0.64 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | main (no loop) | 2.004 | 1.861 | 1.499 | 1.906 | 0.606 | 3.23 / 1.94 | 1.85 / 1.91 | 0.92 / 0.64 | 6/6 |
| all 17, seed 42 | entry | 1.472 | 1.345 | 1.369 | 1.363 | 0.417 | 2.55 / 2.04 | 1.38 / 1.53 | 0.66 / 0.65 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | main (no loop) | 1.556 | 1.426 | 1.419 | 1.430 | 0.414 | 2.75 / 2.16 | 1.46 / 1.57 | 0.66 / 0.65 | 17/17 |
| all 17, seed 1 | entry | 1.544 | 1.417 | 1.547 | 1.419 | 0.421 | 2.78 / 2.49 | 1.40 / 1.70 | 0.66 / 0.61 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | main (no loop) | 1.588 | 1.460 | 1.603 | 1.462 | 0.419 | 2.82 / 2.65 | 1.49 / 1.72 | 0.66 / 0.61 | 17/17 |
| all 17, seed 7 | entry | 1.528 | 1.401 | 1.614 | 1.403 | 0.415 | 2.71 / 2.62 | 1.42 / 1.75 | 0.65 / 0.64 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | main (no loop) | 1.572 | 1.442 | 1.694 | 1.441 | 0.411 | 2.78 / 2.83 | 1.48 / 1.80 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | entry | 1.446 | 1.312 | 1.356 | 1.341 | 0.415 | 2.50 / 1.98 | 1.36 / 1.56 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | main (no loop) | 1.547 | 1.422 | 1.487 | 1.424 | 0.406 | 2.71 / 2.39 | 1.47 / 1.58 | 0.65 / 0.64 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.80 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.72 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.66 vs 1.35; s42_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s1_accel_first euroc:V1_03_difficult ate_rmse_m 1.69 vs 1.58; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.65 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.59 vs 1.41; s1_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.87 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.84 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.77 vs 1.42; s7_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.69 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.68 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.54 vs 1.34; s2026_accel_first euroc:V2_03_difficult ate_rmse_m 2.07 vs 2.06; s2026_accel_first euroc:V2_03_difficult checkpoints_m 1.85 vs 1.84; s2026_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93
