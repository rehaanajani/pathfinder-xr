# Accelerometer-scale estimate: tested, **not kept**

**Idea** (ZeroDrift carries an accelerometer-scale state in its two-pass error-state filter; `competitor_code_notes.md`):
the organisers' script random-walks the accelerometer scale (up to 1-2 % by the end of a file), which misreads
gravity by 0.1-0.2 m/s². Implementation in this branch's history (commit "Added an optional accelerometer-scale
estimate ...", reverted in the next): in calm moments (turn rate below `--acc-scale-calm`, |f| within 1 m/s² of g)
a slow mean of |f| is tracked with time constant `--acc-scale-tau`; every reading is scaled by g / mean, clamped to
±5 %. ZeroDrift's own scale state is learned inside its EKF; our two-pass start covers only the first 10 s, where the
damage is near zero, so a two-pass version could only carry a scale of 1.

**Result: not kept.** On the development recordings τ = 10 s moves every metric by 0.002-0.005 m; longer τ trades
blind for checkpoints (τ = 60 s: checkpoints 1.296 → 1.272, blind 1.622 → 1.736). The one-time check of τ = 10 s is
worse on every set, most on the Vicon rooms (seed 42 Vicon ATE 1.44 → 1.55). Our slow acceleration-residual removal
(`--acc-res-tau`) already takes out the gravity error a scale error causes; on a drone in flight |f| is not g in calm
moments either (thrust and vibration), so the extra estimate adds error.

| inputs (mean of seeds 42, 1, 7, 2026) | setting | checkpoints | blind 5 s | dropouts |
|---|---|---:|---:|---:|
| held-out 6 | entry | 1.983 | 1.805 | 2.038 |
| held-out 6 | scale tau 10 s | 2.069 | 1.876 | 2.128 |
| all 17 | entry | 1.538 | 1.741 | 1.541 |
| all 17 | scale tau 10 s | 1.566 | 1.770 | 1.570 |


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | scale tau 10 s | 1.428 | 1.294 | 1.617 | 1.272 | 0.358 | 2.73 / 3.22 | 1.29 / 1.40 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | scale tau 30 s | 1.422 | 1.289 | 1.637 | 1.266 | 0.368 | 2.72 / 3.29 | 1.29 / 1.39 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | scale tau 60 s | 1.411 | 1.272 | 1.736 | 1.266 | 0.375 | 2.68 / 3.66 | 1.28 / 1.40 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | scale tau 30 s, calm 1 rad/s | 1.427 | 1.296 | 1.624 | 1.270 | 0.366 | 2.72 / 3.23 | 1.29 / 1.41 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.115 | 1.989 | 1.690 | 2.028 | 0.620 | 3.72 / 2.65 | 1.73 / 1.84 | 0.89 / 0.58 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | scale tau 10 s | 2.222 | 2.067 | 1.786 | 2.122 | 0.635 | 3.72 / 2.64 | 2.06 / 2.14 | 0.89 / 0.57 | 6/6 |
| all 17, seed 42 | entry | 1.672 | 1.541 | 1.646 | 1.539 | 0.453 | 3.12 / 2.98 | 1.44 / 1.56 | 0.69 / 0.62 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | scale tau 10 s | 1.708 | 1.567 | 1.677 | 1.572 | 0.456 | 3.12 / 2.99 | 1.55 / 1.65 | 0.69 / 0.61 | 17/17 |
| all 17, seed 1 | entry | 1.685 | 1.550 | 1.786 | 1.551 | 0.450 | 3.16 / 3.30 | 1.46 / 1.71 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | scale tau 10 s | 1.721 | 1.578 | 1.825 | 1.577 | 0.457 | 3.16 / 3.29 | 1.56 / 1.84 | 0.68 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 1.661 | 1.531 | 1.825 | 1.529 | 0.440 | 3.09 / 3.43 | 1.46 / 1.70 | 0.67 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | scale tau 10 s | 1.697 | 1.562 | 1.851 | 1.560 | 0.446 | 3.09 / 3.44 | 1.56 / 1.77 | 0.68 / 0.61 | 17/17 |
| all 17, seed 2026 | entry | 1.669 | 1.532 | 1.708 | 1.545 | 0.446 | 3.05 / 3.11 | 1.50 / 1.64 | 0.68 / 0.61 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | scale tau 10 s | 1.701 | 1.557 | 1.729 | 1.571 | 0.454 | 3.05 / 3.08 | 1.60 / 1.73 | 0.68 / 0.60 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.70 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.62 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.55 vs 1.35; s42_accel_first euroc:V2_03_difficult ate_rmse_m 2.41 vs 2.06; s42_accel_first euroc:V2_03_difficult checkpoints_m 2.06 vs 1.84; s42_accel_first euroc:V2_03_difficult dropout_m 2.26 vs 1.92; s42_accel_first euroc:V2_03_difficult position_drift_1s_m 0.80 vs 0.80; s1_accel_first euroc:V2_01_easy position_drift_1s_m 0.38 vs 0.34; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.54 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.46 vs 1.41; s1_accel_first euroc:V2_03_difficult ate_rmse_m 2.45 vs 2.06; s1_accel_first euroc:V2_03_difficult checkpoints_m 2.15 vs 1.84; s1_accel_first euroc:V2_03_difficult dropout_m 2.21 vs 1.97; s1_accel_first euroc:V2_03_difficult position_drift_1s_m 0.81 vs 0.80; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.79 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.77 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.67 vs 1.42; s7_accel_first euroc:V2_03_difficult ate_rmse_m 2.47 vs 2.06; s7_accel_first euroc:V2_03_difficult checkpoints_m 2.20 vs 1.84; s7_accel_first euroc:V2_03_difficult dropout_m 2.25 vs 1.94; s7_accel_first euroc:V2_03_difficult position_drift_1s_m 0.80 vs 0.80; s2026_accel_first euroc:V2_01_easy position_drift_1s_m 0.39 vs 0.34; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.64 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.63 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.49 vs 1.34; s2026_accel_first euroc:V2_03_difficult ate_rmse_m 2.52 vs 2.06; s2026_accel_first euroc:V2_03_difficult checkpoints_m 2.26 vs 1.84; s2026_accel_first euroc:V2_03_difficult dropout_m 2.38 vs 1.99

