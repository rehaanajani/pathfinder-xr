# Fading the network velocity on the drone along the damage ramp: tested, **not kept**

**Idea** (from `mh_blind_diagnosis.md`): in the last ~16 s of the Machine Hall flights our speed collapses to the
network's under-read (0.1-0.6 m/s against a true 0.7-1.25 m/s) while the rotor-drag reading stays near the truth. On
the drone mount, divide the network's velocity gain by 1 + c (t/T)^1.7 (the organisers' damage ramp, T = the file's
span from one pass over its timestamps), so drag and the inertial path weigh more as the damage grows. Implementation
in this branch's history (commit "Added an optional fade of the network velocity ...", reverted in the next).
"entry" = `main` @ 920eeb0. Development sweep at seeds 42 and 1 (the gap is at seeds 1 and 7).

**Result: not kept.** The development sweep improves blind at both seeds for every c from 0.3 to 1.5 (c = 1: seed 42
1.401 → 1.229, seed 1 1.629 → 1.528) with checkpoints within ±0.008 m. The one-time check of c = 1 lowers the
4-seed all-17 blind 1.590 → 1.514, but **through the Vicon rooms (1.71 → 1.50), not the Machine Hall** (2.60 → 2.59;
held-out Machine Hall worse, 2.51 → 2.79); seed 2026 gets worse (1.569 → 1.622), the 4-seed checkpoints are slightly
worse (1.478 → 1.481), and V2_01 (blind, seed 2026) and V1_02 (blind, seed 1) newly lose to standing still.

| inputs | setting | checkpoints | blind 5 s | dropouts | Machine Hall blind | Vicon blind | TUM-VI blind |
|---|---|---:|---:|---:|---:|---:|---:|
| all 17, seed 42 | entry | 1.467 | 1.443 | 1.464 | 2.22 | 1.58 | 0.65 |
| all 17, seed 42 | net fade 1 | 1.472 | 1.328 | 1.458 | 2.16 | 1.31 | 0.65 |
| all 17, seed 1 | entry | 1.493 | 1.623 | 1.484 | 2.71 | 1.73 | 0.61 |
| all 17, seed 1 | net fade 1 | 1.493 | 1.528 | 1.476 | 2.67 | 1.49 | 0.61 |
| all 17, seed 7 | entry | 1.473 | 1.727 | 1.465 | 2.89 | 1.84 | 0.64 |
| all 17, seed 7 | net fade 1 | 1.468 | 1.579 | 1.452 | 2.83 | 1.48 | 0.64 |
| all 17, seed 2026 | entry | 1.478 | 1.569 | 1.481 | 2.56 | 1.68 | 0.64 |
| all 17, seed 2026 | net fade 1 | 1.490 | 1.622 | 1.496 | 2.70 | 1.71 | 0.64 |
| held-out 6, 4 seeds | entry | 1.914 | 1.703 | 1.966 | 2.51 | 1.98 | 0.61 |
| held-out 6, 4 seeds | net fade 1 | 1.919 | 1.654 | 1.972 | 2.78 | 1.56 | 0.61 |
| all 17, 4 seeds | entry | 1.478 | 1.590 | 1.473 | 2.60 | 1.71 | 0.63 |
| all 17, 4 seeds | net fade 1 | 1.481 | 1.514 | 1.471 | 2.59 | 1.50 | 0.63 |


### Development sweep (11 development recordings; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | net fade 1 | 1.358 | 1.230 | 1.229 | 1.187 | 0.337 | 2.63 / 2.07 | 1.23 / 1.17 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | net fade 3 | 1.438 | 1.270 | 1.363 | 1.266 | 0.357 | 2.80 / 2.27 | 1.33 / 1.39 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | net fade 9 | 1.771 | 1.531 | 2.117 | 1.590 | 0.397 | 3.27 / 3.67 | 1.89 / 2.41 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag blend 0.25 s | 1.433 | 1.282 | 1.312 | 1.245 | 0.349 | 2.85 / 2.15 | 1.27 / 1.34 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag robust 3 m/s | 1.378 | 1.253 | 1.384 | 1.204 | 0.331 | 2.66 / 2.24 | 1.27 / 1.47 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | entry | 1.354 | 1.226 | 1.401 | 1.198 | 0.314 | 2.58 / 2.38 | 1.26 / 1.41 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | net fade 0.3 | 1.355 | 1.231 | 1.307 | 1.195 | 0.324 | 2.59 / 2.25 | 1.26 / 1.25 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | net fade 0.5 | 1.355 | 1.230 | 1.293 | 1.185 | 0.332 | 2.60 / 2.19 | 1.25 / 1.26 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | net fade 0.7 | 1.355 | 1.230 | 1.263 | 1.184 | 0.334 | 2.61 / 2.13 | 1.24 / 1.22 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | net fade 1.5 | 1.370 | 1.234 | 1.211 | 1.198 | 0.343 | 2.67 / 2.03 | 1.24 / 1.16 | 0.53 / 0.65 | 11/11 |
| dev, seed 1 | net fade 1 | 1.382 | 1.266 | 1.528 | 1.219 | 0.335 | 2.66 / 2.80 | 1.27 / 1.48 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | net fade 3 | 1.442 | 1.314 | 1.541 | 1.274 | 0.349 | 2.81 / 2.78 | 1.33 / 1.53 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | net fade 9 | 1.729 | 1.522 | 1.941 | 1.529 | 0.392 | 3.20 / 3.34 | 1.82 / 2.21 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag blend 0.25 s | 1.457 | 1.332 | 1.585 | 1.279 | 0.342 | 2.88 / 2.80 | 1.32 / 1.63 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag robust 3 m/s | 1.406 | 1.287 | 1.643 | 1.244 | 0.331 | 2.69 / 2.89 | 1.31 / 1.73 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | entry | 1.383 | 1.265 | 1.629 | 1.231 | 0.314 | 2.62 / 3.02 | 1.31 / 1.59 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | standing still | 2.294 | 2.139 | 2.322 | 2.105 | 0.740 | 4.19 / 4.67 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 1 | net fade 0.3 | 1.382 | 1.267 | 1.547 | 1.230 | 0.319 | 2.63 / 2.93 | 1.30 / 1.43 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | net fade 0.5 | 1.381 | 1.267 | 1.570 | 1.224 | 0.331 | 2.64 / 2.88 | 1.29 / 1.53 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | net fade 0.7 | 1.381 | 1.265 | 1.554 | 1.220 | 0.333 | 2.65 / 2.84 | 1.28 / 1.52 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | net fade 1.5 | 1.389 | 1.272 | 1.503 | 1.225 | 0.338 | 2.70 / 2.75 | 1.27 / 1.45 | 0.53 / 0.62 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.041 | 1.910 | 1.519 | 1.951 | 0.605 | 3.41 / 1.99 | 1.79 / 1.93 | 0.92 / 0.64 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | net fade 1 | 2.044 | 1.914 | 1.511 | 1.957 | 0.603 | 3.42 / 2.29 | 1.79 / 1.60 | 0.92 / 0.64 | 6/6 |
| all 17, seed 42 | entry | 1.597 | 1.467 | 1.443 | 1.464 | 0.417 | 2.91 / 2.22 | 1.44 / 1.58 | 0.66 / 0.65 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | net fade 1 | 1.600 | 1.472 | 1.328 | 1.458 | 0.431 | 2.95 / 2.16 | 1.42 / 1.31 | 0.66 / 0.65 | 17/17 |
| all 17, seed 1 | entry | 1.617 | 1.493 | 1.623 | 1.484 | 0.419 | 2.95 / 2.71 | 1.46 / 1.73 | 0.66 / 0.61 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | net fade 1 | 1.613 | 1.493 | 1.528 | 1.476 | 0.431 | 2.98 / 2.67 | 1.42 / 1.49 | 0.66 / 0.61 | 17/17 |
| all 17, seed 7 | entry | 1.604 | 1.473 | 1.727 | 1.465 | 0.414 | 2.91 / 2.89 | 1.47 / 1.84 | 0.65 / 0.64 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | net fade 1 | 1.601 | 1.468 | 1.579 | 1.452 | 0.432 | 2.93 / 2.83 | 1.44 / 1.48 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | entry | 1.607 | 1.478 | 1.569 | 1.481 | 0.410 | 2.89 / 2.56 | 1.49 / 1.68 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | net fade 1 | 1.627 | 1.490 | 1.622 | 1.496 | 0.424 | 2.95 / 2.70 | 1.50 / 1.71 | 0.65 / 0.64 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V2_01_easy position_drift_1s_m 0.43 vs 0.34; s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.77 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.70 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.62 vs 1.35; s42_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s1_accel_first euroc:V2_01_easy position_drift_1s_m 0.43 vs 0.34; s1_accel_first euroc:V1_03_difficult ate_rmse_m 1.61 vs 1.58; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.60 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.52 vs 1.41; s1_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s7_accel_first euroc:V2_01_easy position_drift_1s_m 0.42 vs 0.34; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.86 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.82 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.74 vs 1.42; s7_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93; s2026_accel_first euroc:V2_01_easy position_drift_1s_m 0.44 vs 0.34; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.65 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.63 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.51 vs 1.34; s2026_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93

