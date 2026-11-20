# Failure-state estimate for the blind window on the drone: tested, **not kept**

**Idea** (from reading SRMIST_vortexxx and ZeroDrift, `competitor_code_notes.md`): on the drone mount, the rows before
the IMU failure come from the offline zero-phase pass and the blind rows from the causal filter. Both competitors start
their blind window from an offline estimate of the state at the failure (vortexxx: its whole-file RTS smoother, then
hold; ZeroDrift: its drift-removed track, then a fit of the last 3 s). Here only the blind rows change; every row
before the failure is the entry's own (`tools/exp_failure_state.py`, which runs the entry in-process with its methods
wrapped to record, per row, the causal and un-pulled positions, the filter velocity, and every network and rotor-drag
velocity reading).

* `anchor W`: failure position = the entry's smoothed row W s before the failure plus the filter's own un-pulled
  displacement since (avoids the edge of the zero-phase pass at the failure); blind rows held there (`hold`) or
  continued with the causal blind motion (`coast`).
* `rts W` / `rtsnet W`: the velocity over the last W s re-estimated by a per-axis Kalman filter + RTS smoother with
  the network velocity (its own std) and, for `rts`, the rotor-drag velocity as measurements; the failure position is
  the smoothed row W s before plus the integral of that velocity.

**Result: not kept.** The `entry` row reproduces the shipped numbers exactly. On the development recordings `anchor 10
hold` lowers blind from 1.622 to 1.475 m (Machine Hall 3.20 → 2.86, Vicon 1.42 → 1.28) without touching any other
row. The one-time check reverses it: held-out blind gets worse at every seed (4-seed mean 1.805 → 2.174 m; Vicon
1.87 → 2.47, Machine Hall 2.97 → 3.48), all-17 blind 1.741 → 1.844, and V1_02, MH_05 and V2_03 newly lose to standing
still on blind. This is the same pattern as the smoothed-position hold in `blind_and_handheld.md`.

`rts` with the rotor-drag readings is far worse (blind 2.9-7.9 m): in the last seconds before the failure the drones
are landing, where the drag model does not hold; on MH_01 the drag readings average (−0.55, −1.25) m/s while the
filter's velocity is about 0.1 m/s. The filter discounts them with its Cauchy weight; a fixed-noise smoother does not.
With the network velocity alone (`rtsnet`) the gain is smaller than anchoring (best 1.540 m).

**What it says.** The Machine Hall flights land within a metre of take-off. The causal blind rows, still pulled
toward the start (`--drag-pos-tau 12`), end up closer to that landing point on unseen recordings than any of the
smoothed failure states tried here. A better blind window on the drone needs information we do not estimate (where
the landing is), not a better smoother of the same velocity.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ckpt / blind / drop | Vicon ckpt / blind / drop | TUM-VI ckpt / blind / drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.46 / 3.20 / 2.31 | 1.18 / 1.42 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 5 hold | 1.427 | 1.296 | 1.506 | 1.272 | 0.360 | 2.46 / 2.96 / 2.31 | 1.18 / 1.29 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 5 coast | 1.429 | 1.296 | 1.549 | 1.272 | 0.360 | 2.46 / 3.02 / 2.31 | 1.18 / 1.36 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 10 hold | 1.427 | 1.296 | 1.475 | 1.272 | 0.360 | 2.46 / 2.86 / 2.31 | 1.18 / 1.28 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 10 coast | 1.428 | 1.296 | 1.508 | 1.272 | 0.361 | 2.46 / 2.90 / 2.31 | 1.18 / 1.33 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 20 hold | 1.429 | 1.296 | 1.491 | 1.272 | 0.364 | 2.46 / 2.80 / 2.31 | 1.18 / 1.36 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | anchor 20 coast | 1.429 | 1.296 | 1.502 | 1.272 | 0.364 | 2.46 / 2.81 / 2.31 | 1.18 / 1.39 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 5 hold | 1.513 | 1.296 | 2.905 | 1.286 | 0.483 | 2.46 / 5.15 / 2.33 | 1.18 / 3.49 / 1.23 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 5 coast | 1.508 | 1.296 | 2.830 | 1.286 | 0.483 | 2.46 / 5.02 / 2.33 | 1.18 / 3.38 / 1.23 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 10 hold | 1.716 | 1.296 | 5.243 | 1.307 | 0.666 | 2.46 / 9.37 / 2.36 | 1.18 / 6.75 / 1.26 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 10 coast | 1.707 | 1.296 | 5.160 | 1.307 | 0.666 | 2.46 / 9.21 / 2.36 | 1.18 / 6.65 / 1.26 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 20 hold | 2.033 | 1.296 | 7.950 | 1.331 | 0.887 | 2.46 / 14.09 / 2.40 | 1.18 / 10.66 / 1.30 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rts 20 coast | 2.022 | 1.296 | 7.866 | 1.331 | 0.886 | 2.46 / 13.92 / 2.40 | 1.18 / 10.56 / 1.30 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 5 hold | 1.428 | 1.296 | 1.542 | 1.272 | 0.360 | 2.46 / 3.03 / 2.31 | 1.18 / 1.33 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 5 coast | 1.430 | 1.296 | 1.588 | 1.272 | 0.360 | 2.46 / 3.10 / 2.31 | 1.18 / 1.41 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 10 hold | 1.429 | 1.296 | 1.540 | 1.272 | 0.360 | 2.46 / 3.00 / 2.31 | 1.18 / 1.35 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 10 coast | 1.430 | 1.296 | 1.580 | 1.272 | 0.361 | 2.46 / 3.05 / 2.31 | 1.18 / 1.42 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 20 hold | 1.431 | 1.296 | 1.580 | 1.273 | 0.363 | 2.46 / 2.94 / 2.31 | 1.18 / 1.50 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | rtsnet 20 coast | 1.433 | 1.296 | 1.599 | 1.273 | 0.363 | 2.46 / 2.97 / 2.31 | 1.18 / 1.54 / 1.21 | 0.53 / 0.63 / 0.56 |
| dev, seed 42 | still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 3.88 / 4.68 / 3.66 | 1.86 / 2.21 / 1.86 | 1.11 / 0.66 / 1.15 |

### Check of the frozen setting (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ckpt / blind / drop | Vicon ckpt / blind / drop | TUM-VI ckpt / blind / drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.115 | 1.989 | 1.690 | 2.028 | 0.620 | 3.46 / 2.65 / 3.65 | 1.60 / 1.84 / 1.62 | 0.90 / 0.58 / 0.81 |
| held-out 6, seed 42 | still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.39 / 4.12 / 6.49 | 1.67 / 1.98 / 1.63 | 0.99 / 0.51 / 0.93 |
| held-out 6, seed 42 | anchor 10 hold | 2.129 | 1.989 | 2.038 | 2.032 | 0.618 | 3.46 / 3.07 / 3.66 | 1.60 / 2.47 / 1.63 | 0.90 / 0.58 / 0.81 |
| held-out 6, seed 1 | entry | 2.138 | 1.992 | 1.763 | 2.040 | 0.626 | 3.49 / 2.81 / 3.72 | 1.58 / 1.92 / 1.58 | 0.91 / 0.56 / 0.82 |
| held-out 6, seed 1 | still | 3.209 | 3.015 | 2.196 | 3.050 | 0.838 | 6.39 / 4.10 / 6.51 | 1.67 / 1.98 / 1.69 | 0.99 / 0.51 / 0.95 |
| held-out 6, seed 1 | anchor 10 hold | 2.151 | 1.992 | 2.054 | 2.043 | 0.624 | 3.49 / 3.16 / 3.73 | 1.58 / 2.45 / 1.58 | 0.91 / 0.56 / 0.82 |
| held-out 6, seed 7 | entry | 2.118 | 1.977 | 1.942 | 2.034 | 0.616 | 3.34 / 3.43 / 3.60 | 1.69 / 1.82 / 1.68 | 0.90 / 0.58 / 0.82 |
| held-out 6, seed 7 | still | 3.208 | 3.016 | 2.197 | 3.043 | 0.838 | 6.39 / 4.10 / 6.51 | 1.67 / 1.98 / 1.68 | 0.99 / 0.51 / 0.94 |
| held-out 6, seed 7 | anchor 10 hold | 2.134 | 1.977 | 2.369 | 2.039 | 0.614 | 3.34 / 4.20 / 3.61 | 1.69 / 2.33 / 1.69 | 0.90 / 0.58 / 0.82 |
| held-out 6, seed 2026 | entry | 2.117 | 1.974 | 1.825 | 2.051 | 0.615 | 3.35 / 3.00 / 3.66 | 1.68 / 1.91 / 1.66 | 0.89 / 0.56 / 0.84 |
| held-out 6, seed 2026 | still | 3.208 | 3.015 | 2.198 | 3.072 | 0.837 | 6.39 / 4.11 / 6.60 | 1.67 / 1.98 / 1.67 | 0.99 / 0.51 / 0.95 |
| held-out 6, seed 2026 | anchor 10 hold | 2.132 | 1.974 | 2.236 | 2.055 | 0.613 | 3.35 / 3.50 / 3.66 | 1.68 / 2.65 / 1.67 | 0.89 / 0.56 / 0.84 |
| all 17, seed 42 | entry | 1.672 | 1.541 | 1.646 | 1.539 | 0.453 | 2.86 / 2.98 / 2.85 | 1.32 / 1.56 / 1.34 | 0.66 / 0.62 / 0.64 |
| all 17, seed 42 | still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 4.88 / 4.46 / 4.80 | 1.80 / 2.14 / 1.78 | 1.07 / 0.61 / 1.08 |
| all 17, seed 42 | anchor 10 hold | 1.675 | 1.541 | 1.674 | 1.540 | 0.451 | 2.86 / 2.94 / 2.85 | 1.32 / 1.68 / 1.35 | 0.66 / 0.62 / 0.64 |
| all 17, seed 1 | entry | 1.685 | 1.550 | 1.786 | 1.551 | 0.450 | 2.89 / 3.30 / 2.90 | 1.33 / 1.71 / 1.35 | 0.65 / 0.59 / 0.63 |
| all 17, seed 1 | still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 4.88 / 4.44 / 4.79 | 1.80 / 2.14 / 1.82 | 1.07 / 0.61 / 1.10 |
| all 17, seed 1 | anchor 10 hold | 1.688 | 1.550 | 1.852 | 1.552 | 0.449 | 2.89 / 3.38 / 2.90 | 1.33 / 1.84 / 1.35 | 0.65 / 0.59 / 0.63 |
| all 17, seed 7 | entry | 1.661 | 1.531 | 1.825 | 1.529 | 0.440 | 2.84 / 3.43 / 2.82 | 1.33 / 1.70 / 1.34 | 0.64 / 0.61 / 0.63 |
| all 17, seed 7 | still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 4.88 / 4.45 / 4.80 | 1.80 / 2.13 / 1.82 | 1.07 / 0.61 / 1.10 |
| all 17, seed 7 | anchor 10 hold | 1.666 | 1.531 | 1.948 | 1.530 | 0.439 | 2.84 / 3.64 / 2.83 | 1.33 / 1.88 / 1.35 | 0.64 / 0.61 / 0.63 |
| all 17, seed 2026 | entry | 1.669 | 1.532 | 1.708 | 1.545 | 0.446 | 2.77 / 3.11 / 2.83 | 1.38 / 1.64 / 1.38 | 0.65 / 0.61 / 0.64 |
| all 17, seed 2026 | still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 4.88 / 4.45 / 4.86 | 1.80 / 2.13 / 1.81 | 1.07 / 0.61 / 1.09 |
| all 17, seed 2026 | anchor 10 hold | 1.676 | 1.532 | 1.902 | 1.547 | 0.445 | 2.77 / 3.36 / 2.83 | 1.38 / 1.98 / 1.38 | 0.65 / 0.61 / 0.64 |
| held-out 6, mean of 4 seeds | entry | 2.122 | 1.983 | 1.805 | 2.038 | 0.619 | 3.41 / 2.97 / 3.66 | 1.64 / 1.87 / 1.63 | 0.90 / 0.57 / 0.82 |
| held-out 6, mean of 4 seeds | still | 3.208 | 3.015 | 2.198 | 3.046 | 0.838 | 6.39 / 4.11 / 6.53 | 1.67 / 1.98 / 1.67 | 0.99 / 0.51 / 0.94 |
| held-out 6, mean of 4 seeds | anchor 10 hold | 2.137 | 1.983 | 2.174 | 2.042 | 0.617 | 3.41 / 3.48 / 3.66 | 1.64 / 2.47 / 1.64 | 0.90 / 0.57 / 0.82 |
| all 17, mean of 4 seeds | entry | 1.672 | 1.538 | 1.741 | 1.541 | 0.447 | 2.84 / 3.21 / 2.85 | 1.34 / 1.66 / 1.35 | 0.65 / 0.61 / 0.64 |
| all 17, mean of 4 seeds | still | 2.617 | 2.448 | 2.279 | 2.438 | 0.775 | 4.88 / 4.45 / 4.81 | 1.80 / 2.13 / 1.81 | 1.07 / 0.61 / 1.09 |
| all 17, mean of 4 seeds | anchor 10 hold | 1.676 | 1.538 | 1.844 | 1.543 | 0.446 | 2.84 / 3.33 / 2.85 | 1.34 / 1.84 / 1.36 | 0.65 / 0.61 / 0.64 |

Newly not better than standing still: s42_accel_first euroc:V1_02_medium blind_mean_m 1.22 vs 1.08; s1_accel_first euroc:V1_02_medium blind_mean_m 1.21 vs 1.09; s1_accel_first euroc:MH_05_difficult blind_mean_m 4.39 vs 4.20; s1_accel_first euroc:V1_03_difficult ate_rmse_m 1.59 vs 1.58; s7_accel_first euroc:V1_02_medium blind_mean_m 1.19 vs 1.09; s7_accel_first euroc:MH_05_difficult blind_mean_m 4.71 vs 4.20; s2026_accel_first euroc:V1_02_medium blind_mean_m 1.26 vs 1.08; s2026_accel_first euroc:MH_05_difficult blind_mean_m 4.49 vs 4.22; s2026_accel_first euroc:V2_03_difficult blind_mean_m 3.05 vs 2.63


## Reproduce

```bash
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>
python tools/exp_failure_state.py sweep
python tools/exp_failure_state.py check --setting "anchor 10 hold"
python tools/exp_failure_state.py tables
```
