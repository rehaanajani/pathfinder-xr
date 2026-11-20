# Stuck runs replaced by interpolated readings: much better means, **fails one keep rule** (decision for the team)

**Idea** (ZeroDrift's white-box inverse preprocessing, `src/inverse_degrade.py` step 5 and 7; see
`competitor_code_notes.md`): the organisers' script makes a sensor "stick" by copying one row onto the next 2-10 rows.
Our health check distrusted a row only from the third bit-identical repeat on, so the first repeats were integrated as
real readings and every longer run became a short dropout: the filter's gravity low-pass and rotor-drag filter
restarted and the dropout bridge broke. Our spike repair (#42) did not touch this.

**Change** (`unstick()` in `src/main.py`, `Unstick` / `read_rows` in `src/main.cpp`): rows whose readings are
bit-identical to the last fresh row (with a later timestamp) are held back until the next different row arrives, then
replaced by readings interpolated linearly in time between the two real rows. A run longer than `--unstick-max`
seconds, a run that ends in an all-zero row (the dead sensor) and a run at the end of the file are left as they are,
so a genuinely frozen sensor is still caught by the health check. Duplicated rows (same timestamp) are untouched; the
filter already skips a zero time step. The organisers' runs are at most 10 rows (0.05 s at 200 Hz); 0.1 s covers them
also at 100 Hz input. On MH_04 / room5 / V1_01 (seed 42) it repairs 228 / 386 / 328 rows against the 260 / 412 / 364
stuck samples the script reports (the rest fell on rows it later dropped). Our own code, from the idea; ZeroDrift's
repository has no licence and none of its code is used.

**Result.** On every seed the means of all three scored metrics improve, the Machine Hall blind most. It passes every
keep rule except one: some recordings newly lose to standing still, on blind 5 s (scored) and on ATE. It is on by
default in this branch (`--unstick-max 0.1`); `--unstick-max 0` restores `main`. Settings were not re-tuned after the
check.

| inputs (mean of seeds 42, 1, 7, 2026) | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ckpt / blind / drop | Vicon ckpt / blind / drop | TUM-VI ckpt / blind / drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6 | entry | 2.122 | 1.983 | 1.805 | 2.038 | 0.619 | 3.41 / 2.97 / 3.66 | 1.64 / 1.87 / 1.63 | 0.90 / 0.57 / 0.82 |
| held-out 6 | unstick 0.1 s | 2.052 | 1.914 | 1.703 | 1.966 | 0.606 | 3.18 / 2.51 / 3.36 | 1.68 / 1.98 / 1.69 | 0.88 / 0.61 / 0.85 |
| held-out 6 | standing still | 3.208 | 3.015 | 2.198 | 3.046 | 0.838 | 6.39 / 4.11 / 6.53 | 1.67 / 1.98 / 1.67 | 0.99 / 0.51 / 0.94 |
| all 17 | entry | 1.672 | 1.538 | 1.741 | 1.541 | 0.447 | 2.84 / 3.21 / 2.85 | 1.34 / 1.66 / 1.35 | 0.65 / 0.61 / 0.64 |
| all 17 | unstick 0.1 s | 1.606 | 1.478 | 1.590 | 1.473 | 0.415 | 2.69 / 2.60 / 2.67 | 1.34 / 1.71 / 1.35 | 0.61 / 0.63 / 0.60 |
| all 17 | standing still | 2.617 | 2.448 | 2.279 | 2.438 | 0.775 | 4.88 / 4.45 / 4.81 | 1.80 / 2.13 / 1.81 | 1.07 / 0.61 / 1.09 |

* All 17, 4-seed mean: checkpoints 1.538 → 1.478, **blind 1.741 → 1.590**, dropouts 1.541 → 1.473, ATE 1.672 → 1.606.
* Held-out, 4-seed mean: checkpoints 1.983 → 1.914, blind 1.805 → 1.703, dropouts 2.038 → 1.966.
* Machine Hall blind (all 17, seed 42): 2.98 → 2.22 m (vortexxx: 1.65, ZeroDrift: 2.24 on the scoreboard).
* TUM-VI blind is slightly worse (all 17, seed 42: 0.62 → 0.65).
* **Newly not better than standing still** (better before, not now): s42 tumvi:room1 blind 5 s 0.52 → 0.59 (standing still 0.54); s42 tumvi:room5 ATE 0.91 → 0.95 (standing still 0.93); s1 euroc:V1_03_difficult ATE 1.55 → 1.61 (standing still 1.58); s1 tumvi:room5 ATE 0.93 → 0.95 (standing still 0.93); s7 tumvi:room1 blind 5 s 0.53 → 0.64 (standing still 0.54); s7 tumvi:room3 blind 5 s 0.67 → 0.69 (standing still 0.69); s7 tumvi:room5 ATE 0.90 → 0.94 (standing still 0.93); s2026 tumvi:room1 blind 5 s 0.52 → 0.63 (standing still 0.54); s2026 euroc:V2_03_difficult blind 5 s 2.49 → 2.71 (standing still 2.63); s2026 tumvi:room5 ATE 0.91 → 0.94 (standing still 0.93).
  So TUM-VI room1's blind window falls behind standing still at three of four seeds (by 0.05-0.10 m), and room3 and
  V2_03 at one seed each; room5's ATE ties or loses by 0.01-0.02 m at every seed (its checkpoints improve, 0.78 → 0.72 m
  at seed 42). V1_03 already lost to standing still on ATE, checkpoints and dropouts at seeds 42, 7 and 2026 before
  this change. The TUM-VI blind loss was visible on the development set too (room1-4 blind 0.63 → 0.65 m).

**Resources** (`tools/measure_python_budget.py`, idle machine, Apple M4): MH_01 algorithm heap 3.48 MiB (unchanged
from `main`), 6.21 MiB including compiling `main.py`; worst CPU 6.41 ms per second of IMU (`main`: 6.39).
**C++**: matches `main.py` within 3e-6 m on room5, room1, MH_01, MH_04 and V1_03. **Tests**: `pytest tests/stress`
60 passed, 2 skipped; `PF_ENTRY=cpp` 62 passed.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | unstick 0.1 s | 1.354 | 1.226 | 1.401 | 1.198 | 0.314 | 2.58 / 2.38 | 1.26 / 1.41 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | unstick 0.5 s | 1.352 | 1.225 | 1.402 | 1.196 | 0.312 | 2.58 / 2.38 | 1.26 / 1.41 | 0.52 / 0.66 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.115 | 1.989 | 1.690 | 2.028 | 0.620 | 3.72 / 2.65 | 1.73 / 1.84 | 0.89 / 0.58 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | unstick 0.1 s | 2.041 | 1.910 | 1.519 | 1.951 | 0.605 | 3.41 / 1.99 | 1.79 / 1.93 | 0.92 / 0.64 | 6/6 |
| all 17, seed 42 | entry | 1.672 | 1.541 | 1.646 | 1.539 | 0.453 | 3.12 / 2.98 | 1.44 / 1.56 | 0.69 / 0.62 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | unstick 0.1 s | 1.597 | 1.467 | 1.443 | 1.464 | 0.417 | 2.91 / 2.22 | 1.44 / 1.58 | 0.66 / 0.65 | 17/17 |
| all 17, seed 1 | entry | 1.685 | 1.550 | 1.786 | 1.551 | 0.450 | 3.16 / 3.30 | 1.46 / 1.71 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | unstick 0.1 s | 1.617 | 1.493 | 1.623 | 1.484 | 0.419 | 2.95 / 2.71 | 1.46 / 1.73 | 0.66 / 0.61 | 17/17 |
| all 17, seed 7 | entry | 1.661 | 1.531 | 1.825 | 1.529 | 0.440 | 3.09 / 3.43 | 1.46 / 1.70 | 0.67 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | unstick 0.1 s | 1.604 | 1.473 | 1.727 | 1.465 | 0.414 | 2.91 / 2.89 | 1.47 / 1.84 | 0.65 / 0.64 | 17/17 |
| all 17, seed 2026 | entry | 1.669 | 1.532 | 1.708 | 1.545 | 0.446 | 3.05 / 3.11 | 1.50 / 1.64 | 0.68 / 0.61 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | unstick 0.1 s | 1.607 | 1.478 | 1.569 | 1.481 | 0.410 | 2.89 / 2.56 | 1.49 / 1.68 | 0.65 / 0.64 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.76 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.69 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.61 vs 1.35; s42_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s1_accel_first euroc:V1_03_difficult ate_rmse_m 1.61 vs 1.58; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.58 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.51 vs 1.41; s1_accel_first tumvi:room5 ate_rmse_m 0.95 vs 0.93; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.83 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.79 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.71 vs 1.42; s7_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.64 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.63 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.50 vs 1.34; s2026_accel_first tumvi:room5 ate_rmse_m 0.94 vs 0.93


## Reproduce

```bash
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>
python tools/exp_option.py --name stuck_runs sweep --variants '{"unstick 0.1 s": ["--unstick-max", "0.1"], "unstick 0.5 s": ["--unstick-max", "0.5"]}'
python tools/exp_option.py --name stuck_runs check --label "unstick 0.1 s" --variant '["--unstick-max", "0.1"]'
```
(run before 0.1 s became the default; "entry" is `main` @ 38b2481. Pass `--unstick-max 0` for the old behaviour.)
