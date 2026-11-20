# Damage-aware trust and rests on handheld mounts (20-21 Nov 2026): not kept

Ideas from vortexxx (no licence, ideas only; [`competitor_training_notes.md`](competitor_training_notes.md) X1, X2).
Their filter has no network, and two of its parts carry its TUM-VI result:

1. process noise that follows the damage ramp and the measured excess gyro noise;
2. zero-velocity updates at every rest of 1.5 s or more, with limits loose enough to survive the damage noise.

Organisers' script, `tools/exp_option.py`; raw rows in [`long_rest.json`](long_rest.json) and
[`tilt_trust.json`](tilt_trust.json).

## Rests of 1.5 s or more

**How often the walkers rest.** Ground-truth speed below 0.05 m/s for at least 1.5 s:

- rooms 1-5 rest only at the very start, which the filter already handles;
- room6 rests twice near its end (2.3 s at 122.8 s and 1.9 s at 129 s), around the IMU failure.

So the most this can do is help a late rest before the blind window.

**Test (our own code, `--rest-s`, handheld mounts only).**

- Over the last 1.5 s, gyro and accelerometer scatter must be below loose limits and |mean accel| within
  0.25 m/s² of g. Velocity then goes to zero; the gyro bias is not learned there.
- At vortexxx's gyro limit (0.06 rad/s) it never fires: the organisers' gyro spikes (up to 1 rad/s) push the
  scatter over a 1.5 s rest to 0.09-0.10 rad/s. vortexxx removes spikes with a running median first.
- The gyro limit was chosen on development only, as the loosest value that leaves development unchanged (0.2 rad/s;
  development has no mid-run rests, so any firing there would be a false rest).


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | rest 1.5 s, gyro 0.06 | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | rest 1.5 s, gyro 0.12 | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.64 | 11/11 |
| dev, seed 42 | rest 1.5 s, gyro 0.2 | 1.430 | 1.297 | 1.621 | 1.272 | 0.362 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | rest 1.5 s, gyro 0.3 | 1.431 | 1.298 | 1.621 | 1.273 | 0.362 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.115 | 1.989 | 1.690 | 2.028 | 0.620 | 3.72 / 2.65 | 1.73 / 1.84 | 0.89 / 0.58 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | frozen | 2.116 | 1.989 | 1.689 | 2.029 | 0.619 | 3.72 / 2.65 | 1.73 / 1.84 | 0.89 / 0.57 | 6/6 |
| all 17, seed 42 | entry | 1.672 | 1.541 | 1.646 | 1.539 | 0.453 | 3.12 / 2.98 | 1.44 / 1.56 | 0.69 / 0.62 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | frozen | 1.672 | 1.541 | 1.645 | 1.539 | 0.452 | 3.12 / 2.98 | 1.44 / 1.56 | 0.69 / 0.61 | 17/17 |
| all 17, seed 1 | entry | 1.685 | 1.550 | 1.786 | 1.551 | 0.450 | 3.16 / 3.30 | 1.46 / 1.71 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | frozen | 1.685 | 1.550 | 1.786 | 1.552 | 0.449 | 3.16 / 3.30 | 1.46 / 1.71 | 0.68 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 1.661 | 1.531 | 1.825 | 1.529 | 0.440 | 3.09 / 3.43 | 1.46 / 1.70 | 0.67 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | frozen | 1.661 | 1.531 | 1.825 | 1.528 | 0.440 | 3.09 / 3.43 | 1.46 / 1.70 | 0.67 / 0.61 | 17/17 |
| all 17, seed 2026 | entry | 1.669 | 1.532 | 1.708 | 1.545 | 0.446 | 3.05 / 3.11 | 1.50 / 1.64 | 0.68 / 0.61 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | frozen | 1.669 | 1.532 | 1.708 | 1.545 | 0.446 | 3.05 / 3.11 | 1.50 / 1.64 | 0.68 / 0.60 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s42_accel_first euroc:V1_03_difficult ate_rmse_m 1.69 vs 1.58; s42_accel_first euroc:V1_03_difficult checkpoints_m 1.61 vs 1.50; s42_accel_first euroc:V1_03_difficult dropout_m 1.54 vs 1.35; s1_accel_first euroc:V1_03_difficult checkpoints_m 1.52 vs 1.50; s1_accel_first euroc:V1_03_difficult dropout_m 1.44 vs 1.41; s7_accel_first euroc:V1_03_difficult ate_rmse_m 1.78 vs 1.58; s7_accel_first euroc:V1_03_difficult checkpoints_m 1.75 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.66 vs 1.42; s2026_accel_first euroc:V2_01_easy position_drift_1s_m 0.36 vs 0.34; s2026_accel_first euroc:V1_03_difficult ate_rmse_m 1.63 vs 1.58; s2026_accel_first euroc:V1_03_difficult checkpoints_m 1.62 vs 1.50; s2026_accel_first euroc:V1_03_difficult dropout_m 1.48 vs 1.34

**Not kept.** TUM-VI blind improves by about 0.01 m (seed 42: 0.62 → 0.61; seed 2026: 0.61 → 0.60). Checkpoints,
dropouts and every drone number are unchanged at all four seeds, so it fails "better on checkpoints".

The "not better than standing still" lines above are the shipped entry's own V1_03 losses at three seeds, not new
ones.

## Damage-aware trust

Scaling the network's velocity trust and the accelerometer's tilt trust by the measured noise can only move the
score as far as changing those trusts does:

- **Network trust:** `--model-vref` 0.3-5 changes ATE by under 1 % ([`network_calibration.md`](network_calibration.md)).
- **Tilt trust:** halving or doubling the gravity-correction gain changes development by under 0.01 m, with TUM-VI
  unchanged:


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | kp 0.15 | 1.428 | 1.290 | 1.630 | 1.267 | 0.362 | 2.72 / 3.20 | 1.30 / 1.44 | 0.59 / 0.64 | 11/11 |
| dev, seed 42 | kp 0.6 | 1.434 | 1.306 | 1.620 | 1.279 | 0.365 | 2.73 / 3.20 | 1.31 / 1.42 | 0.59 / 0.64 | 11/11 |
| dev, seed 42 | kp 0.1, tilt-lp 0.3 | 1.428 | 1.292 | 1.637 | 1.264 | 0.366 | 2.72 / 3.21 | 1.29 / 1.46 | 0.59 / 0.64 | 11/11 |

**Not built.** A noise-scaled schedule would act within those same ranges. vortexxx's large ablation (TUM-VI 0.54 →
1.25 m without it) belongs to a filter that integrates the accelerometer with no learned velocity, where process
noise decides everything. In ours the network's velocity and the pull to the start dominate.
