# Calibrating the network's uncertainty (20 Nov 2026): not kept

Idea: the stated velocity standard deviation is too small (training logs: `vel_z2` 2-9, where 1 is calibrated);
scale it on validation so the filter trusts the network the right amount.

## How miscalibrated, and where

`python tools/network_calibration.py --run runs/airio/r0_s2` on the window cache (our degradation), z² = mean of
(error / stated std)² per body axis ([`network/calibration.json`](network/calibration.json)):

| Recording | Split | z² (x, y, z body axes) | RMS error (m/s) | RMS stated std (m/s) |
|---|---|---|---|---|
| euroc:MH_01_easy | train | 0.96, 1.28, 0.75 | 0.14, 0.25, 0.21 | 0.14, 0.27, 0.23 |
| euroc:MH_02_easy | train | 0.85, 1.03, 0.92 | 0.13, 0.26, 0.25 | 0.14, 0.28, 0.24 |
| euroc:MH_03_medium | train | 0.98, 1.36, 0.81 | 0.16, 0.34, 0.26 | 0.16, 0.32, 0.28 |
| euroc:V1_01_easy | train | 0.90, 1.00, 0.80 | 0.12, 0.26, 0.16 | 0.13, 0.25, 0.19 |
| euroc:V1_02_medium | train | 1.16, 0.88, 0.96 | 0.17, 0.25, 0.24 | 0.16, 0.26, 0.25 |
| euroc:V2_01_easy | train | 0.85, 0.71, 0.91 | 0.11, 0.16, 0.18 | 0.12, 0.20, 0.19 |
| euroc:V2_02_medium | train | 1.09, 0.86, 0.82 | 0.16, 0.22, 0.21 | 0.16, 0.23, 0.23 |
| tumvi:room1 | train | 0.83, 0.83, 0.94 | 0.18, 0.16, 0.14 | 0.19, 0.18, 0.14 |
| tumvi:room2 | train | 0.91, 0.87, 0.74 | 0.15, 0.12, 0.08 | 0.16, 0.12, 0.09 |
| tumvi:room3 | train | 0.82, 1.00, 0.78 | 0.13, 0.13, 0.07 | 0.15, 0.14, 0.08 |
| tumvi:room4 | train | 0.86, 0.83, 0.83 | 0.13, 0.13, 0.11 | 0.15, 0.14, 0.12 |
| euroc:MH_04_difficult | validation | 6.22, 2.83, 1.77 | 0.29, 0.51, 0.36 | 0.15, 0.31, 0.27 |
| euroc:V1_03_difficult | validation | 5.75, 2.18, 3.51 | 0.34, 0.33, 0.41 | 0.15, 0.24, 0.23 |
| tumvi:room5 | validation | 7.18, 11.06, 4.95 | 0.49, 0.51, 0.28 | 0.19, 0.17, 0.14 |

Calibrated on every training recording, over-confident 2-11x on every axis of the three validation recordings: the
network's confidence does not transfer to data it has not seen. The over-confidence is similar across axes (pooled
validation scale ≈ 2.5 / 2.3 / 1.9), so one factor captures it, and scaling the std by one factor is the same as
scaling `--model-vref` (gain = vref² / (vref² + σ²)).

## Effect on the score

Organisers' `degrade_imu.py` (their seed 42 unless stated), accel-first input, C++ entry (`tools/eval_organiser.py`;
same output as `main.py` within 1e-6 m), metres. Choices on training and validation recordings only; the held-out
recordings (MH_05, V2_03, room6) and the fresh seeds 1, 7, 2026 were not used. Raw results:
[`network/`](network/); tables generated with `python tools/network_tables.py`.

Validation (MH_04, V1_03, room5), seed 42:

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| vref 5 | 2.407 | 2.403 | 1.528 | 2.303 |
| vref 2 | 2.406 | 2.402 | 1.525 | 2.303 |
| vref 1 (current) | 2.404 | 2.401 | 1.518 | 2.301 |
| vref 0.5 | 2.396 | 2.396 | 1.495 | 2.296 |
| vref 0.3 | 2.385 | 2.387 | 1.476 | 2.291 |
| vref 0.2 | 2.382 | 2.382 | 1.496 | 2.292 |
| vref 0.1 | 2.472 | 2.447 | 1.728 | 2.374 |
| vref 0.05 | 2.705 | 2.654 | 2.032 | 2.599 |
| no network | 3.311 | 3.211 | 3.024 | 3.161 |

All 14 training + validation recordings, seed 42:

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| vref 1 (current) | 1.895 | 1.766 | 1.824 | 1.731 |
| vref 0.5 | 1.898 | 1.769 | 1.793 | 1.731 |
| vref 0.3 | 1.910 | 1.780 | 1.822 | 1.740 |
| vref 0.2 | 1.937 | 1.805 | 1.908 | 1.759 |

The 14 recordings prefer the current trust (11 of them are the network's training data, where its std is
calibrated); validation alone prefers less. The candidate both agree on is vref 0.5. Over five selection seeds of
the organisers' script (101-105; not 42 and not the fresh check seeds), validation, paired with vref 1:

| Seed | ATE (first) | Checkpoints (first) | Blind 5 s (first) | Dropouts (first) | ATE (second) | Checkpoints (second) | Blind 5 s (second) | Dropouts (second) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2.374 | 2.390 | 1.596 | 2.315 | 2.372 | 2.386 | 1.607 | 2.315 |
| 102 | 2.341 | 2.331 | 1.696 | 2.276 | 2.343 | 2.334 | 1.706 | 2.279 |
| 103 | 2.405 | 2.391 | 1.815 | 2.294 | 2.404 | 2.391 | 1.823 | 2.296 |
| 104 | 2.365 | 2.376 | 1.534 | 2.302 | 2.356 | 2.371 | 1.502 | 2.293 |
| 105 | 2.420 | 2.418 | 1.544 | 2.322 | 2.414 | 2.415 | 1.509 | 2.317 |

Paired differences (second minus first) over 15 recording-seed pairs:

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| mean | -0.003 | -0.001 | -0.008 | -0.002 |
| standard error | 0.002 | 0.002 | 0.009 | 0.002 |
| pairs where second is lower | 12 / 15 | 11 / 15 | 7 / 15 | 9 / 15 |

**Not kept.** vref 0.5 is consistently a hair better on ATE (0.003 m, about 0.1 %) and level on blind within its
standard error: far below anything the score can resolve, so it is not worth a change to the shipped entry. The
network matters (no network: ATE 3.31 vs 2.40), but how strongly the filter trusts it hardly does between vref 0.3
and 5.
