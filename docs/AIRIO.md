# AirIO-style input for the velocity network (branch `airio-debadrita`)

Goal: after `evo`'s SE(3) alignment, an estimate that stays near its start can only beat "stand still" through how
well the trajectory *shape* matches, and every filter setting has been tried (`docs/RESEARCH_REVIEW.md`). So the
network is the lever. This branch changes what the network reads, following AirIO.

**AirIO** is Qiu et al., *Learning Inertial Odometry with Enhanced IMU Feature Observability*, RA-L 2025,
[arxiv 2501.15659](https://arxiv.org/abs/2501.15659). The id 2506.10607 cited in the outside review is a different
paper. Its point, for drones: rotating the IMU into a global frame throws away what the body-frame signal says
about the motion (thrust along the body axes, attitude through the gravity component). Feed the body-frame IMU with
gravity kept, plus the attitude, and predict body-frame velocity.

**Default since the last commit of this branch**: `src/velnet_weights.bin` is the AirIO network below (called
`velnet_airio.bin` in the tables); the previous network is `src/velnet_weights_world.bin`. Dropping that commit
restores the old default.

## Result in one paragraph

On validation (EuRoC MH_04 + V1_03, seeds 1-3, C++ entry) ATE drops from **4.06 to 3.82 m** (stand still 4.17);
on the held-out set (MH_05, V2_03, room6, run once after the model was frozen) from **3.19 to 3.08 m** (stand still
3.28). Checkpoints and dropouts improve on both sets as well. Against stand still the margin grows from about 3 % to
about 6 % on held-out and from 3 % to 8 % on validation. What gets worse: the blind 5 s on validation (2.58 vs
2.54 m; better on held-out, 2.34 vs 2.36), the handheld TUM-VI rooms (room5 0.89 vs 0.83, room6 0.88 vs 0.84,
still ahead of stand still), and robustness to an IMU mounted another way (below). Runtime 6.7 ms per second of IMU
and about 5.2 MB peak in C++.

## Numbers

Metres, mean over seeds 1-3 of `tools/degrade_imu.py`, `tools/eval_track3.py` with `PF_ENTRY=cpp` (the C++ build).
Columns: ATE (SE(3)-aligned RMSE), checkpoints, blind 5 s, dropouts, RPE over 1 s.

**Validation** (what every choice was made on):

| | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current default (`velnet_weights.bin`, world-frame input) | 4.06 | 3.97 | **2.54** | 3.84 | 0.99 |
| stand still (`--v-max 0.0000001`) | 4.17 | 4.05 | 2.72 | 3.95 | 0.99 |
| **AirIO input (`velnet_airio.bin`)** | **3.82** | **3.63** | 2.58 | **3.82** | **0.90** |
| per recording: MH_04 default / AirIO | 6.63 / 6.32 | 6.45 / 6.06 | 3.91 / 3.97 | 6.82 / 6.69 | |
| per recording: V1_03 default / AirIO | 1.50 / 1.32 | 1.48 / 1.21 | 1.18 / 1.19 | 0.86 / 0.95 | |
| room5 (not in the validation command; the only preview of room6) default / AirIO | 0.83 / 0.89 | 0.82 / 0.91 | 0.67 / 0.63 | 0.98 / 0.93 | |

**Held-out** (reported once, after the model was frozen; the default reproduces the status page's 3.19 / 2.93 /
2.36 / 3.27 exactly):

| | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current default | 3.19 | 2.93 | 2.36 | 3.27 | 0.83 |
| stand still | 3.28 | 3.03 | 2.38 | 3.27 | 0.85 |
| **AirIO input** | **3.08** | **2.86** | **2.34** | **3.15** | **0.80** |

| Held-out per recording | default | stand still | AirIO |
|---|---|---|---|
| EuRoC MH_05 (ATE / checkpoints / blind / dropouts) | 6.72 / 6.11 / 4.04 / 6.94 | 6.80 / 6.17 / 4.18 / 6.92 | **6.42 / 5.86** / 4.04 / **6.73** |
| EuRoC V2_03 | 2.02 / 1.77 / 2.65 / 1.98 | 2.05 / 1.85 / 2.60 / 1.94 | **1.95 / 1.72** / 2.62 / **1.88** |
| TUM-VI room6 | **0.84 / 0.90** / 0.41 / 0.90 | 1.00 / 1.09 / 0.37 / 0.94 | 0.89 / 1.00 / **0.37 / 0.83** |

All runs wrote exactly one pose per input row.

**Window level** (end-velocity RMSE on validation windows, m/s; what the network alone does):

| | AirIO 1 s | AirIO 2 s | old world-frame model | predict zero |
|---|---:|---:|---:|---:|
| MH_04 | **0.66** | 0.73 | 1.05 | 1.14 |
| V1_03 | **0.63** | 0.64 | 0.72 | 0.86 |
| room5 | 0.78 | 0.86 | **0.64** | 1.04 |

On the drone the body-frame network removes 40 % of the velocity error instead of 10 %; on the handheld room it is
worse than the old input, which matches AirIO's own claim (body frame for drones).

## The network

- **Input** (10 × 200, 1 s at 200 Hz, one sample per slot), all from `main.py`'s own state, no ground truth:
  bias-corrected gyro **before** the gravity correction (body frame), accelerometer with **gravity kept** (body
  frame), our **"up" direction in the body frame** (AirIO's attitude input without the unobservable yaw), and the
  healthy flag. Scales 1 / 0.1 / 1. Unhealthy or missing samples are empty slots, as before.
- **Layers**: unchanged structure: four conv1d (32, 48, 64, 64 channels; kernels 7, 5, 5, 3; stride 2; ReLU;
  BatchNorm folded) → mean pool → dense 64 → 12 outputs. **42,716 parameters** (171 KB; budget 100k).
- **Output**: velocity over the last 0.25 s and displacement over the window, both in the body frame at the window
  end, each with a log-std. `learned_update` rotates them into the world frame with the current orientation; the
  per-axis std goes through the same rotation (`sqrt(diag(R diag(s²) Rᵀ))`), then the heading-doubt shrink and the
  gains are as before.
- **Weights file**: same format; **version 3** marks body-frame input and outputs (the C++ entry now reads the
  version). Version-2 files (`velnet_weights.bin` until now) run exactly as before: output byte-identical to the
  previous `main.py`, and C++ = Python within 1 µm.

## Training (`tools/train_airio.py`)

- Same split and degraded copies as `tools/train_velocity.py` (13 per training recording: clean, two seeds of each
  dropout/death mode, 0.5× and 1.5× noise and bias drift), replayed through `main.py` with `--model-input body`,
  which records exactly the slots the network gets at run time.
- **Targets** from ground truth rotated into the body frame with the ground-truth orientation. Both ground truths
  are the IMU frame: EuRoC's `state_groundtruth_estimate0` by definition; for TUM-VI `mocap0` we checked it (body
  rate from the mocap orientation vs the gyro: correlation 0.993 / 0.997 / 0.999 per axis).
- **Augmentation**: mirror image (y → −y; gyro x, z flip), a 2° random tilt error on the "up" channel (our own tilt
  error is 1° median, 1.5-3° at the 90th percentile, up to 6° on V2_02), windows starting inside a dropout. The
  rising noise, bias drift and dropouts come from the degraded copies. No heading augmentation: the body frame has no
  heading.
- Loss MSE + β-NLL (β = 0.5); AdamW, one-cycle to 2e-3, 20 epochs of 150k windows (GPU, about 6 s per epoch); best
  epoch by validation velocity error.
- **Choice of the shipped seed**, fixed in advance: lowest validation ATE among three training seeds of the chosen
  configuration (3.839 / 3.822 / **3.820**); it was also best on blind 5 s and room5.

## What was tried and lost (validation only)

| Change | Validation ATE / blind | Note |
|---|---|---|
| 2 s window (400 slots) instead of 1 s | 3.86 / 2.67 | worse at window level too |
| Training-seed spread of the chosen config | 3.82-3.84 / 2.58-2.66 | the noise floor of every comparison here |
| No mirror augmentation | — | window error MH_04 0.87 vs 0.66 m/s |
| 4° tilt augmentation, wider network (88k params) | — | no better at window level |
| Random IMU mounting in 50 % of windows (3 seeds) | 3.84-3.90 / 2.58-2.61 | rotated-mount ATE worse (4.12) |
| Random IMU mounting in all windows (3 seeds) | 4.07-4.13 / 2.64-2.72 | the gain is gone: it comes from the platform's body axes |
| Trust the network less (`--model-vref` 0.3, 0.5) | 3.57-3.59 on MH_04 + room5 (1.0: 3.57) | no effect |
| Velocity decay in the blind 5 s 0.3 / 0.5 / 1.0 s | 3.84 / 2.59-2.60 | no effect, as the review found |

## Ceiling (`tools/oracle_ceiling.py`)

Ground-truth body velocity in place of the network (room5, seed 1): ATE 0.61 m, against 0.81 for the old network
and 0.93 for stand still. With that perfect velocity, `--pos-tau` 3 s is still about the best (0 → 1.70, 10 → 0.73,
30 → 1.02): heading lost in dropouts keeps the pull to the start necessary even for a perfect network.

## Limitations

- **IMU mounted another way.** The body-frame network has learned the platform's axes. With the IMU frame rotated by
  a fixed 90° about x (`local_data/rot`, as in `test_unknown_gravity_rotated_input`) validation ATE is 3.98 (old model
  4.06, normal mount 3.82) but the blind 5 s is 2.95 vs 2.66. If the organisers' "unknown gravity alignment" means a
  remounted or rotated IMU frame, not just an unknown starting attitude, the old world-frame model is the safer
  choice. Training on random mountings removes the gain entirely (table above).
- **Handheld rooms**: worse than the old model on room5 and room6 (still better than stand still on room6).
- **Blind 5 s on validation**: 0.04 m worse; on held-out 0.02 m better. It is a property of the position error at
  the moment the IMU dies, not of the decay settings.
- Our degradation is still the stand-in `tools/degrade_imu.py`.
- The `test_peak_ram_under_8_mb` contract test only runs on macOS. On this Windows laptop the peak working set
  (static build) is about 5.2 MB with either model, polled while the process runs.

## Budgets (C++, Windows 11 laptop, static MSYS2 g++ 14 -O2 build, best of 7)

| | per second of IMU | peak working set |
|---|---:|---:|
| no network (`--model none`) | 1.8 ms | — |
| old world-frame network | 4.6 ms (7.1 ms before the loop change below) | 5.2 MB |
| AirIO network | **6.7 ms** (11.7 ms before) | 5.2 MB |

The convolution in `main.cpp` now pads once and puts the output position in the inner loop, so it vectorises; for
every output the products are added in the same order as before, and the output is byte-identical to the previous
build for both weight files.

## Checks

- Python vs C++ with `velnet_airio.bin` on three degraded recordings (MH_04 seed 1 missing/zeros, V1_03 seed 3
  zeros/NaN, room5 seed 2 NaN/frozen): same rows and timestamps, positions within 1e-6 m (the printed precision),
  identical quaternions.
- `PF_ENTRY=cpp pytest tests/stress/test_main_contract.py` and the Python run: 23 passed, 1 skipped (the macOS-only
  RSS test), with the new weights in the default slot.

## Reproduce

```
python tools/fetch_euroc.py --out local_data/euroc --pause 30     # TUM-VI rooms: imu0 + mocap0 into local_data/tumvi
python tools/train_airio.py prep --jobs 5
bash tools/sweep_airio.sh r0_s2 --seed 2                         # train + export + validation scores
cp runs/airio/r0_s2/velnet_body.bin src/velnet_airio.bin
make -C src
PF_ENTRY=cpp python tools/eval_track3.py --data euroc:MH_04_difficult euroc:V1_03_difficult --seeds 1 2 3 -- --model src/velnet_airio.bin
```

## Not done

- Gyro denoiser in front of the orientation (Brossard et al.).
- Final-submission retrain on all 17 recordings (cannot be validated without leakage).
