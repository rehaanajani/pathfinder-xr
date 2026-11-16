# Learned velocity for the IMU-only Track 3 entry

Branch `learned-velocity-debadrita`. Goal: make IMU-only tracking beat a "stand still" estimate on degraded
EuRoC / TUM-VI input (see `docs/TRACK3_BRIEF.md`).

## Result in one paragraph

On the held-out recordings (MH_05, V2_03, room6; degraded with seeds 1-3; never used for any choice),
`python src/main.py <input> --model src/velnet_weights.bin` scores **3.19 m ATE against 3.28 m for stand still and
4.54 m for the current `main.py`**. It is ahead of stand still on ATE, checkpoints, the blind 5 s and RPE, and level on
dropouts, but the margin is small (about 3 %), and on the held-out set the retuned filter *without* the network does
just as well. The clear win is over the current `main.py`; the win over stand still is real but thin. Why, and what
would change it, is in *What limits it*.

## Held-out numbers (reported once)

Metres, mean over seeds 1-3 of `tools/degrade_imu.py`, scored by `tools/eval_track3.py` (SE(3)-aligned ATE RMSE;
checkpoints, blind 5 s and dropouts as defined in `tools/score_track3.py`; translational RPE over 1 s).

Mean of the three held-out recordings:

| | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| (a) current `main.py` | 4.54 | 4.19 | 4.03 | 4.47 | 0.97 |
| (b) stand still (`--v-max 0.0000001`) | 3.28 | 3.03 | 2.38 | 3.27 | 0.85 |
| retuned filter, no network (`--preset learned`) | 3.19 | 2.92 | 2.31 | 3.26 | 0.88 |
| **`--model src/velnet_weights.bin`** | **3.19** | **2.93** | **2.36** | **3.27** | **0.83** |

Per recording:

| EuRoC MH_05_difficult | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current `main.py` | 6.64 | 5.97 | 4.08 | 6.98 | 1.18 |
| stand still | 6.80 | 6.16 | 4.18 | 6.92 | 1.15 |
| retuned filter, no network | 6.70 | 6.11 | 4.05 | 6.92 | 1.15 |
| `--model` | 6.72 | 6.11 | 4.04 | 6.94 | 1.14 |

| EuRoC V2_03_difficult | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current `main.py` | 5.24 | 5.22 | 4.70 | 4.49 | 1.13 |
| stand still | 2.04 | 1.85 | 2.59 | 1.94 | 0.84 |
| retuned filter, no network | 2.01 | 1.68 | 2.49 | 1.88 | 0.95 |
| `--model` | 2.02 | 1.77 | 2.65 | 1.97 | 0.85 |

| TUM-VI room6 | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current `main.py` | 1.73 | 1.38 | 3.31 | 1.93 | 0.60 |
| stand still | 1.00 | 1.09 | 0.37 | 0.94 | 0.56 |
| retuned filter, no network | 0.87 | 0.96 | 0.38 | 0.99 | 0.53 |
| `--model` | 0.84 | 0.90 | 0.41 | 0.90 | 0.49 |

All 36 held-out runs produced exactly one pose per input row with unit quaternions.

## Validation numbers (what the choices were made on)

Mean of MH_04_difficult, V1_03_difficult, room5, seeds 1-3:

| | ATE | checkpoints | blind 5 s | dropouts | RPE 1 s |
|---|---:|---:|---:|---:|---:|
| current `main.py` | 6.58 | 5.25 | 8.72 | 6.72 | 1.29 |
| stand still | 3.09 | 2.99 | 2.03 | 2.97 | 0.94 |
| retuned filter, no network | 3.16 | 2.96 | 1.98 | 3.20 | 1.07 |
| `--model` | 2.99 | 2.92 | 1.92 | 2.89 | 0.90 |

On validation the network helped beyond the retuned filter (room5: 0.83 vs 1.20 m ATE; V1_03: 1.50 vs 1.67); on the
held-out set that extra gain did not repeat. Treat the network's own contribution as unproven and the filter retune
as the dependable part.

## What is in the branch

| File | What |
|---|---|
| `tools/train_velocity.py` | `prep` (degraded copies → window cache), `train`, `export` |
| `src/velnet.py` | weights file format and the numpy reference forward pass (no PyTorch / ONNX at run time) |
| `src/velnet_weights.bin` | the trained network: 27,020 parameters, 108,232 bytes |
| `src/main.py` | `--model`, the `learned` preset, and the filter changes listed below |
| `tools/eval_track3.py` | data folder can be `local_data/` in the repo or `$PFX_DATA` |
| `tools/fetch_euroc.py` | `--pause` between requests (the ETH server answers 429 to bursts) |

## The network

- **Input** (7 × 200): the last 1 s at 200 Hz of bias-corrected gyro and specific force minus g, rotated into
  `main.py`'s own gravity-aligned frame (its orientation, no ground truth), plus a "sample is healthy" channel.
  Unhealthy or missing samples are empty slots, so the window stays on a uniform time grid and the network can be
  called again 0.25 s after a dropout ends.
- **Layers**: four 1-D convolutions (24, 32, 48, 64 channels; kernels 7, 5, 5, 3; stride 2; ReLU; BatchNorm folded
  in at export) → global average pool → dense 64 → 12 outputs.
- **Output**: displacement over the window (3) and velocity over its last 0.25 s (3), each with a log standard
  deviation (the confidence), all in the same gravity-aligned frame.
- **Cost**: 626k multiply-adds per call, 10 calls per second of IMU. Weights 108 KB plus a 200 × 7 float window.
  The file layout is documented at the top of `src/velnet.py`; `export --header` also writes it as a C array.

## Training

- **Data**: for every training recording, 13 copies made with `tools/degrade_imu.py`: one clean; two seeds for each of
  the four dropout/death modes (missing, zeros, frozen, NaN); and four at 0.5× and 1.5× the default noise and bias
  drift with four dropouts. Each copy is replayed through `src/main.py` itself, which records the exact window slots
  it would give the network, so tilt error, heading drift and dropouts in training are the real ones.
- **Targets**: ground-truth displacement / velocity rotated into the estimator's frame at the window end.
- **Augmentation**: random heading, mirror image, and windows that start inside a dropout.
- **Loss**: MSE + β-NLL (β = 0.5). AdamW, one-cycle to 2e-3, 20 epochs of 150k windows, best epoch by validation.
- **Split**: train MH_01-03, V1_01-02, V2_01-02, room1-4; validate MH_04, V1_03, room5; hold out MH_05, V2_03, room6.
  The degradation seeds used for training (≥ 100) are never the scoring seeds (1-3).

Window-level accuracy of the chosen checkpoint on validation (end-velocity RMSE, against predicting zero):

| | network | predict zero |
|---|---:|---:|
| room5 | 0.64 m/s | 1.04 m/s |
| V1_03 | 0.72 m/s | 0.86 m/s |
| MH_04 | 1.05 m/s | 1.14 m/s |

The reported standard deviation is over-confident (squared z-score 1.6-2.5 on validation instead of 1).

Choices made on validation and what lost: a 49k-parameter network was no better than 27k; 2 s and 4 s windows
(averaged slots) were worse than 1 s; using the displacement head as a position measurement was worse than using
the velocity head alone; a drag-based extra input for the drone gave no signal.

## How `main.py` uses it

`--model` switches the option defaults to the `learned` preset (`PRESETS` in `src/main.py`); every value can still
be overridden, and without `--model` the filter behaves exactly as before.

1. Every 20 healthy samples the network's velocity replaces the filter velocity (gain 1, down-weighted by its own
   standard deviation); the accelerometer only fills in between calls.
2. **Heading doubt**: each second of dropout adds 2 rad² of heading variance and the network's horizontal velocity
   is shrunk by exp(−variance / 2). An IMU cannot see how far the device turned during a 2 s gap, and a velocity
   in the wrong direction is worse than none.
3. **Position reverts to the start point** with a 3 s time constant (`--pos-tau`). This is the same prior that makes
   stand still strong; with it the estimate tracks short-term motion and cannot drift away.
4. **Blind 5 s and dropouts**: the last velocity decays with a 0.7 s time constant.

Filter changes in the same preset, needed because the network's input is only as good as the orientation:

| Change | Effect on training / validation recordings |
|---|---|
| Lower gravity gain (kp 0.3), gyro bias learned only in calm moments, 0.1 s low-pass on the gravity reference, fast re-levelling for ~1 s after a dropout | tilt error on healthy samples about 6° → 2° (rooms), 3.5° → about 2° (EuRoC) |
| "Still" detected from the spread of the readings instead of the gyro magnitude, and a pre-scan of the first 60 s for the first still stretch | EuRoC's 0.08 rad/s gyro bias is now found; clean-flight heading drift fell from 2-3°/s to about 3° in total |

## Budgets

Measured on a 2-core cloud VM, room5 (142 s of IMU):

| | Python `main.py` | with `--model` |
|---|---:|---:|
| time per 1 s of IMU | 4.3 ms | 11.5 ms |
| peak RSS | — | 35 MB (Python + numpy) |

The time budget holds even in Python. The 8 MB memory budget does not, as the brief already notes; it needs the
C++ port, where the network adds about 0.12 MB. **The C++ forward pass is not written or measured yet.**

## What limits it

- **Velocity is barely observable here.** The network removes about 40 % of the velocity error in rooms and 10-15 % on
  the drone, compared with predicting zero. Integrated over a minute that is not enough to follow a 7 m hall flight, so MH_04 / MH_05 stay within
  2 % of stand still for every method.
- **Too little training data**: 11 recordings, about 25 minutes of distinct motion. Validation error stops improving
  after a few epochs while training error keeps falling.
- **Dropouts lose the heading for good**, which is why the estimate is deliberately pulled toward "stay put" after them.
- **Our degradation is a stand-in.** All numbers depend on `tools/degrade_imu.py`; the organizers' script may differ.
- **The scoring alignment is assumed** (SE(3) alignment, as `evo_ape -a`).

## What I would do next

1. Port the forward pass to C++ and measure memory and time there.
2. For the final submission, retrain on all 17 recordings (the hidden set is drawn from the same two datasets);
   that model cannot be validated without leakage, so keep these numbers as the honest estimate.
3. More handheld training data with ground truth would help the rooms most.

## Reproduce

```
python tools/fetch_euroc.py --out local_data/euroc --pause 30        # TUM-VI rooms: mav0/imu0 + mav0/mocap0 into local_data/tumvi
python tools/train_velocity.py prep
python tools/train_velocity.py train --out runs/vel/final
python tools/train_velocity.py export --run runs/vel/final --out src/velnet_weights.bin
D="euroc:MH_05_difficult euroc:V2_03_difficult tumvi:room6"
python tools/eval_track3.py --data $D --seeds 1 2 3 --tag model -- --model src/velnet_weights.bin
python tools/eval_track3.py --data $D --seeds 1 2 3 --tag physics
python tools/eval_track3.py --data $D --seeds 1 2 3 --tag still -- --v-max 0.0000001
```

`local_data/` is git-ignored; no dataset file is in the branch. The existing `tests/` suite was not run to
completion here (it did not finish within the time I gave it); it does not import `src/` or `tools/`.
