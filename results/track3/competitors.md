# Competitors: head-to-head on the same degraded files

Every public Track 3 entry with a runnable Python `main.py`, run on **exactly the same inputs** as ours and scored with
**our** `tools/score_track3.py` (same SE(3) alignment, same checkpoint, blind, dropout and drift windows). Generated
from `competitors_raw.json` by `tools/competitor_tables.py`; nothing below is typed by hand except the prose.

**Ours here is `main` @ 4122d58, before rotor-drag aiding** (the `rotor-drag` branch was not on the remote on 19 Nov,
when this was run). Re-run the `ours` row after it is merged:
`python tools/competitor_scoreboard.py run --entries ours --fresh --tag rotor ...`.

## Setup

* **Inputs** (`tools/competitor_scoreboard.py inputs`): the 11 EuRoC + 6 TUM-VI recordings, damaged by the organisers'
  `degrade_imu.py` (sha256 `cc7398db…2f31`, the copy in the VIO-Edge repo, identical to the emailed one). Four sets:
  seed 42 with accel-first columns (the script's documented order), seed 42 with the EuRoC file as is (gyro first),
  and seeds 1 and 7 accel-first (only `SEED = ` changed). 68 files; every entry reads the same 68.
* **Runs** (`... run`): one process per file under `/usr/bin/time -l`, BLAS/OpenMP threads pinned to 1, Apple M4,
  Python 3.11.17, numpy 2.4.6. Competitor code ran from its own clone in a separate virtualenv (numpy + pandas only),
  after reading it (no network access, no file deletion outside its own temp files). Nothing of theirs is in this repo.
* **Standing still**: the start pose on every row, written by the harness.
* **Contract**: one pose per input row, timestamps equal to the input's, 8 finite columns, unit quaternions.
* **Heap** (`... heap`): tracemalloc around a full execution of each script, numpy imported and the source compiled
  first. *Warm*: the script is executed once untraced first, so lazily imported library modules are not counted (this
  is how vortexxx measures itself); *cold*: the first execution is traced. Which one the organisers use is not known.

| Team | Repo | Commit run | Licence |
|---|---|---|---|
| SRMIST_vortexxx | [`RaktimChandra/pathfinder`](https://github.com/RaktimChandra/pathfinder) | `a67f4ee` | none found |
| ZeroDrift | [`Gem2005/VITChennai_ZeroDrift`](https://github.com/Gem2005/VITChennai_ZeroDrift) | `9c03e6d` | none found |
| VIO-Edge | [`rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation`](https://github.com/rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation) | `56abee2` | MIT |
| Vortex | [`ArnavJain2005/Vortex`](https://github.com/ArnavJain2005/Vortex) | `3559fe4` | MIT |
| TrustNav | [`Am-berneal/TrustNav-GPS-Denied-Navigation`](https://github.com/Am-berneal/TrustNav-GPS-Denied-Navigation) | `bec7863` | none found |
| Ridhima Chopra | [`ridhima936/PathFinder-IMU-Trajectory`](https://github.com/ridhima936/PathFinder-IMU-Trajectory) | `b90f29c` | none found |
| VITC_SingleThread | [`jeeva-m-21/samsung-track3-pathfinder`](https://github.com/jeeva-m-21/samsung-track3-pathfinder) | `19a0651` | none found |

VITC_SingleThread was not run (C++ only, which the FAQ excludes). Ridhima Chopra's entry is plain dead reckoning
(hundreds of metres); it is listed for completeness. `rishita2201/Samsung_XR_Hackathon` was still empty.

## Headline (seed 42, accel-first: the organisers' documented setting)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.71 | 4.92 | 0.75 | 2.31 | 17/17 |
| SRMIST_vortexxx | 1.41 | 3.97 | 0.50 | 1.84 | 16/17 |
| ZeroDrift | 1.31 | 3.08 | 0.80 | 1.65 | 17/17 |
| VIO-Edge | 1.79 | 4.57 | 0.71 | 2.23 | 17/17 |
| Vortex | 1.89 | 4.83 | 2.08 | 2.82 | 9/17 |
| TrustNav (as shipped) | 2.01 | 5.14 | 1.56 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.63 | 286.09 | 291.98 | 242.48 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

* **We are third of the eligible entries on ATE**: ZeroDrift 1.65 m, vortexxx 1.84 m, VIO-Edge 2.23 m, ours 2.31 m,
  standing still 2.62 m. The order is the same on all four input sets (tables below).
* **We are one of three entries that beat standing still on all 17** (with ZeroDrift and VIO-Edge; vortexxx is
  worse than standing still on V1_03, 1.62 vs 1.58 m). Vortex and TrustNav are worse than standing still on most recordings.
* Our CPU (7.5 ms per second of IMU) is the highest of the field, still well under 20 ms; our heap is among the lowest.

## Where we lose (this decides section B)

Group means, metres, ours vs the rival, every metric where the rival is at least 0.05 m better:

* **ZeroDrift**, seed 42 accel-first: Vicon rooms ATE 1.71 vs 1.31; Vicon rooms checkpoints 1.55 vs 1.14; Vicon rooms blind 5 s 2.10 vs 2.03; Vicon rooms dropouts 1.59 vs 1.20; Vicon rooms drift 1 s 0.50 vs 0.25; Machine Hall ATE 4.92 vs 3.08; Machine Hall checkpoints 4.51 vs 2.92; Machine Hall blind 5 s 4.36 vs 2.24; Machine Hall dropouts 4.50 vs 2.73; Machine Hall drift 1 s 0.73 vs 0.33; TUM-VI drift 1 s 0.51 vs 0.43.
* **SRMIST_vortexxx**, seed 42 accel-first: Vicon rooms ATE 1.71 vs 1.41; Vicon rooms checkpoints 1.55 vs 1.22; Vicon rooms dropouts 1.59 vs 1.31; Vicon rooms drift 1 s 0.50 vs 0.30; Machine Hall ATE 4.92 vs 3.97; Machine Hall checkpoints 4.51 vs 3.90; Machine Hall blind 5 s 4.36 vs 1.65; Machine Hall dropouts 4.50 vs 3.63; Machine Hall drift 1 s 0.73 vs 0.47; TUM-VI ATE 0.75 vs 0.50; TUM-VI checkpoints 0.72 vs 0.38; TUM-VI dropouts 0.71 vs 0.41; TUM-VI drift 1 s 0.51 vs 0.30.
* **VIO-Edge**, seed 42 accel-first: Machine Hall ATE 4.92 vs 4.57; Machine Hall checkpoints 4.51 vs 4.18; Machine Hall blind 5 s 4.36 vs 3.87; Machine Hall dropouts 4.50 vs 4.09; TUM-VI checkpoints 0.72 vs 0.67; TUM-VI dropouts 0.71 vs 0.65; TUM-VI drift 1 s 0.51 vs 0.45.

The same gaps appear on seeds 1 and 7 and on the EuRoC column order (full list at the end). In short:

1. **Machine Hall, every metric.** ZeroDrift's ATE is 3.08 m against our 4.92 m (standing still 5.25). This is the
   largest gap in metres.
2. **Blind 5 s on the Machine Hall.** vortexxx 1.65 m, ZeroDrift 2.24 m, ours 4.36 m (standing still 4.46). Our blind
   error is set by the position error at the moment of failure; theirs is lower because their position there is better.
3. **TUM-VI handheld.** vortexxx 0.50 m ATE and 0.38 m checkpoints against our 0.75 / 0.72 m.
4. **Vicon rooms.** ZeroDrift 1.31 m and vortexxx 1.41 m against our 1.71 m.
5. **Drift over 1 s.** ZeroDrift and vortexxx: 0.25 / 0.30 m (Vicon), 0.33 / 0.47 m (Machine Hall), 0.43 / 0.30 m
   (TUM-VI), against our 0.50 / 0.73 / 0.51 m.

Where we hold: TUM-VI blind 5 s (ours 0.60 m, vortexxx 0.64, ZeroDrift 1.38), Vicon blind 5 s (2.10, level with the
field), contract (68/68) and heap.

## Why they win (from reading their code and reports)

Both leaders are **non-causal**: they use the whole file, which the FAQ allows. We run a causal filter.

* **vortexxx**: attitude by a complementary filter, then translation by a **Kalman filter + RTS smoother at 2.5 Hz**
  (0.4 s blocks) with a **workspace-centre state** (soft prior σ = 3.5 m on EuRoC, 3 m otherwise), rotor drag with a
  Cauchy-weighted innovation, zero-velocity updates at every rest of ≥ 1.5 s, and process noise that follows the
  organisers' severity ramp `0.02 + 0.98 (t/T)^1.7` plus the measured excess gyro noise. Their ablation: without that
  noise schedule ATE goes 1.39 → 2.51 m (Vicon) and 0.54 → 1.25 m (TUM-VI). They also hard-code the EuRoC platform
  (gravity direction in the sensor frame, gyro bias, drag constants) and switch to a world-frame gravity reference when
  the platform is not recognised (TUM-VI). Blind: hold position (velocity decays with τ = 0.2 s).
* **ZeroDrift**: undoes the reversible damage first (duplicates, stuck runs, spikes; a smoother whose bandwidth follows
  the severity ramp), RK4 sub-steps across missing rows, a two-pass error-state EKF with an accelerometer-scale state,
  then **zero-phase drift removal**: the dead-reckoned position is split into slow drift and motion by penalised least
  squares, with a cut-off that shortens as severity grows, plus rotor drag as displacement information when in flight.
* **VIO-Edge**: closest to us (ESKF + learned velocity), level with us overall, better on Machine Hall and TUM-VI.

Our pull back toward the start is a *causal* version of what both do: it removes slow drift but lags. A zero-phase
(forward-backward) version of the same idea, or an RTS smoother over our translation, attacks gaps 1, 2, 4 and 5 at
once; a severity-following noise level attacks gap 3. That is the proposed order for section B.

## All tables

### seed 42, accel-first (documented order)

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.71 | 4.92 | 0.75 | 2.31 | 17/17 |
| SRMIST_vortexxx | 1.41 | 3.97 | 0.50 | 1.84 | 16/17 |
| ZeroDrift | 1.31 | 3.08 | 0.80 | 1.65 | 17/17 |
| VIO-Edge | 1.79 | 4.57 | 0.71 | 2.23 | 17/17 |
| Vortex | 1.89 | 4.83 | 2.08 | 2.82 | 9/17 |
| TrustNav (as shipped) | 2.01 | 5.14 | 1.56 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.63 | 286.09 | 291.98 | 242.48 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.55 | 4.51 | 0.72 | 2.13 | 17/17 |
| SRMIST_vortexxx | 1.22 | 3.90 | 0.38 | 1.71 | 16/17 |
| ZeroDrift | 1.14 | 2.92 | 0.73 | 1.52 | 17/17 |
| VIO-Edge | 1.67 | 4.18 | 0.67 | 2.06 | 16/17 |
| Vortex | 1.71 | 4.40 | 1.73 | 2.51 | 10/17 |
| TrustNav (as shipped) | 1.71 | 4.72 | 1.18 | 2.41 | 13/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.76 | 1.23 | 2.45 | 11/17 |
| Ridhima Chopra | 116.84 | 215.16 | 216.39 | 180.89 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 2.10 | 4.36 | 0.60 | 2.24 | 12/17 |
| SRMIST_vortexxx | 2.19 | 1.65 | 0.64 | 1.49 | 11/17 |
| ZeroDrift | 2.03 | 2.24 | 1.38 | 1.86 | 10/17 |
| VIO-Edge | 2.16 | 3.87 | 0.70 | 2.15 | 12/17 |
| Vortex | 2.87 | 4.14 | 3.85 | 3.59 | 4/17 |
| TrustNav (as shipped) | 3.26 | 6.26 | 1.82 | 3.63 | 1/17 |
| TrustNav `--no-fill --blind 0` | 3.17 | 5.71 | 1.71 | 3.40 | 1/17 |
| Ridhima Chopra | 351.52 | 649.34 | 651.02 | 544.82 | 0/17 |
| standing still | 2.14 | 4.46 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.59 | 4.50 | 0.71 | 2.13 | 17/17 |
| SRMIST_vortexxx | 1.31 | 3.63 | 0.41 | 1.68 | 16/17 |
| ZeroDrift | 1.20 | 2.73 | 0.73 | 1.48 | 17/17 |
| VIO-Edge | 1.64 | 4.09 | 0.65 | 2.01 | 17/17 |
| Vortex | 1.67 | 4.48 | 1.80 | 2.54 | 11/17 |
| TrustNav (as shipped) | 1.75 | 4.70 | 1.21 | 2.43 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.69 | 1.25 | 2.44 | 9/17 |
| Ridhima Chopra | 126.05 | 230.22 | 234.82 | 195.08 | 0/17 |
| standing still | 1.78 | 4.80 | 1.08 | 2.42 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.50 | 0.73 | 0.51 | 0.57 | 17/17 |
| SRMIST_vortexxx | 0.30 | 0.47 | 0.30 | 0.35 | 17/17 |
| ZeroDrift | 0.25 | 0.33 | 0.43 | 0.34 | 17/17 |
| VIO-Edge | 0.54 | 0.69 | 0.45 | 0.55 | 15/17 |
| Vortex | 0.50 | 0.72 | 0.61 | 0.60 | 13/17 |
| TrustNav (as shipped) | 0.71 | 0.92 | 0.88 | 0.83 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.89 | 0.83 | 0/17 |
| Ridhima Chopra | 5.96 | 7.36 | 8.88 | 7.40 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.77 | – |

### seed 42, EuRoC file as is (gyro first)

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.71 | 4.91 | 0.75 | 2.31 | 17/17 |
| SRMIST_vortexxx | 1.42 | 4.20 | 0.50 | 1.92 | 16/17 |
| ZeroDrift | 1.40 | 3.38 | 0.79 | 1.77 | 17/17 |
| VIO-Edge | 1.77 | 4.75 | 0.71 | 2.27 | 17/17 |
| Vortex | 1.88 | 4.93 | 2.02 | 2.83 | 9/17 |
| TrustNav (as shipped) | 2.01 | 5.18 | 2.53 | 3.13 | 7/17 |
| TrustNav `--no-fill --blind 0` | 3.35 | 5.78 | 2.57 | 3.79 | 2/17 |
| Ridhima Chopra | 848.77 | 16403.11 | 19554.87 | 12025.73 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.56 | 4.47 | 0.73 | 2.12 | 17/17 |
| SRMIST_vortexxx | 1.25 | 3.91 | 0.39 | 1.73 | 17/17 |
| ZeroDrift | 1.18 | 3.07 | 0.73 | 1.58 | 17/17 |
| VIO-Edge | 1.63 | 4.36 | 0.67 | 2.09 | 16/17 |
| Vortex | 1.73 | 4.41 | 1.74 | 2.52 | 10/17 |
| TrustNav (as shipped) | 1.83 | 4.71 | 2.30 | 2.84 | 8/17 |
| TrustNav `--no-fill --blind 0` | 2.92 | 5.03 | 2.31 | 3.33 | 6/17 |
| Ridhima Chopra | 495.26 | 12262.56 | 14024.92 | 8731.40 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 2.09 | 4.36 | 0.60 | 2.23 | 12/17 |
| SRMIST_vortexxx | 2.13 | 3.00 | 0.60 | 1.85 | 10/17 |
| ZeroDrift | 2.46 | 3.80 | 1.44 | 2.49 | 7/17 |
| VIO-Edge | 2.19 | 4.31 | 0.73 | 2.30 | 8/17 |
| Vortex | 2.58 | 5.25 | 3.36 | 3.64 | 3/17 |
| TrustNav (as shipped) | 2.61 | 5.06 | 3.50 | 3.64 | 4/17 |
| TrustNav `--no-fill --blind 0` | 5.73 | 9.20 | 3.97 | 6.13 | 1/17 |
| Ridhima Chopra | 2580.69 | 37613.66 | 45296.19 | 27960.56 | 0/17 |
| standing still | 2.14 | 4.46 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.59 | 4.49 | 0.71 | 2.13 | 17/17 |
| SRMIST_vortexxx | 1.34 | 3.86 | 0.42 | 1.76 | 16/17 |
| ZeroDrift | 1.25 | 3.02 | 0.73 | 1.59 | 17/17 |
| VIO-Edge | 1.61 | 4.36 | 0.64 | 2.08 | 17/17 |
| Vortex | 1.69 | 4.52 | 1.78 | 2.56 | 11/17 |
| TrustNav (as shipped) | 1.81 | 4.67 | 2.28 | 2.82 | 8/17 |
| TrustNav `--no-fill --blind 0` | 2.95 | 5.34 | 2.29 | 3.42 | 2/17 |
| Ridhima Chopra | 594.62 | 13134.48 | 15233.84 | 9449.60 | 0/17 |
| standing still | 1.78 | 4.80 | 1.08 | 2.42 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.50 | 0.75 | 0.51 | 0.58 | 16/17 |
| SRMIST_vortexxx | 0.35 | 0.59 | 0.30 | 0.40 | 17/17 |
| ZeroDrift | 0.32 | 0.49 | 0.43 | 0.41 | 16/17 |
| VIO-Edge | 0.54 | 0.73 | 0.44 | 0.56 | 15/17 |
| Vortex | 0.51 | 0.78 | 0.60 | 0.62 | 13/17 |
| TrustNav (as shipped) | 0.73 | 1.01 | 1.38 | 1.04 | 2/17 |
| TrustNav `--no-fill --blind 0` | 0.92 | 1.06 | 1.42 | 1.14 | 0/17 |
| Ridhima Chopra | 45.57 | 414.72 | 594.24 | 347.79 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.77 | – |

### seed 1, accel-first

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.71 | 4.92 | 0.75 | 2.31 | 17/17 |
| SRMIST_vortexxx | 1.44 | 4.06 | 0.51 | 1.88 | 16/17 |
| ZeroDrift | 1.37 | 3.25 | 0.81 | 1.73 | 16/17 |
| VIO-Edge | 1.80 | 4.54 | 0.74 | 2.23 | 16/17 |
| Vortex | 2.16 | 5.00 | 2.68 | 3.18 | 7/17 |
| TrustNav (as shipped) | 2.02 | 5.11 | 1.57 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.13 | 1.53 | 2.75 | 6/17 |
| Ridhima Chopra | 157.14 | 285.70 | 294.90 | 243.58 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.55 | 4.49 | 0.73 | 2.12 | 17/17 |
| SRMIST_vortexxx | 1.25 | 3.97 | 0.38 | 1.74 | 16/17 |
| ZeroDrift | 1.18 | 3.11 | 0.73 | 1.59 | 17/17 |
| VIO-Edge | 1.70 | 4.16 | 0.72 | 2.08 | 16/17 |
| Vortex | 1.78 | 4.46 | 1.81 | 2.58 | 10/17 |
| TrustNav (as shipped) | 1.73 | 4.69 | 1.22 | 2.42 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.73 | 4.72 | 1.23 | 2.43 | 13/17 |
| Ridhima Chopra | 117.65 | 213.73 | 219.34 | 181.80 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 2.08 | 4.34 | 0.59 | 2.22 | 14/17 |
| SRMIST_vortexxx | 2.31 | 1.86 | 0.62 | 1.58 | 11/17 |
| ZeroDrift | 2.52 | 2.37 | 1.50 | 2.12 | 6/17 |
| VIO-Edge | 2.23 | 3.75 | 0.61 | 2.10 | 9/17 |
| Vortex | 4.55 | 7.26 | 6.57 | 6.06 | 2/17 |
| TrustNav (as shipped) | 3.58 | 6.06 | 2.01 | 3.75 | 0/17 |
| TrustNav `--no-fill --blind 0` | 3.25 | 5.89 | 1.53 | 3.42 | 1/17 |
| Ridhima Chopra | 346.39 | 646.48 | 650.56 | 542.01 | 0/17 |
| standing still | 2.14 | 4.44 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.61 | 4.49 | 0.71 | 2.14 | 17/17 |
| SRMIST_vortexxx | 1.32 | 3.74 | 0.42 | 1.71 | 16/17 |
| ZeroDrift | 1.22 | 2.92 | 0.73 | 1.55 | 17/17 |
| VIO-Edge | 1.67 | 4.08 | 0.68 | 2.03 | 17/17 |
| Vortex | 1.83 | 4.64 | 2.16 | 2.77 | 7/17 |
| TrustNav (as shipped) | 1.79 | 4.70 | 1.26 | 2.46 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.78 | 4.68 | 1.23 | 2.44 | 11/17 |
| Ridhima Chopra | 126.28 | 229.34 | 235.80 | 195.25 | 0/17 |
| standing still | 1.82 | 4.79 | 1.10 | 2.44 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.50 | 0.74 | 0.51 | 0.57 | 17/17 |
| SRMIST_vortexxx | 0.31 | 0.47 | 0.30 | 0.36 | 17/17 |
| ZeroDrift | 0.26 | 0.33 | 0.45 | 0.35 | 17/17 |
| VIO-Edge | 0.53 | 0.68 | 0.49 | 0.56 | 15/17 |
| Vortex | 0.49 | 0.74 | 0.72 | 0.65 | 12/17 |
| TrustNav (as shipped) | 0.72 | 0.92 | 0.88 | 0.83 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.88 | 0.83 | 0/17 |
| Ridhima Chopra | 5.90 | 7.38 | 8.88 | 7.39 | 0/17 |
| standing still | 0.67 | 0.87 | 0.80 | 0.77 | – |

### seed 7, accel-first

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.71 | 4.88 | 0.75 | 2.30 | 17/17 |
| SRMIST_vortexxx | 1.43 | 4.02 | 0.49 | 1.86 | 16/17 |
| ZeroDrift | 1.39 | 3.17 | 0.79 | 1.70 | 17/17 |
| VIO-Edge | 1.78 | 4.56 | 0.71 | 2.22 | 17/17 |
| Vortex | 1.86 | 4.85 | 2.20 | 2.86 | 8/17 |
| TrustNav (as shipped) | 2.00 | 5.12 | 1.57 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 2.00 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.61 | 279.18 | 295.28 | 241.60 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.55 | 4.46 | 0.72 | 2.11 | 17/17 |
| SRMIST_vortexxx | 1.25 | 3.97 | 0.41 | 1.75 | 15/17 |
| ZeroDrift | 1.15 | 3.01 | 0.74 | 1.55 | 17/17 |
| VIO-Edge | 1.67 | 4.17 | 0.67 | 2.05 | 16/17 |
| Vortex | 1.66 | 4.46 | 1.57 | 2.45 | 8/17 |
| TrustNav (as shipped) | 1.71 | 4.72 | 1.22 | 2.42 | 12/17 |
| TrustNav `--no-fill --blind 0` | 1.73 | 4.77 | 1.28 | 2.47 | 12/17 |
| Ridhima Chopra | 117.72 | 210.20 | 218.46 | 180.48 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 2.07 | 4.36 | 0.62 | 2.23 | 12/17 |
| SRMIST_vortexxx | 2.19 | 1.99 | 0.64 | 1.58 | 9/17 |
| ZeroDrift | 2.54 | 3.33 | 1.41 | 2.37 | 5/17 |
| VIO-Edge | 2.07 | 4.10 | 0.63 | 2.16 | 11/17 |
| Vortex | 3.31 | 5.19 | 5.23 | 4.54 | 3/17 |
| TrustNav (as shipped) | 3.47 | 6.27 | 1.89 | 3.73 | 1/17 |
| TrustNav `--no-fill --blind 0` | 3.32 | 5.78 | 1.70 | 3.47 | 0/17 |
| Ridhima Chopra | 344.02 | 632.75 | 657.52 | 539.59 | 0/17 |
| standing still | 2.13 | 4.45 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.61 | 4.46 | 0.71 | 2.13 | 17/17 |
| SRMIST_vortexxx | 1.31 | 3.68 | 0.42 | 1.69 | 16/17 |
| ZeroDrift | 1.21 | 2.79 | 0.74 | 1.51 | 17/17 |
| VIO-Edge | 1.66 | 4.10 | 0.65 | 2.02 | 17/17 |
| Vortex | 1.65 | 4.52 | 1.76 | 2.53 | 9/17 |
| TrustNav (as shipped) | 1.78 | 4.70 | 1.27 | 2.46 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.79 | 4.71 | 1.26 | 2.46 | 10/17 |
| Ridhima Chopra | 126.61 | 224.16 | 235.64 | 193.78 | 0/17 |
| standing still | 1.82 | 4.80 | 1.10 | 2.44 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.50 | 0.71 | 0.51 | 0.56 | 17/17 |
| SRMIST_vortexxx | 0.34 | 0.47 | 0.29 | 0.36 | 17/17 |
| ZeroDrift | 0.26 | 0.33 | 0.44 | 0.34 | 17/17 |
| VIO-Edge | 0.53 | 0.68 | 0.45 | 0.55 | 16/17 |
| Vortex | 0.49 | 0.73 | 0.63 | 0.61 | 12/17 |
| TrustNav (as shipped) | 0.71 | 0.92 | 0.89 | 0.84 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.88 | 0.83 | 0/17 |
| Ridhima Chopra | 5.81 | 7.10 | 9.03 | 7.33 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.78 | – |

### Per recording, ATE (m), seed 42 accel-first (best non-baseline in bold)

| recording | Path Finder (ours) | SRMIST_vortexxx | ZeroDrift | VIO-Edge | Vortex | TrustNav (as shipped) | TrustNav `--no-fill --blind 0` | Ridhima Chopra | standing still |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01 | 4.21 | 3.88 | **3.36** | 4.17 | 4.10 | 4.24 | 4.18 | 778.20 | 4.32 |
| MH_02 | 4.51 | 4.31 | **3.80** | 4.38 | 4.34 | 4.32 | 4.47 | 417.65 | 4.67 |
| MH_03 | 3.09 | 2.41 | **1.75** | 2.98 | 2.87 | 3.57 | 3.56 | 94.84 | 3.59 |
| MH_04 | 6.35 | 4.53 | **3.23** | 5.67 | 6.26 | 6.74 | 6.73 | 61.74 | 6.81 |
| MH_05 | 6.45 | 4.72 | **3.27** | 5.65 | 6.57 | 6.82 | 6.82 | 78.03 | 6.87 |
| V1_01 | 1.78 | 1.31 | **1.23** | 1.82 | 2.01 | 1.90 | 1.87 | 128.99 | 1.85 |
| V1_02 | 1.28 | 0.84 | **0.83** | 1.38 | 1.34 | 1.91 | 1.88 | 152.73 | 1.78 |
| V1_03 | 1.32 | 1.62 | **0.92** | 1.40 | 1.50 | 1.54 | 1.52 | 53.56 | 1.58 |
| V2_01 | 2.11 | **2.00** | 2.03 | 2.24 | 2.22 | 2.34 | 2.34 | 226.02 | 2.29 |
| V2_02 | 1.78 | **1.37** | 1.41 | 2.04 | 2.31 | 2.16 | 2.20 | 180.98 | 2.12 |
| V2_03 | 1.98 | **1.30** | 1.45 | 1.87 | 1.97 | 2.19 | 2.15 | 197.53 | 2.06 |
| room1 | 0.57 | **0.43** | 0.49 | 0.75 | 2.27 | 1.48 | 1.52 | 147.49 | 1.01 |
| room2 | 0.66 | 0.50 | 0.87 | **0.49** | 1.40 | 1.60 | 1.61 | 84.13 | 1.24 |
| room3 | 0.72 | **0.58** | 1.19 | 0.95 | 2.08 | 1.73 | 1.72 | 435.13 | 1.28 |
| room4 | 0.75 | **0.69** | 0.94 | 0.83 | 1.87 | 1.51 | 1.54 | 154.36 | 1.09 |
| room5 | 0.91 | **0.46** | 0.57 | 0.65 | 2.41 | 1.43 | 1.40 | 767.34 | 0.93 |
| room6 | 0.87 | **0.36** | 0.71 | 0.61 | 2.43 | 1.59 | 1.50 | 163.44 | 1.00 |

### Per recording, blind 5 s (m), seed 42 accel-first

| recording | Path Finder (ours) | SRMIST_vortexxx | ZeroDrift | VIO-Edge | Vortex | TrustNav (as shipped) | TrustNav `--no-fill --blind 0` | Ridhima Chopra | standing still |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01 | 5.11 | **3.20** | 4.50 | 4.59 | 5.76 | 7.48 | 5.90 | 1722.12 | 5.18 |
| MH_02 | 5.56 | 3.04 | **1.06** | 5.33 | 3.26 | 6.83 | 6.86 | 941.34 | 5.69 |
| MH_03 | 3.12 | **0.88** | 2.99 | 2.19 | 2.44 | 4.94 | 4.59 | 242.62 | 3.18 |
| MH_04 | 3.89 | **0.47** | 1.37 | 3.61 | 4.85 | 5.86 | 5.49 | 152.65 | 4.01 |
| MH_05 | 4.13 | **0.68** | 1.28 | 3.60 | 4.38 | 6.20 | 5.70 | 187.99 | 4.23 |
| V1_01 | 1.81 | 1.54 | **1.48** | 2.08 | 3.20 | 3.38 | 3.21 | 300.87 | 1.77 |
| V1_02 | 0.94 | 0.95 | 0.91 | **0.66** | 1.75 | 2.94 | 2.65 | 326.26 | 1.08 |
| V1_03 | 1.22 | 2.03 | 0.94 | 1.56 | 2.21 | 1.44 | **0.65** | 104.89 | 1.32 |
| V2_01 | 3.22 | 3.50 | **2.31** | 3.18 | 2.90 | 5.19 | 4.58 | 523.03 | 3.22 |
| V2_02 | 2.73 | 2.43 | 3.06 | 2.87 | 2.75 | **1.86** | 4.01 | 405.00 | 2.79 |
| V2_03 | 2.69 | 2.70 | 3.49 | **2.61** | 4.42 | 4.74 | 3.93 | 449.04 | 2.63 |
| room1 | **0.52** | 0.55 | 1.10 | 0.90 | 2.27 | 2.30 | 1.93 | 265.13 | 0.54 |
| room2 | 0.70 | 0.87 | 0.87 | **0.49** | 1.88 | 1.68 | 1.60 | 234.41 | 0.67 |
| room3 | 0.64 | **0.63** | 3.43 | 1.44 | 5.91 | 1.69 | 1.28 | 926.27 | 0.68 |
| room4 | 0.72 | 0.73 | 0.67 | **0.67** | 4.32 | 1.64 | 2.14 | 319.60 | 0.76 |
| room5 | 0.62 | 0.73 | 1.67 | **0.55** | 6.00 | 1.77 | 1.40 | 1762.08 | 0.66 |
| room6 | 0.40 | 0.35 | 0.52 | **0.12** | 2.71 | 1.82 | 1.94 | 398.61 | 0.36 |

### Contract

| entry | runs | one pose per row, same timestamps | crashes | rows out vs in (typical) |
|---|---:|---:|---:|---|
| **Path Finder (ours)** | 68 | 68/68 | 0 | +0 (range +0 to +0) |
| SRMIST_vortexxx | 68 | 68/68 | 0 | +0 (range +0 to +0) |
| ZeroDrift | 68 | 68/68 | 0 | +0 (range +0 to +0) |
| VIO-Edge | 68 | 68/68 | 0 | +0 (range +0 to +0) |
| Vortex | 68 | 17/68 | 0 | +1 (range +0 to +1) |
| TrustNav (as shipped) | 68 | 0/68 | 0 | +1728 (range +1410 to +2191) |
| TrustNav `--no-fill --blind 0` | 68 | 0/68 | 0 | -28 (range -49 to -11) |
| Ridhima Chopra | 68 | 68/68 | 0 | +0 (range +0 to +0) |

Vortex writes a pose for the CSV header line (timestamp 0). TrustNav as shipped fills gaps at 200 Hz and appends 5 s; with `--no-fill --blind 0` it collapses duplicate rows instead, so it is short by the duplicate count.

### Resources (all 68 files; heap on MH_04, V1_03, MH_05, V2_03, room6, seed 42)

| entry | CPU per s of IMU, median (max), ms | process peak RSS, median (max), MB | heap peak, warm protocol, max of 5, MiB | heap peak, cold protocol, max of 5, MiB |
|---|---:|---:|---:|---:|
| **Path Finder (ours)** | 7.5 (8.6) | 45 (45) | 1.06 | 3.25 |
| SRMIST_vortexxx | 2.1 (3.0) | 54 (56) | 2.53 | 12.94 |
| ZeroDrift | 2.5 (3.4) | 43 (44) | 2.19 | 5.20 |
| VIO-Edge | 5.7 (6.6) | 47 (48) | 2.33 | 4.58 |
| Vortex | 4.1 (5.1) | 25 (26) | 0.73 | 5.12 |
| TrustNav (as shipped) | 2.4 (2.7) | 21 (21) | – | – |
| TrustNav `--no-fill --blind 0` | 2.3 (2.8) | 21 (21) | 0.78 | 1.93 |
| Ridhima Chopra | 1.0 (1.2) | 21 (21) | 0.08 | 2.48 |

RSS includes the interpreter and numpy, so it is not the organisers' measure. vortexxx's cold-protocol peak (12.9 MiB) comes from its first execution; it measures itself with the warm protocol (2.5 MiB).

### Where each rival beats us, every set

* **ZeroDrift, seed 42, accel-first (documented order)**: Vicon rooms ATE 1.71 vs 1.31; Vicon rooms checkpoints 1.55 vs 1.14; Vicon rooms blind 5 s 2.10 vs 2.03; Vicon rooms dropouts 1.59 vs 1.20; Vicon rooms drift 1 s 0.50 vs 0.25; Machine Hall ATE 4.92 vs 3.08; Machine Hall checkpoints 4.51 vs 2.92; Machine Hall blind 5 s 4.36 vs 2.24; Machine Hall dropouts 4.50 vs 2.73; Machine Hall drift 1 s 0.73 vs 0.33; TUM-VI drift 1 s 0.51 vs 0.43
* **ZeroDrift, seed 42, EuRoC file as is (gyro first)**: Vicon rooms ATE 1.71 vs 1.40; Vicon rooms checkpoints 1.56 vs 1.18; Vicon rooms dropouts 1.59 vs 1.25; Vicon rooms drift 1 s 0.50 vs 0.32; Machine Hall ATE 4.91 vs 3.38; Machine Hall checkpoints 4.47 vs 3.07; Machine Hall blind 5 s 4.36 vs 3.80; Machine Hall dropouts 4.49 vs 3.02; Machine Hall drift 1 s 0.75 vs 0.49; TUM-VI drift 1 s 0.51 vs 0.43
* **ZeroDrift, seed 1, accel-first**: Vicon rooms ATE 1.71 vs 1.37; Vicon rooms checkpoints 1.55 vs 1.18; Vicon rooms dropouts 1.61 vs 1.22; Vicon rooms drift 1 s 0.50 vs 0.26; Machine Hall ATE 4.92 vs 3.25; Machine Hall checkpoints 4.49 vs 3.11; Machine Hall blind 5 s 4.34 vs 2.37; Machine Hall dropouts 4.49 vs 2.92; Machine Hall drift 1 s 0.74 vs 0.33; TUM-VI drift 1 s 0.51 vs 0.45
* **ZeroDrift, seed 7, accel-first**: Vicon rooms ATE 1.71 vs 1.39; Vicon rooms checkpoints 1.55 vs 1.15; Vicon rooms dropouts 1.61 vs 1.21; Vicon rooms drift 1 s 0.50 vs 0.26; Machine Hall ATE 4.88 vs 3.17; Machine Hall checkpoints 4.46 vs 3.01; Machine Hall blind 5 s 4.36 vs 3.33; Machine Hall dropouts 4.46 vs 2.79; Machine Hall drift 1 s 0.71 vs 0.33; TUM-VI drift 1 s 0.51 vs 0.44
* **SRMIST_vortexxx, seed 42, accel-first (documented order)**: Vicon rooms ATE 1.71 vs 1.41; Vicon rooms checkpoints 1.55 vs 1.22; Vicon rooms dropouts 1.59 vs 1.31; Vicon rooms drift 1 s 0.50 vs 0.30; Machine Hall ATE 4.92 vs 3.97; Machine Hall checkpoints 4.51 vs 3.90; Machine Hall blind 5 s 4.36 vs 1.65; Machine Hall dropouts 4.50 vs 3.63; Machine Hall drift 1 s 0.73 vs 0.47; TUM-VI ATE 0.75 vs 0.50; TUM-VI checkpoints 0.72 vs 0.38; TUM-VI dropouts 0.71 vs 0.41; TUM-VI drift 1 s 0.51 vs 0.30
* **SRMIST_vortexxx, seed 42, EuRoC file as is (gyro first)**: Vicon rooms ATE 1.71 vs 1.42; Vicon rooms checkpoints 1.56 vs 1.25; Vicon rooms dropouts 1.59 vs 1.34; Vicon rooms drift 1 s 0.50 vs 0.35; Machine Hall ATE 4.91 vs 4.20; Machine Hall checkpoints 4.47 vs 3.91; Machine Hall blind 5 s 4.36 vs 3.00; Machine Hall dropouts 4.49 vs 3.86; Machine Hall drift 1 s 0.75 vs 0.59; TUM-VI ATE 0.75 vs 0.50; TUM-VI checkpoints 0.73 vs 0.39; TUM-VI dropouts 0.71 vs 0.42; TUM-VI drift 1 s 0.51 vs 0.30
* **SRMIST_vortexxx, seed 1, accel-first**: Vicon rooms ATE 1.71 vs 1.44; Vicon rooms checkpoints 1.55 vs 1.25; Vicon rooms dropouts 1.61 vs 1.32; Vicon rooms drift 1 s 0.50 vs 0.31; Machine Hall ATE 4.92 vs 4.06; Machine Hall checkpoints 4.49 vs 3.97; Machine Hall blind 5 s 4.34 vs 1.86; Machine Hall dropouts 4.49 vs 3.74; Machine Hall drift 1 s 0.74 vs 0.47; TUM-VI ATE 0.75 vs 0.51; TUM-VI checkpoints 0.73 vs 0.38; TUM-VI dropouts 0.71 vs 0.42; TUM-VI drift 1 s 0.51 vs 0.30
* **SRMIST_vortexxx, seed 7, accel-first**: Vicon rooms ATE 1.71 vs 1.43; Vicon rooms checkpoints 1.55 vs 1.25; Vicon rooms dropouts 1.61 vs 1.31; Vicon rooms drift 1 s 0.50 vs 0.34; Machine Hall ATE 4.88 vs 4.02; Machine Hall checkpoints 4.46 vs 3.97; Machine Hall blind 5 s 4.36 vs 1.99; Machine Hall dropouts 4.46 vs 3.68; Machine Hall drift 1 s 0.71 vs 0.47; TUM-VI ATE 0.75 vs 0.49; TUM-VI checkpoints 0.72 vs 0.41; TUM-VI dropouts 0.71 vs 0.42; TUM-VI drift 1 s 0.51 vs 0.29
* **VIO-Edge, seed 42, accel-first (documented order)**: Machine Hall ATE 4.92 vs 4.57; Machine Hall checkpoints 4.51 vs 4.18; Machine Hall blind 5 s 4.36 vs 3.87; Machine Hall dropouts 4.50 vs 4.09; TUM-VI checkpoints 0.72 vs 0.67; TUM-VI dropouts 0.71 vs 0.65; TUM-VI drift 1 s 0.51 vs 0.45
* **VIO-Edge, seed 42, EuRoC file as is (gyro first)**: Machine Hall ATE 4.91 vs 4.75; Machine Hall checkpoints 4.47 vs 4.36; Machine Hall dropouts 4.49 vs 4.36; TUM-VI checkpoints 0.73 vs 0.67; TUM-VI dropouts 0.71 vs 0.64; TUM-VI drift 1 s 0.51 vs 0.44
* **VIO-Edge, seed 1, accel-first**: Machine Hall ATE 4.92 vs 4.54; Machine Hall checkpoints 4.49 vs 4.16; Machine Hall blind 5 s 4.34 vs 3.75; Machine Hall dropouts 4.49 vs 4.08; Machine Hall drift 1 s 0.74 vs 0.68
* **VIO-Edge, seed 7, accel-first**: Machine Hall ATE 4.88 vs 4.56; Machine Hall checkpoints 4.46 vs 4.17; Machine Hall blind 5 s 4.36 vs 4.10; Machine Hall dropouts 4.46 vs 4.10; TUM-VI checkpoints 0.72 vs 0.67; TUM-VI dropouts 0.71 vs 0.65; TUM-VI drift 1 s 0.51 vs 0.45

## Reproduce

```bash
git clone --depth 1 https://github.com/<owner>/<repo>.git ../competitors/<owner>_<repo>    # each repo above
python -m venv ../competitors/.venv && ../competitors/.venv/bin/pip install numpy pandas
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python ../competitors/.venv/bin/python
python tools/competitor_scoreboard.py run  --competitors ../competitors --cpy ../competitors/.venv/bin/python --tag all
python tools/competitor_scoreboard.py heap --competitors ../competitors --cpy ../competitors/.venv/bin/python --protocol warm
python tools/competitor_scoreboard.py heap --competitors ../competitors --cpy ../competitors/.venv/bin/python --protocol cold
cp runs/scoreboard/results_all.json results/track3/competitors_raw.json   # (errors trimmed to 300 chars)
python tools/competitor_tables.py results/track3/competitors_raw.json \
    --heap results/track3/competitors_heap_warm.json --heap-cold results/track3/competitors_heap_cold.json
```
