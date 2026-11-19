# Competitors: head-to-head on the same degraded files

Every public Track 3 entry with a runnable Python `main.py`, run on **exactly the same inputs** as ours and scored with
**our** `tools/score_track3.py` (same SE(3) alignment, same checkpoint, blind, dropout and drift windows). Tables are
generated from `competitors_raw.json` by `tools/competitor_tables.py`; only the prose is written by hand.

**Ours here is rotor drag (`85b3a30`, PR #44) + dropout bridging (PR #45)**, the code that is meant to ship. VIO-Edge and
Vortex were re-run at their 19 Nov commits; the other entries had not changed since 19 Nov morning.

## Setup

* **Inputs** (`tools/competitor_scoreboard.py inputs`): the 11 EuRoC + 6 TUM-VI recordings, damaged by the organisers'
  `degrade_imu.py` (sha256 `cc7398db…2f31`, the copy in the VIO-Edge repo, identical to the emailed one). Four sets:
  seed 42 with accel-first columns (the script's documented order), seed 42 with the EuRoC file as is (gyro first),
  and seeds 1 and 7 accel-first (only `SEED = ` changed). 68 files; every entry reads the same 68.
* **Runs** (`... run`): one process per file under `/usr/bin/time -l`, BLAS/OpenMP threads pinned to 1, Apple M4,
  Python 3.11, numpy 2.4.6. Competitor code ran from its own clone in a separate virtualenv (numpy + pandas only),
  after reading it (no network access, no file deletion outside its own temp files). Nothing of theirs is in this repo.
  CPU here is wall-clock process CPU on a busy machine; our official budget is `python_budget.json`.
* **Standing still**: the start pose on every row, written by the harness.
* **Contract**: one pose per input row, timestamps equal to the input's, 8 finite columns, unit quaternions.
* **Heap** (`... heap`): tracemalloc around a full execution of each script. *Warm*: the script runs once untraced
  first, so lazily imported library modules are not counted (how vortexxx measures itself); *cold*: the first
  execution is traced. Which one the organisers use is not known.

| Team | Repo | Commit run | Licence |
|---|---|---|---|
| SRMIST_vortexxx | [`RaktimChandra/pathfinder`](https://github.com/RaktimChandra/pathfinder) | `a67f4ee` | none found |
| ZeroDrift | [`Gem2005/VITChennai_ZeroDrift`](https://github.com/Gem2005/VITChennai_ZeroDrift) | `9c03e6d` | none found |
| VIO-Edge | [`rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation`](https://github.com/rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation) | `87f5c4a (new look-ahead weights, 19 Nov)` | MIT |
| Vortex | [`ArnavJain2005/Vortex`](https://github.com/ArnavJain2005/Vortex) | `5ba1c4b (19 Nov)` | MIT |
| TrustNav | [`Am-berneal/TrustNav-GPS-Denied-Navigation`](https://github.com/Am-berneal/TrustNav-GPS-Denied-Navigation) | `bec7863` | none found |
| Ridhima Chopra | [`ridhima936/PathFinder-IMU-Trajectory`](https://github.com/ridhima936/PathFinder-IMU-Trajectory) | `b90f29c` | none found |
| VITC_SingleThread | [`jeeva-m-21/samsung-track3-pathfinder`](https://github.com/jeeva-m-21/samsung-track3-pathfinder) | `19a0651` | none found |

VITC_SingleThread was not run (C++ only, which the FAQ excludes). Ridhima Chopra's entry is plain dead reckoning.
New Track 3 repositories searched on 19 Nov ("Path Finder trajectory", "Samsung XR Hackathon", "degrade_imu", "PRISM XR
Hackathon"): none. `Prabh-84/Samsung-XR-Hackathon` is empty; `rishita2201/Samsung_XR_Hackathon` is a Track 4 entry.

## Headline (seed 42, accel-first: the organisers' documented setting)

Points come from checkpoints (30), blind 5 s (20) and dropouts (10), so all three are shown, with ATE.

**Checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.35 | 3.85 | 0.66 | 1.84 | 17/17 |
| SRMIST_vortexxx | 1.22 | 3.90 | 0.38 | 1.71 | 16/17 |
| ZeroDrift | 1.14 | 2.92 | 0.73 | 1.52 | 17/17 |
| VIO-Edge | 1.43 | 3.80 | 0.55 | 1.82 | 17/17 |
| Vortex | 1.71 | 4.40 | 0.88 | 2.21 | 14/17 |
| TrustNav (as shipped) | 1.71 | 4.72 | 1.18 | 2.41 | 13/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.76 | 1.23 | 2.45 | 11/17 |
| Ridhima Chopra | 116.84 | 215.16 | 216.39 | 180.89 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**Blind 5 s**

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.74 | 3.13 | 0.61 | 1.75 | 13/17 |
| SRMIST_vortexxx | 2.19 | 1.65 | 0.64 | 1.49 | 11/17 |
| ZeroDrift | 2.03 | 2.24 | 1.38 | 1.86 | 10/17 |
| VIO-Edge | 2.08 | 3.94 | 0.66 | 2.13 | 13/17 |
| Vortex | 2.87 | 4.14 | 1.27 | 2.68 | 5/17 |
| TrustNav (as shipped) | 3.26 | 6.26 | 1.82 | 3.63 | 1/17 |
| TrustNav `--no-fill --blind 0` | 3.17 | 5.71 | 1.71 | 3.40 | 1/17 |
| Ridhima Chopra | 351.52 | 649.34 | 651.02 | 544.82 | 0/17 |
| standing still | 2.14 | 4.46 | 0.61 | 2.28 | – |

**Dropouts**

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.36 | 3.71 | 0.64 | 1.80 | 17/17 |
| SRMIST_vortexxx | 1.31 | 3.63 | 0.41 | 1.68 | 16/17 |
| ZeroDrift | 1.20 | 2.73 | 0.73 | 1.48 | 17/17 |
| VIO-Edge | 1.47 | 3.71 | 0.57 | 1.81 | 17/17 |
| Vortex | 1.67 | 4.48 | 0.88 | 2.22 | 16/17 |
| TrustNav (as shipped) | 1.75 | 4.70 | 1.21 | 2.43 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.69 | 1.25 | 2.44 | 9/17 |
| Ridhima Chopra | 126.05 | 230.22 | 234.82 | 195.08 | 0/17 |
| standing still | 1.78 | 4.80 | 1.08 | 2.42 | – |

**ATE**

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.48 | 4.07 | 0.69 | 1.96 | 17/17 |
| SRMIST_vortexxx | 1.41 | 3.97 | 0.50 | 1.84 | 16/17 |
| ZeroDrift | 1.31 | 3.08 | 0.80 | 1.65 | 17/17 |
| VIO-Edge | 1.59 | 4.08 | 0.61 | 1.98 | 17/17 |
| Vortex | 1.89 | 4.83 | 0.91 | 2.41 | 15/17 |
| TrustNav (as shipped) | 2.01 | 5.14 | 1.56 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.63 | 286.09 | 291.98 | 242.48 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

* **Blind 5 s: we are second** (1.75 m; vortexxx 1.49 m, ZeroDrift 1.86, VIO-Edge 2.13) and **best in the Vicon rooms**
  (1.74 m; next ZeroDrift 2.03). vortexxx wins the blind window on the Machine Hall (1.65 vs our 3.13 m).
* **Checkpoints: third** (1.84 m; ZeroDrift 1.52, vortexxx 1.71, VIO-Edge 1.82). **Dropouts: third, level with
  VIO-Edge** (1.80 vs 1.81 m; ZeroDrift 1.48, vortexxx 1.68). **ATE: third** (1.96 m; ZeroDrift 1.65, vortexxx 1.84,
  VIO-Edge 1.98). The order holds on all four input sets.
* We, ZeroDrift and VIO-Edge beat standing still on all 17 for ATE; vortexxx loses V1_03. Vortex is much improved
  since 19 Nov morning (ATE 2.82 → 2.41 m) but still worse than standing still on 2 recordings.
* Resources: every eligible entry is under 8 MB on the warm protocol (ours 1.06 MiB). On the cold protocol vortexxx
  (12.9 MiB) and VIO-Edge's new build (14.4 MiB) exceed 8 MB; ours is 2.9 MiB.

## Where we lose (decides what to work on)

Group means, metres, ours vs the rival, every metric where the rival is at least 0.05 m better (seed 42, accel-first):

* **ZeroDrift**: Vicon rooms ATE 1.48 vs 1.31; Vicon rooms checkpoints 1.35 vs 1.14; Vicon rooms dropouts 1.36 vs 1.20; Vicon rooms drift 1 s 0.44 vs 0.25; Machine Hall ATE 4.07 vs 3.08; Machine Hall checkpoints 3.85 vs 2.92; Machine Hall blind 5 s 3.13 vs 2.24; Machine Hall dropouts 3.71 vs 2.73; Machine Hall drift 1 s 0.54 vs 0.33.
* **SRMIST_vortexxx**: Vicon rooms ATE 1.48 vs 1.41; Vicon rooms checkpoints 1.35 vs 1.22; Vicon rooms drift 1 s 0.44 vs 0.30; Machine Hall ATE 4.07 vs 3.97; Machine Hall blind 5 s 3.13 vs 1.65; Machine Hall dropouts 3.71 vs 3.63; Machine Hall drift 1 s 0.54 vs 0.47; TUM-VI ATE 0.69 vs 0.50; TUM-VI checkpoints 0.66 vs 0.38; TUM-VI dropouts 0.64 vs 0.41; TUM-VI drift 1 s 0.44 vs 0.30.
* **VIO-Edge**: TUM-VI ATE 0.69 vs 0.61; TUM-VI checkpoints 0.66 vs 0.55; TUM-VI dropouts 0.64 vs 0.57; TUM-VI drift 1 s 0.44 vs 0.37.

In short: (1) **Machine Hall**, every metric, against ZeroDrift (ATE 3.08 vs our 4.07 m) and on blind against
vortexxx (1.65 vs 3.13 m); (2) **TUM-VI** against vortexxx (checkpoints 0.38 vs 0.66 m, ATE 0.50 vs 0.69 m) and now
VIO-Edge (0.55 / 0.61); (3) **Vicon checkpoints and dropouts** against ZeroDrift (1.14 / 1.20 vs 1.35 / 1.36 m);
(4) **drift over 1 s** against both leaders.

## Why they win (from reading their code and reports)

Both leaders are **non-causal**: they use the whole file, which the FAQ allows. We run a causal filter.

* **vortexxx**: attitude by a complementary filter, then translation by a **Kalman filter + RTS smoother at 2.5 Hz**
  with a **workspace-centre state** (soft prior σ = 3.5 m on EuRoC), rotor drag with a Cauchy-weighted innovation,
  zero-velocity updates at every rest of ≥ 1.5 s, process noise that follows the organisers' severity ramp plus the
  measured excess gyro noise, and hard-coded EuRoC platform constants (gravity direction, gyro bias, drag). Blind:
  hold position (τ = 0.2 s). On the Machine Hall its trajectory keeps its real extent, so the landing point (within
  1 m of take-off) aligns correctly; ours is pulled toward the start and the landing aligns to the path's centre.
* **ZeroDrift**: undoes the reversible damage first, RK4 sub-steps across missing rows, a two-pass error-state EKF
  with an accelerometer-scale state, then **zero-phase drift removal** with a cut-off that shortens as severity grows,
  plus rotor drag as displacement information in flight.
* **VIO-Edge**: ESKF + learned velocity like us; its 19 Nov weights moved it level with us overall and ahead on TUM-VI.

What we tried from these (each with its own results page): dropout bridging (kept, `gap_bridge_rotor.md`); a
zero-phase pull (`zero_phase_pull.md`) and an RTS smoother with a workspace state (`rts_smoother.md`), both better
on average but worse than standing still on held-out room5/room6; spike repair (no effect, `spike_repair.md`);
noise-adaptive rest detection (worse, `noise_adaptive_zupt.md`); a two-pass start (better on average, fails one
recording at one seed, `two_pass.md`); averaging the blackout velocity (no gain, `blackout_velocity.md`).

## All tables

### seed 42, accel-first (documented order)

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.48 | 4.07 | 0.69 | 1.96 | 17/17 |
| SRMIST_vortexxx | 1.41 | 3.97 | 0.50 | 1.84 | 16/17 |
| ZeroDrift | 1.31 | 3.08 | 0.80 | 1.65 | 17/17 |
| VIO-Edge | 1.59 | 4.08 | 0.61 | 1.98 | 17/17 |
| Vortex | 1.89 | 4.83 | 0.91 | 2.41 | 15/17 |
| TrustNav (as shipped) | 2.01 | 5.14 | 1.56 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.63 | 286.09 | 291.98 | 242.48 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.35 | 3.85 | 0.66 | 1.84 | 17/17 |
| SRMIST_vortexxx | 1.22 | 3.90 | 0.38 | 1.71 | 16/17 |
| ZeroDrift | 1.14 | 2.92 | 0.73 | 1.52 | 17/17 |
| VIO-Edge | 1.43 | 3.80 | 0.55 | 1.82 | 17/17 |
| Vortex | 1.71 | 4.40 | 0.88 | 2.21 | 14/17 |
| TrustNav (as shipped) | 1.71 | 4.72 | 1.18 | 2.41 | 13/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.76 | 1.23 | 2.45 | 11/17 |
| Ridhima Chopra | 116.84 | 215.16 | 216.39 | 180.89 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.74 | 3.13 | 0.61 | 1.75 | 13/17 |
| SRMIST_vortexxx | 2.19 | 1.65 | 0.64 | 1.49 | 11/17 |
| ZeroDrift | 2.03 | 2.24 | 1.38 | 1.86 | 10/17 |
| VIO-Edge | 2.08 | 3.94 | 0.66 | 2.13 | 13/17 |
| Vortex | 2.87 | 4.14 | 1.27 | 2.68 | 5/17 |
| TrustNav (as shipped) | 3.26 | 6.26 | 1.82 | 3.63 | 1/17 |
| TrustNav `--no-fill --blind 0` | 3.17 | 5.71 | 1.71 | 3.40 | 1/17 |
| Ridhima Chopra | 351.52 | 649.34 | 651.02 | 544.82 | 0/17 |
| standing still | 2.14 | 4.46 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.36 | 3.71 | 0.64 | 1.80 | 17/17 |
| SRMIST_vortexxx | 1.31 | 3.63 | 0.41 | 1.68 | 16/17 |
| ZeroDrift | 1.20 | 2.73 | 0.73 | 1.48 | 17/17 |
| VIO-Edge | 1.47 | 3.71 | 0.57 | 1.81 | 17/17 |
| Vortex | 1.67 | 4.48 | 0.88 | 2.22 | 16/17 |
| TrustNav (as shipped) | 1.75 | 4.70 | 1.21 | 2.43 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.75 | 4.69 | 1.25 | 2.44 | 9/17 |
| Ridhima Chopra | 126.05 | 230.22 | 234.82 | 195.08 | 0/17 |
| standing still | 1.78 | 4.80 | 1.08 | 2.42 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.44 | 0.54 | 0.44 | 0.47 | 17/17 |
| SRMIST_vortexxx | 0.30 | 0.47 | 0.30 | 0.35 | 17/17 |
| ZeroDrift | 0.25 | 0.33 | 0.43 | 0.34 | 17/17 |
| VIO-Edge | 0.40 | 0.53 | 0.37 | 0.43 | 17/17 |
| Vortex | 0.50 | 0.72 | 0.55 | 0.59 | 13/17 |
| TrustNav (as shipped) | 0.71 | 0.92 | 0.88 | 0.83 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.89 | 0.83 | 0/17 |
| Ridhima Chopra | 5.96 | 7.36 | 8.88 | 7.40 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.77 | – |

### seed 42, EuRoC file as is (gyro first)

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.54 | 4.16 | 0.69 | 2.01 | 17/17 |
| SRMIST_vortexxx | 1.42 | 4.20 | 0.50 | 1.92 | 16/17 |
| ZeroDrift | 1.40 | 3.38 | 0.79 | 1.77 | 17/17 |
| VIO-Edge | 1.63 | 4.21 | 0.61 | 2.02 | 17/17 |
| Vortex | 1.88 | 4.93 | 0.91 | 2.43 | 15/17 |
| TrustNav (as shipped) | 2.01 | 5.18 | 2.53 | 3.13 | 7/17 |
| TrustNav `--no-fill --blind 0` | 3.35 | 5.78 | 2.57 | 3.79 | 2/17 |
| Ridhima Chopra | 848.77 | 16403.11 | 19554.87 | 12025.73 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.45 | 3.91 | 0.66 | 1.90 | 15/17 |
| SRMIST_vortexxx | 1.25 | 3.91 | 0.39 | 1.73 | 17/17 |
| ZeroDrift | 1.18 | 3.07 | 0.73 | 1.58 | 17/17 |
| VIO-Edge | 1.49 | 3.79 | 0.54 | 1.83 | 17/17 |
| Vortex | 1.73 | 4.41 | 0.88 | 2.22 | 14/17 |
| TrustNav (as shipped) | 1.83 | 4.71 | 2.30 | 2.84 | 8/17 |
| TrustNav `--no-fill --blind 0` | 2.92 | 5.03 | 2.31 | 3.33 | 6/17 |
| Ridhima Chopra | 495.26 | 12262.56 | 14024.92 | 8731.40 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.87 | 3.47 | 0.62 | 1.90 | 13/17 |
| SRMIST_vortexxx | 2.13 | 3.00 | 0.60 | 1.85 | 10/17 |
| ZeroDrift | 2.46 | 3.80 | 1.44 | 2.49 | 7/17 |
| VIO-Edge | 2.12 | 4.58 | 0.69 | 2.34 | 10/17 |
| Vortex | 2.58 | 5.25 | 1.21 | 2.88 | 3/17 |
| TrustNav (as shipped) | 2.61 | 5.06 | 3.50 | 3.64 | 4/17 |
| TrustNav `--no-fill --blind 0` | 5.73 | 9.20 | 3.97 | 6.13 | 1/17 |
| Ridhima Chopra | 2580.69 | 37613.66 | 45296.19 | 27960.56 | 0/17 |
| standing still | 2.14 | 4.46 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.43 | 3.87 | 0.65 | 1.87 | 14/17 |
| SRMIST_vortexxx | 1.34 | 3.86 | 0.42 | 1.76 | 16/17 |
| ZeroDrift | 1.25 | 3.02 | 0.73 | 1.59 | 17/17 |
| VIO-Edge | 1.52 | 3.82 | 0.56 | 1.86 | 17/17 |
| Vortex | 1.69 | 4.52 | 0.88 | 2.24 | 16/17 |
| TrustNav (as shipped) | 1.81 | 4.67 | 2.28 | 2.82 | 8/17 |
| TrustNav `--no-fill --blind 0` | 2.95 | 5.34 | 2.29 | 3.42 | 2/17 |
| Ridhima Chopra | 594.62 | 13134.48 | 15233.84 | 9449.60 | 0/17 |
| standing still | 1.78 | 4.80 | 1.08 | 2.42 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.46 | 0.64 | 0.44 | 0.51 | 15/17 |
| SRMIST_vortexxx | 0.35 | 0.59 | 0.30 | 0.40 | 17/17 |
| ZeroDrift | 0.32 | 0.49 | 0.43 | 0.41 | 16/17 |
| VIO-Edge | 0.44 | 0.62 | 0.36 | 0.46 | 15/17 |
| Vortex | 0.51 | 0.78 | 0.55 | 0.61 | 13/17 |
| TrustNav (as shipped) | 0.73 | 1.01 | 1.38 | 1.04 | 2/17 |
| TrustNav `--no-fill --blind 0` | 0.92 | 1.06 | 1.42 | 1.14 | 0/17 |
| Ridhima Chopra | 45.57 | 414.72 | 594.24 | 347.79 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.77 | – |

### seed 1, accel-first

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.49 | 4.01 | 0.68 | 1.95 | 17/17 |
| SRMIST_vortexxx | 1.44 | 4.06 | 0.51 | 1.88 | 16/17 |
| ZeroDrift | 1.37 | 3.25 | 0.81 | 1.73 | 16/17 |
| VIO-Edge | 1.60 | 4.06 | 0.65 | 1.99 | 17/17 |
| Vortex | 2.16 | 5.00 | 0.92 | 2.56 | 13/17 |
| TrustNav (as shipped) | 2.02 | 5.11 | 1.57 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 1.99 | 5.13 | 1.53 | 2.75 | 6/17 |
| Ridhima Chopra | 157.14 | 285.70 | 294.90 | 243.58 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.37 | 3.79 | 0.65 | 1.83 | 17/17 |
| SRMIST_vortexxx | 1.25 | 3.97 | 0.38 | 1.74 | 16/17 |
| ZeroDrift | 1.18 | 3.11 | 0.73 | 1.59 | 17/17 |
| VIO-Edge | 1.44 | 3.77 | 0.60 | 1.83 | 17/17 |
| Vortex | 1.78 | 4.46 | 0.86 | 2.24 | 13/17 |
| TrustNav (as shipped) | 1.73 | 4.69 | 1.22 | 2.42 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.73 | 4.72 | 1.23 | 2.43 | 13/17 |
| Ridhima Chopra | 117.65 | 213.73 | 219.34 | 181.80 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.83 | 3.23 | 0.59 | 1.81 | 13/17 |
| SRMIST_vortexxx | 2.31 | 1.86 | 0.62 | 1.58 | 11/17 |
| ZeroDrift | 2.52 | 2.37 | 1.50 | 2.12 | 6/17 |
| VIO-Edge | 2.15 | 3.76 | 0.54 | 2.06 | 12/17 |
| Vortex | 4.55 | 7.26 | 1.14 | 4.14 | 3/17 |
| TrustNav (as shipped) | 3.58 | 6.06 | 2.01 | 3.75 | 0/17 |
| TrustNav `--no-fill --blind 0` | 3.25 | 5.89 | 1.53 | 3.42 | 1/17 |
| Ridhima Chopra | 346.39 | 646.48 | 650.56 | 542.01 | 0/17 |
| standing still | 2.14 | 4.44 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.39 | 3.64 | 0.63 | 1.78 | 17/17 |
| SRMIST_vortexxx | 1.32 | 3.74 | 0.42 | 1.71 | 16/17 |
| ZeroDrift | 1.22 | 2.92 | 0.73 | 1.55 | 17/17 |
| VIO-Edge | 1.49 | 3.68 | 0.61 | 1.82 | 17/17 |
| Vortex | 1.83 | 4.64 | 0.89 | 2.33 | 13/17 |
| TrustNav (as shipped) | 1.79 | 4.70 | 1.26 | 2.46 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.78 | 4.68 | 1.23 | 2.44 | 11/17 |
| Ridhima Chopra | 126.28 | 229.34 | 235.80 | 195.25 | 0/17 |
| standing still | 1.82 | 4.79 | 1.10 | 2.44 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.44 | 0.52 | 0.43 | 0.46 | 17/17 |
| SRMIST_vortexxx | 0.31 | 0.47 | 0.30 | 0.36 | 17/17 |
| ZeroDrift | 0.26 | 0.33 | 0.45 | 0.35 | 17/17 |
| VIO-Edge | 0.40 | 0.52 | 0.41 | 0.44 | 17/17 |
| Vortex | 0.49 | 0.74 | 0.56 | 0.59 | 14/17 |
| TrustNav (as shipped) | 0.72 | 0.92 | 0.88 | 0.83 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.88 | 0.83 | 0/17 |
| Ridhima Chopra | 5.90 | 7.38 | 8.88 | 7.39 | 0/17 |
| standing still | 0.67 | 0.87 | 0.80 | 0.77 | – |

### seed 7, accel-first

**ATE** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.52 | 4.01 | 0.68 | 1.95 | 17/17 |
| SRMIST_vortexxx | 1.43 | 4.02 | 0.49 | 1.86 | 16/17 |
| ZeroDrift | 1.39 | 3.17 | 0.79 | 1.70 | 17/17 |
| VIO-Edge | 1.58 | 4.09 | 0.62 | 1.98 | 17/17 |
| Vortex | 1.86 | 4.85 | 0.88 | 2.40 | 14/17 |
| TrustNav (as shipped) | 2.00 | 5.12 | 1.57 | 2.77 | 6/17 |
| TrustNav `--no-fill --blind 0` | 2.00 | 5.15 | 1.55 | 2.77 | 6/17 |
| Ridhima Chopra | 156.61 | 279.18 | 295.28 | 241.60 | 0/17 |
| standing still | 1.95 | 5.25 | 1.09 | 2.62 | – |

**checkpoints** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.38 | 3.80 | 0.64 | 1.83 | 16/17 |
| SRMIST_vortexxx | 1.25 | 3.97 | 0.41 | 1.75 | 15/17 |
| ZeroDrift | 1.15 | 3.01 | 0.74 | 1.55 | 17/17 |
| VIO-Edge | 1.42 | 3.80 | 0.56 | 1.81 | 17/17 |
| Vortex | 1.66 | 4.46 | 0.82 | 2.18 | 13/17 |
| TrustNav (as shipped) | 1.71 | 4.72 | 1.22 | 2.42 | 12/17 |
| TrustNav `--no-fill --blind 0` | 1.73 | 4.77 | 1.28 | 2.47 | 12/17 |
| Ridhima Chopra | 117.72 | 210.20 | 218.46 | 180.48 | 0/17 |
| standing still | 1.80 | 4.88 | 1.07 | 2.45 | – |

**blind 5 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.84 | 3.35 | 0.61 | 1.85 | 13/17 |
| SRMIST_vortexxx | 2.19 | 1.99 | 0.64 | 1.58 | 9/17 |
| ZeroDrift | 2.54 | 3.33 | 1.41 | 2.37 | 5/17 |
| VIO-Edge | 1.94 | 4.33 | 0.62 | 2.18 | 11/17 |
| Vortex | 3.31 | 5.19 | 1.18 | 3.11 | 3/17 |
| TrustNav (as shipped) | 3.47 | 6.27 | 1.89 | 3.73 | 1/17 |
| TrustNav `--no-fill --blind 0` | 3.32 | 5.78 | 1.70 | 3.47 | 0/17 |
| Ridhima Chopra | 344.02 | 632.75 | 657.52 | 539.59 | 0/17 |
| standing still | 2.13 | 4.45 | 0.61 | 2.28 | – |

**dropouts** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 1.41 | 3.63 | 0.64 | 1.79 | 16/17 |
| SRMIST_vortexxx | 1.31 | 3.68 | 0.42 | 1.69 | 16/17 |
| ZeroDrift | 1.21 | 2.79 | 0.74 | 1.51 | 17/17 |
| VIO-Edge | 1.48 | 3.72 | 0.58 | 1.82 | 17/17 |
| Vortex | 1.65 | 4.52 | 0.86 | 2.22 | 15/17 |
| TrustNav (as shipped) | 1.78 | 4.70 | 1.27 | 2.46 | 10/17 |
| TrustNav `--no-fill --blind 0` | 1.79 | 4.71 | 1.26 | 2.46 | 10/17 |
| Ridhima Chopra | 126.61 | 224.16 | 235.64 | 193.78 | 0/17 |
| standing still | 1.82 | 4.80 | 1.10 | 2.44 | – |

**drift 1 s** (mean metres per group)

| entry | Vicon rooms | Machine Hall | TUM-VI | all 17 | beats standing still |
|---|---:|---:|---:|---:|---:|
| **Path Finder (ours)** | 0.43 | 0.52 | 0.43 | 0.46 | 17/17 |
| SRMIST_vortexxx | 0.34 | 0.47 | 0.29 | 0.36 | 17/17 |
| ZeroDrift | 0.26 | 0.33 | 0.44 | 0.34 | 17/17 |
| VIO-Edge | 0.40 | 0.53 | 0.38 | 0.43 | 17/17 |
| Vortex | 0.49 | 0.73 | 0.54 | 0.58 | 13/17 |
| TrustNav (as shipped) | 0.71 | 0.92 | 0.89 | 0.84 | 0/17 |
| TrustNav `--no-fill --blind 0` | 0.72 | 0.91 | 0.88 | 0.83 | 0/17 |
| Ridhima Chopra | 5.81 | 7.10 | 9.03 | 7.33 | 0/17 |
| standing still | 0.67 | 0.88 | 0.80 | 0.78 | – |

### Per recording, ATE (m), seed 42 accel-first (best non-baseline in bold)

| recording | Path Finder (ours) | SRMIST_vortexxx | ZeroDrift | VIO-Edge | Vortex | TrustNav (as shipped) | TrustNav `--no-fill --blind 0` | Ridhima Chopra | standing still |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01 | 4.11 | 3.88 | **3.36** | 4.26 | 4.10 | 4.24 | 4.18 | 778.20 | 4.32 |
| MH_02 | 4.10 | 4.31 | **3.80** | 4.46 | 4.34 | 4.32 | 4.47 | 417.65 | 4.67 |
| MH_03 | 2.45 | 2.41 | **1.75** | 2.53 | 2.87 | 3.57 | 3.56 | 94.84 | 3.59 |
| MH_04 | 4.65 | 4.53 | **3.23** | 4.59 | 6.26 | 6.74 | 6.73 | 61.74 | 6.81 |
| MH_05 | 5.02 | 4.72 | **3.27** | 4.56 | 6.57 | 6.82 | 6.82 | 78.03 | 6.87 |
| V1_01 | 1.54 | 1.31 | **1.23** | 1.62 | 2.01 | 1.90 | 1.87 | 128.99 | 1.85 |
| V1_02 | **0.80** | 0.84 | 0.83 | 1.06 | 1.34 | 1.91 | 1.88 | 152.73 | 1.78 |
| V1_03 | 1.39 | 1.62 | **0.92** | 1.16 | 1.50 | 1.54 | 1.52 | 53.56 | 1.58 |
| V2_01 | **1.94** | 2.00 | 2.03 | 2.22 | 2.22 | 2.34 | 2.34 | 226.02 | 2.29 |
| V2_02 | **1.22** | 1.37 | 1.41 | 1.89 | 2.31 | 2.16 | 2.20 | 180.98 | 2.12 |
| V2_03 | 1.98 | **1.30** | 1.45 | 1.58 | 1.97 | 2.19 | 2.15 | 197.53 | 2.06 |
| room1 | 0.52 | **0.43** | 0.49 | 0.56 | 0.61 | 1.48 | 1.52 | 147.49 | 1.01 |
| room2 | 0.51 | 0.50 | 0.87 | **0.36** | 1.13 | 1.60 | 1.61 | 84.13 | 1.24 |
| room3 | 0.60 | **0.58** | 1.19 | 1.03 | 1.27 | 1.73 | 1.72 | 435.13 | 1.28 |
| room4 | 0.73 | **0.69** | 0.94 | 0.72 | 1.00 | 1.51 | 1.54 | 154.36 | 1.09 |
| room5 | 0.91 | **0.46** | 0.57 | 0.52 | 0.70 | 1.43 | 1.40 | 767.34 | 0.93 |
| room6 | 0.88 | **0.36** | 0.71 | 0.50 | 0.73 | 1.59 | 1.50 | 163.44 | 1.00 |

### Per recording, checkpoints (m), seed 42 accel-first

| recording | Path Finder (ours) | SRMIST_vortexxx | ZeroDrift | VIO-Edge | Vortex | TrustNav (as shipped) | TrustNav `--no-fill --blind 0` | Ridhima Chopra | standing still |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01 | 3.89 | 3.93 | **3.17** | 4.00 | 3.74 | 4.00 | 3.99 | 594.43 | 4.11 |
| MH_02 | 3.75 | 4.06 | 3.50 | 3.90 | **3.49** | 3.62 | 3.73 | 313.53 | 4.04 |
| MH_03 | 2.24 | 2.33 | **1.53** | 2.49 | 2.38 | 3.42 | 3.43 | 67.58 | 3.47 |
| MH_04 | 4.72 | 4.42 | **3.22** | 4.38 | 5.95 | 6.50 | 6.53 | 43.44 | 6.61 |
| MH_05 | 4.63 | 4.77 | **3.19** | 4.23 | 6.46 | 6.09 | 6.12 | 56.81 | 6.17 |
| V1_01 | 1.36 | 1.19 | **1.11** | 1.42 | 1.66 | 1.80 | 1.79 | 94.04 | 1.86 |
| V1_02 | 0.76 | 0.72 | **0.71** | 1.07 | 1.20 | 1.51 | 1.60 | 114.60 | 1.67 |
| V1_03 | 1.40 | 1.41 | **0.82** | 1.05 | 1.37 | 1.47 | 1.50 | 44.79 | 1.50 |
| V2_01 | **1.78** | 1.91 | 1.94 | 2.10 | 2.10 | 2.09 | 2.10 | 161.39 | 2.19 |
| V2_02 | **1.04** | 1.13 | 1.19 | 1.59 | 2.11 | 1.65 | 1.71 | 139.69 | 1.72 |
| V2_03 | 1.77 | **0.96** | 1.07 | 1.34 | 1.82 | 1.77 | 1.79 | 146.56 | 1.84 |
| room1 | 0.47 | **0.32** | 0.37 | 0.51 | 0.51 | 0.89 | 1.05 | 124.19 | 0.90 |
| room2 | 0.46 | **0.27** | 0.92 | 0.27 | 1.20 | 1.21 | 1.25 | 57.70 | 1.29 |
| room3 | 0.51 | **0.27** | 1.02 | 0.91 | 1.25 | 1.25 | 1.58 | 328.30 | 1.18 |
| room4 | 0.72 | 0.66 | 0.90 | **0.63** | 0.96 | 1.16 | 1.10 | 113.66 | 1.09 |
| room5 | 0.78 | 0.46 | **0.41** | 0.45 | 0.65 | 1.06 | 1.03 | 558.63 | 0.89 |
| room6 | 1.03 | **0.31** | 0.73 | 0.51 | 0.73 | 1.52 | 1.39 | 115.85 | 1.08 |

### Per recording, blind 5 s (m), seed 42 accel-first

| recording | Path Finder (ours) | SRMIST_vortexxx | ZeroDrift | VIO-Edge | Vortex | TrustNav (as shipped) | TrustNav `--no-fill --blind 0` | Ridhima Chopra | standing still |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01 | 4.18 | **3.20** | 4.50 | 5.01 | 5.76 | 7.48 | 5.90 | 1722.12 | 5.18 |
| MH_02 | 4.02 | 3.04 | **1.06** | 5.04 | 3.26 | 6.83 | 6.86 | 941.34 | 5.69 |
| MH_03 | 2.13 | **0.88** | 2.99 | 1.84 | 2.44 | 4.94 | 4.59 | 242.62 | 3.18 |
| MH_04 | 2.24 | **0.47** | 1.37 | 3.69 | 4.85 | 5.86 | 5.49 | 152.65 | 4.01 |
| MH_05 | 3.07 | **0.68** | 1.28 | 4.11 | 4.38 | 6.20 | 5.70 | 187.99 | 4.23 |
| V1_01 | 1.72 | 1.54 | **1.48** | 2.39 | 3.20 | 3.38 | 3.21 | 300.87 | 1.77 |
| V1_02 | **0.48** | 0.95 | 0.91 | 0.73 | 1.75 | 2.94 | 2.65 | 326.26 | 1.08 |
| V1_03 | 1.57 | 2.03 | 0.94 | 1.16 | 2.21 | 1.44 | **0.65** | 104.89 | 1.32 |
| V2_01 | 2.63 | 3.50 | **2.31** | 3.03 | 2.90 | 5.19 | 4.58 | 523.03 | 3.22 |
| V2_02 | 1.95 | 2.43 | 3.06 | 2.93 | 2.75 | **1.86** | 4.01 | 405.00 | 2.79 |
| V2_03 | **2.09** | 2.70 | 3.49 | 2.27 | 4.42 | 4.74 | 3.93 | 449.04 | 2.63 |
| room1 | **0.52** | 0.55 | 1.10 | 0.77 | 0.58 | 2.30 | 1.93 | 265.13 | 0.54 |
| room2 | 0.63 | 0.87 | 0.87 | **0.47** | 1.47 | 1.68 | 1.60 | 234.41 | 0.67 |
| room3 | 0.69 | **0.63** | 3.43 | 1.36 | 2.72 | 1.69 | 1.28 | 926.27 | 0.68 |
| room4 | 0.69 | 0.73 | 0.67 | **0.66** | 1.34 | 1.64 | 2.14 | 319.60 | 0.76 |
| room5 | 0.69 | 0.73 | 1.67 | **0.64** | 1.25 | 1.77 | 1.40 | 1762.08 | 0.66 |
| room6 | 0.46 | 0.35 | 0.52 | **0.08** | 0.25 | 1.82 | 1.94 | 398.61 | 0.36 |

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

Vortex writes a pose for the CSV header line (timestamp 0) on some inputs. TrustNav as shipped fills gaps at 200 Hz and appends 5 s; with `--no-fill --blind 0` it collapses duplicate rows instead, so it is short by the duplicate count.

### Resources (all 68 files; heap on MH_04, V1_03, MH_05, V2_03, room6, seed 42)

| entry | CPU per s of IMU, median (max), ms | process peak RSS, median (max), MB | heap peak, warm protocol, max of 5, MiB | heap peak, cold protocol, max of 5, MiB |
|---|---:|---:|---:|---:|
| **Path Finder (ours)** | 15.8 (17.5) | 44 (45) | 1.06 | 2.85 |
| SRMIST_vortexxx | 2.1 (3.0) | 54 (56) | 2.53 | 12.94 |
| ZeroDrift | 2.5 (3.4) | 43 (44) | 2.19 | 5.20 |
| VIO-Edge | 14.3 (16.6) | 59 (60) | 4.35 | 14.36 |
| Vortex | 8.4 (9.5) | 26 (27) | 0.73 | 6.22 |
| TrustNav (as shipped) | 2.4 (2.7) | 21 (21) | – | – |
| TrustNav `--no-fill --blind 0` | 2.3 (2.8) | 21 (21) | 0.78 | 1.93 |
| Ridhima Chopra | 1.0 (1.2) | 21 (21) | 0.08 | 2.48 |

RSS includes the interpreter and numpy, so it is not the organisers' measure.

### Where each rival beats us, every set

* **ZeroDrift, seed 42, accel-first (documented order)**: Vicon rooms ATE 1.48 vs 1.31; Vicon rooms checkpoints 1.35 vs 1.14; Vicon rooms dropouts 1.36 vs 1.20; Vicon rooms drift 1 s 0.44 vs 0.25; Machine Hall ATE 4.07 vs 3.08; Machine Hall checkpoints 3.85 vs 2.92; Machine Hall blind 5 s 3.13 vs 2.24; Machine Hall dropouts 3.71 vs 2.73; Machine Hall drift 1 s 0.54 vs 0.33
* **ZeroDrift, seed 42, EuRoC file as is (gyro first)**: Vicon rooms ATE 1.54 vs 1.40; Vicon rooms checkpoints 1.45 vs 1.18; Vicon rooms dropouts 1.43 vs 1.25; Vicon rooms drift 1 s 0.46 vs 0.32; Machine Hall ATE 4.16 vs 3.38; Machine Hall checkpoints 3.91 vs 3.07; Machine Hall dropouts 3.87 vs 3.02; Machine Hall drift 1 s 0.64 vs 0.49
* **ZeroDrift, seed 1, accel-first**: Vicon rooms ATE 1.49 vs 1.37; Vicon rooms checkpoints 1.37 vs 1.18; Vicon rooms dropouts 1.39 vs 1.22; Vicon rooms drift 1 s 0.44 vs 0.26; Machine Hall ATE 4.01 vs 3.25; Machine Hall checkpoints 3.79 vs 3.11; Machine Hall blind 5 s 3.23 vs 2.37; Machine Hall dropouts 3.64 vs 2.92; Machine Hall drift 1 s 0.52 vs 0.33
* **ZeroDrift, seed 7, accel-first**: Vicon rooms ATE 1.52 vs 1.39; Vicon rooms checkpoints 1.38 vs 1.15; Vicon rooms dropouts 1.41 vs 1.21; Vicon rooms drift 1 s 0.43 vs 0.26; Machine Hall ATE 4.01 vs 3.17; Machine Hall checkpoints 3.80 vs 3.01; Machine Hall dropouts 3.63 vs 2.79; Machine Hall drift 1 s 0.52 vs 0.33
* **SRMIST_vortexxx, seed 42, accel-first (documented order)**: Vicon rooms ATE 1.48 vs 1.41; Vicon rooms checkpoints 1.35 vs 1.22; Vicon rooms drift 1 s 0.44 vs 0.30; Machine Hall ATE 4.07 vs 3.97; Machine Hall blind 5 s 3.13 vs 1.65; Machine Hall dropouts 3.71 vs 3.63; Machine Hall drift 1 s 0.54 vs 0.47; TUM-VI ATE 0.69 vs 0.50; TUM-VI checkpoints 0.66 vs 0.38; TUM-VI dropouts 0.64 vs 0.41; TUM-VI drift 1 s 0.44 vs 0.30
* **SRMIST_vortexxx, seed 42, EuRoC file as is (gyro first)**: Vicon rooms ATE 1.54 vs 1.42; Vicon rooms checkpoints 1.45 vs 1.25; Vicon rooms dropouts 1.43 vs 1.34; Vicon rooms drift 1 s 0.46 vs 0.35; Machine Hall blind 5 s 3.47 vs 3.00; Machine Hall drift 1 s 0.64 vs 0.59; TUM-VI ATE 0.69 vs 0.50; TUM-VI checkpoints 0.66 vs 0.39; TUM-VI dropouts 0.65 vs 0.42; TUM-VI drift 1 s 0.44 vs 0.30
* **SRMIST_vortexxx, seed 1, accel-first**: Vicon rooms ATE 1.49 vs 1.44; Vicon rooms checkpoints 1.37 vs 1.25; Vicon rooms dropouts 1.39 vs 1.32; Vicon rooms drift 1 s 0.44 vs 0.31; Machine Hall blind 5 s 3.23 vs 1.86; Machine Hall drift 1 s 0.52 vs 0.47; TUM-VI ATE 0.68 vs 0.51; TUM-VI checkpoints 0.65 vs 0.38; TUM-VI dropouts 0.63 vs 0.42; TUM-VI drift 1 s 0.43 vs 0.30
* **SRMIST_vortexxx, seed 7, accel-first**: Vicon rooms ATE 1.52 vs 1.43; Vicon rooms checkpoints 1.38 vs 1.25; Vicon rooms dropouts 1.41 vs 1.31; Vicon rooms drift 1 s 0.43 vs 0.34; Machine Hall blind 5 s 3.35 vs 1.99; Machine Hall drift 1 s 0.52 vs 0.47; TUM-VI ATE 0.68 vs 0.49; TUM-VI checkpoints 0.64 vs 0.41; TUM-VI dropouts 0.64 vs 0.42; TUM-VI drift 1 s 0.43 vs 0.29
* **VIO-Edge, seed 42, accel-first (documented order)**: TUM-VI ATE 0.69 vs 0.61; TUM-VI checkpoints 0.66 vs 0.55; TUM-VI dropouts 0.64 vs 0.57; TUM-VI drift 1 s 0.44 vs 0.37
* **VIO-Edge, seed 42, EuRoC file as is (gyro first)**: Machine Hall checkpoints 3.91 vs 3.79; TUM-VI ATE 0.69 vs 0.61; TUM-VI checkpoints 0.66 vs 0.54; TUM-VI dropouts 0.65 vs 0.56; TUM-VI drift 1 s 0.44 vs 0.36
* **VIO-Edge, seed 1, accel-first**: TUM-VI checkpoints 0.65 vs 0.60
* **VIO-Edge, seed 7, accel-first**: TUM-VI ATE 0.68 vs 0.62; TUM-VI checkpoints 0.64 vs 0.56; TUM-VI dropouts 0.64 vs 0.58; TUM-VI drift 1 s 0.43 vs 0.38

## Reproduce

```bash
git clone --depth 1 https://github.com/<owner>/<repo>.git ../competitors/<owner>_<repo>    # each repo above
python -m venv ../competitors/.venv && ../competitors/.venv/bin/pip install numpy pandas
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python ../competitors/.venv/bin/python
python tools/competitor_scoreboard.py run  --competitors ../competitors --cpy ../competitors/.venv/bin/python --tag all
python tools/competitor_scoreboard.py heap --competitors ../competitors --cpy ../competitors/.venv/bin/python --protocol warm
python tools/competitor_scoreboard.py heap --competitors ../competitors --cpy ../competitors/.venv/bin/python --protocol cold
python tools/competitor_tables.py results/track3/competitors_raw.json \
    --heap results/track3/competitors_heap_warm.json --heap-cold results/track3/competitors_heap_cold.json
```
