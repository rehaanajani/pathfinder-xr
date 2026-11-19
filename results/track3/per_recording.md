# Per-recording results, organisers' degradation script

All 11 EuRoC and 6 TUM-VI recordings, damaged by the organisers' `degrade_imu.py` (their seed, accel-first input),
run with `src/main.py` and scored with `tools/score_track3.py`. Each cell: Path Finder / standing still, metres;
**bold** = Path Finder lower. Raw numbers: [`organiser_per_recording.json`](organiser_per_recording.json); the C++ port
gives the same scores ([`organiser_per_recording_cpp.json`](organiser_per_recording_cpp.json)).

| Recording | ATE | Checkpoints | Blind 5 s | Dropouts | Drift 1 s |
|---|---:|---:|---:|---:|---:|
| MH_01_easy | **4.14** / 4.32 | **3.92** / 4.11 | **4.42** / 5.18 | **3.56** / 3.78 | **0.43** / 0.50 |
| MH_02_easy | **4.16** / 4.67 | **3.76** / 4.04 | **4.75** / 5.69 | **3.66** / 3.96 | **0.44** / 0.53 |
| MH_03_medium | **2.67** / 3.59 | **2.34** / 3.47 | **2.44** / 3.18 | **2.28** / 3.25 | **0.64** / 1.19 |
| MH_04_difficult | **4.99** / 6.81 | **5.02** / 6.61 | **2.66** / 4.01 | **4.81** / 6.45 | **0.78** / 1.12 |
| MH_05_difficult | **5.09** / 6.87 | **4.71** / 6.17 | **3.23** / 4.23 | **4.86** / 6.54 | **0.73** / 1.04 |
| V1_01_easy | **1.56** / 1.85 | **1.38** / 1.86 | **1.70** / 1.77 | **1.43** / 1.69 | **0.35** / 0.44 |
| V1_02_medium | **0.87** / 1.78 | **0.80** / 1.67 | **0.63** / 1.08 | **0.78** / 1.63 | **0.44** / 0.96 |
| V1_03_difficult | **1.31** / 1.58 | **1.33** / 1.50 | **1.27** / 1.32 | **1.25** / 1.35 | **0.55** / 0.73 |
| V2_01_easy | **1.94** / 2.29 | **1.81** / 2.19 | **2.51** / 3.22 | **1.84** / 2.21 | **0.26** / 0.34 |
| V2_02_medium | **1.27** / 2.11 | **1.04** / 1.72 | **1.97** / 2.79 | **1.17** / 1.90 | **0.43** / 0.75 |
| V2_03_difficult | **1.98** / 2.06 | **1.78** / 1.84 | **2.36** / 2.63 | **1.82** / 1.92 | **0.74** / 0.80 |
| room1 | **0.57** / 1.01 | **0.52** / 0.90 | **0.52** / 0.54 | **0.53** / 0.97 | **0.51** / 0.92 |
| room2 | **0.66** / 1.24 | **0.61** / 1.29 | 0.70 / 0.67 | **0.62** / 1.27 | **0.51** / 0.95 |
| room3 | **0.72** / 1.28 | **0.61** / 1.18 | **0.64** / 0.68 | **0.70** / 1.29 | **0.50** / 0.99 |
| room4 | **0.75** / 1.09 | **0.73** / 1.09 | **0.72** / 0.76 | **0.75** / 1.08 | **0.33** / 0.57 |
| room5 | **0.91** / 0.93 | **0.86** / 0.89 | **0.62** / 0.66 | **0.85** / 0.91 | **0.72** / 0.81 |
| room6 | **0.87** / 1.00 | **1.02** / 1.08 | 0.40 / 0.36 | **0.79** / 0.96 | **0.48** / 0.53 |
| **Mean** | **2.03** / 2.62 | **1.90** / 2.45 | **1.85** / 2.28 | **1.86** / 2.42 | **0.52** / 0.77 |
| **Recordings better** | 17 / 17 | 17 / 17 | 15 / 17 | 17 / 17 | 17 / 17 |

Monado headset recordings (8, IMU mounted unlike the training data, so the mount guard selects the gravity-aligned network; rotor drag stays off): ATE 0.57 vs 0.83 m, better on 8 / 8.
