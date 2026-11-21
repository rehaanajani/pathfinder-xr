# Per-recording results, organisers' degradation script

All 11 EuRoC and 6 TUM-VI recordings, damaged by the organisers' `degrade_imu.py` (their seed, accel-first input),
run with `src/main.py` at commit `7005f0b` and scored with `tools/score_track3.py`. Each cell: Path Finder /
standing still, metres; **bold** = Path Finder lower. Raw numbers: [`organiser_per_recording.json`](organiser_per_recording.json);
the C++ port gives the same scores within 1e-06 m ([`organiser_per_recording_cpp.json`](organiser_per_recording_cpp.json)).
Regenerate with `python tools/make_per_recording.py --script <degrade_imu.py> --commit <hash>`.

| Recording | ATE | Checkpoints | Blind 5 s | Dropouts | Drift 1 s |
|---|---:|---:|---:|---:|---:|
| MH_01_easy | **2.47** / 4.32 | **2.49** / 4.11 | **2.34** / 5.18 | **2.13** / 3.78 | **0.33** / 0.50 |
| MH_02_easy | **2.93** / 4.67 | **2.55** / 4.04 | **2.79** / 5.69 | **2.52** / 3.96 | **0.34** / 0.53 |
| MH_03_medium | **1.23** / 3.59 | **0.96** / 3.47 | **1.31** / 3.18 | **1.08** / 3.25 | **0.35** / 1.19 |
| MH_04_difficult | **2.95** / 6.81 | **2.89** / 6.61 | **1.43** / 4.01 | **2.77** / 6.45 | **0.58** / 1.12 |
| MH_05_difficult | **3.19** / 6.87 | **2.73** / 6.17 | **2.32** / 4.23 | **3.10** / 6.54 | **0.58** / 1.04 |
| V1_01_easy | **1.07** / 1.85 | **1.01** / 1.86 | **1.07** / 1.77 | **1.01** / 1.69 | **0.27** / 0.44 |
| V1_02_medium | **0.79** / 1.78 | **0.73** / 1.67 | **0.76** / 1.08 | **0.80** / 1.63 | **0.33** / 0.96 |
| V1_03_difficult | 1.77 / 1.58 | 1.80 / 1.50 | 1.77 / 1.32 | 1.70 / 1.35 | **0.62** / 0.73 |
| V2_01_easy | **1.62** / 2.29 | **1.53** / 2.19 | **2.08** / 3.22 | **1.63** / 2.21 | **0.25** / 0.34 |
| V2_02_medium | **1.17** / 2.11 | **1.00** / 1.72 | **1.74** / 2.79 | **1.09** / 1.90 | **0.37** / 0.75 |
| V2_03_difficult | **1.87** / 2.06 | **1.54** / 1.84 | **1.77** / 2.63 | **1.73** / 1.92 | **0.73** / 0.80 |
| room1 | **0.47** / 1.01 | **0.45** / 0.90 | 0.59 / 0.54 | **0.43** / 0.97 | **0.34** / 0.92 |
| room2 | **0.40** / 1.24 | **0.32** / 1.29 | **0.63** / 0.67 | **0.33** / 1.27 | **0.25** / 0.95 |
| room3 | **0.51** / 1.28 | **0.41** / 1.18 | 0.70 / 0.68 | **0.46** / 1.29 | **0.31** / 0.99 |
| room4 | **0.73** / 1.09 | **0.70** / 1.09 | **0.69** / 0.76 | **0.72** / 1.08 | **0.30** / 0.57 |
| room5 | 0.95 / 0.93 | **0.72** / 0.89 | 0.74 / 0.66 | **0.86** / 0.91 | **0.68** / 0.81 |
| room6 | **0.90** / 1.00 | **1.04** / 1.08 | 0.54 / 0.36 | **0.82** / 0.96 | **0.48** / 0.53 |
| **Mean** | **1.47** / 2.62 | **1.34** / 2.45 | **1.37** / 2.28 | **1.36** / 2.42 | **0.42** / 0.77 |
| **Recordings better** | 15 / 17 | 16 / 17 | 12 / 17 | 16 / 17 | 17 / 17 |

Monado headset recordings (8, IMU mounted unlike the training data, so the mount guard selects the gravity-aligned network; rotor drag stays off): ATE 0.59 vs 0.83 m, better on 8 / 8.
