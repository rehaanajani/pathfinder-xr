# Per-recording results, organisers' degradation script

All 11 EuRoC and 6 TUM-VI recordings, damaged by the organisers' `degrade_imu.py` (their seed, accel-first input),
run with `src/main.py` at commit `fa7b2af` and scored with `tools/score_track3.py`. Each cell: Path Finder /
standing still, metres; **bold** = Path Finder lower. Raw numbers: [`organiser_per_recording.json`](organiser_per_recording.json);
the C++ port gives the same scores within 8e-07 m ([`organiser_per_recording_cpp.json`](organiser_per_recording_cpp.json)).
Regenerate with `python tools/make_per_recording.py --script <degrade_imu.py> --commit <hash>`.

| Recording | ATE | Checkpoints | Blind 5 s | Dropouts | Drift 1 s |
|---|---:|---:|---:|---:|---:|
| MH_01_easy | **4.11** / 4.32 | **3.89** / 4.11 | **4.18** / 5.18 | **3.55** / 3.78 | **0.41** / 0.50 |
| MH_02_easy | **4.10** / 4.67 | **3.75** / 4.04 | **4.02** / 5.69 | **3.62** / 3.96 | **0.38** / 0.53 |
| MH_03_medium | **2.45** / 3.59 | **2.24** / 3.47 | **2.13** / 3.18 | **2.12** / 3.25 | **0.49** / 1.19 |
| MH_04_difficult | **4.65** / 6.81 | **4.72** / 6.61 | **2.24** / 4.01 | **4.49** / 6.45 | **0.70** / 1.12 |
| MH_05_difficult | **5.02** / 6.87 | **4.63** / 6.17 | **3.07** / 4.23 | **4.77** / 6.54 | **0.70** / 1.04 |
| V1_01_easy | **1.54** / 1.85 | **1.36** / 1.86 | **1.72** / 1.77 | **1.39** / 1.69 | **0.33** / 0.44 |
| V1_02_medium | **0.80** / 1.78 | **0.76** / 1.67 | **0.48** / 1.08 | **0.70** / 1.63 | **0.36** / 0.96 |
| V1_03_difficult | **1.39** / 1.58 | **1.40** / 1.50 | 1.57 / 1.32 | **1.32** / 1.35 | **0.53** / 0.73 |
| V2_01_easy | **1.94** / 2.29 | **1.78** / 2.19 | **2.63** / 3.22 | **1.82** / 2.21 | **0.26** / 0.34 |
| V2_02_medium | **1.22** / 2.11 | **1.04** / 1.72 | **1.95** / 2.79 | **1.14** / 1.90 | **0.40** / 0.75 |
| V2_03_difficult | **1.98** / 2.06 | **1.77** / 1.84 | **2.09** / 2.63 | **1.82** / 1.92 | **0.74** / 0.80 |
| room1 | **0.52** / 1.01 | **0.47** / 0.90 | **0.52** / 0.54 | **0.48** / 0.97 | **0.41** / 0.92 |
| room2 | **0.51** / 1.24 | **0.46** / 1.29 | **0.63** / 0.67 | **0.46** / 1.27 | **0.37** / 0.95 |
| room3 | **0.60** / 1.28 | **0.51** / 1.18 | 0.69 / 0.68 | **0.58** / 1.29 | **0.39** / 0.99 |
| room4 | **0.73** / 1.09 | **0.72** / 1.09 | **0.69** / 0.76 | **0.72** / 1.08 | **0.31** / 0.57 |
| room5 | **0.91** / 0.93 | **0.78** / 0.89 | 0.69 / 0.66 | **0.83** / 0.91 | **0.69** / 0.81 |
| room6 | **0.88** / 1.00 | **1.03** / 1.08 | 0.46 / 0.36 | **0.79** / 0.96 | **0.48** / 0.53 |
| **Mean** | **1.96** / 2.62 | **1.84** / 2.45 | **1.75** / 2.28 | **1.80** / 2.42 | **0.47** / 0.77 |
| **Recordings better** | 17 / 17 | 17 / 17 | 13 / 17 | 17 / 17 | 17 / 17 |

Monado headset recordings (8, IMU mounted unlike the training data, so the mount guard selects the gravity-aligned network; rotor drag stays off): ATE 0.57 vs 0.83 m, better on 8 / 8 (measured at `85b3a30`, before dropout bridging; not re-run here).
