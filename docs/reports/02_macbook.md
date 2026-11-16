# Report 2: MacBook Air (Apple M5, macOS) — commit b725bec, 17 Nov 2026

Frozen benchmark of the final entry, produced by `tools/benchmark_report.sh` (raw files:
[`results/reports/20261117-macbook-air-m5-b725bec/`](../../results/reports/20261117-macbook-air-m5-b725bec/)).
C++ entry, default settings (AirIO network + mount guard + input layout detection), Apple clang 21, `-O3`, one thread.

## Resources (5 recordings, 5 runs each; the process's own usage via `wait4`)

| Recording | IMU s | Peak RAM | CPU per s of IMU |
|---|---:|---:|---:|
| EuRoC MH_05 | 113.6 | 2.47 MB | 1.78 ms |
| EuRoC V2_03 | 116.8 | 2.45 MB | 1.82 ms |
| TUM-VI room6 | 131.8 | 2.47 MB | 1.84 ms |
| EuRoC MH_04 | 101.6 | 2.45 MB | 1.81 ms |
| EuRoC V1_03 | 107.5 | 2.45 MB | 1.79 ms |
| **Worst** | | **2.47 MB of 8** | **1.84 ms of 20** |

## Accuracy (ATE / checkpoints / blind 5 s / dropouts, metres; lower is better)

**Held-out** (EuRoC MH_05, V2_03, TUM-VI room6; never used for training or tuning; our degradation stand-in, seeds 1-3):

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| **Ours** | **3.08** | **2.86** | **2.34** | **3.15** |
| Standing still | 3.28 | 3.03 | 2.38 | 3.27 |
| Earlier network | 3.19 | 2.93 | 2.36 | 3.27 |

**The organisers' own degradation script** (16 Nov; MH_04, V1_03, MH_05, V2_03, room6; their fixed seed):

| Input layout | Ours | Standing still |
|---|---|---|
| accel first (their documented order) | **3.40 / 3.22 / 2.47 / 3.21** | 3.66 / 3.44 / 2.51 / 3.44 |
| EuRoC order (gyro first) | **3.37 / 3.18 / 2.44 / 3.18** | 3.66 / 3.44 / 2.51 / 3.44 |

## Tests

| Suite | Result |
|---|---|
| C++ entry, all stress tests (contract, review fixes, mount guard, input layout) | 56 passed |
| Root suite (estimator, vision, tools, Python reference) | 264 passed, 2 skipped |
| Demo line (run inside `demo_line/`) | 86 passed, 26 skipped (datasets not on this machine), 1 expected failure |
