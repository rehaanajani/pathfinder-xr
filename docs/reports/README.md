# Benchmark reports

| # | Machine | Code | What |
|---|---|---|---|
| [1](01_windows_laptop.md) | Windows laptop, Intel i7-11800H (WSL) | `main` c382494 (16 Nov) | First independent benchmark of the merged repository: resources, output contracts, camera + IMU benchmark, all tests |
| [2](02_macbook.md) | MacBook Air, Apple M5 | b725bec (17 Nov, final) | Resources, held-out accuracy, accuracy on the organisers' own degradation script, all tests |
| [3](03_phone.md) | Samsung Galaxy M36, Exynos 1380 | b725bec | The scored entry run on the phone's CPU; live camera-covered demo |

## The C++ entry on three machines

| | Windows laptop (WSL) | MacBook Air M5 | Galaxy M36 phone | Limit |
|---|---:|---:|---:|---:|
| Peak RAM, worst | 5.06 MiB | **2.47 MB** | 4.04 MB | 8 |
| CPU per second of IMU, worst recording | 13.97 ms | **1.84 ms** (median of 5 runs) | 4.97 ms (best of 5 runs) | 20 |
| Output contract | 114/114 | 56 tests pass | one pose per row on all 5 recordings | |

**Which is better?** Accuracy does not depend on the machine: the arithmetic is fixed (`-O3` without fast-math, no
platform-specific instructions). The trajectory the phone wrote for MH_05 was **byte-identical** to the Mac's when we compared them on 17 Nov (the file hash was not recorded), and C++
matches the Python reference to 1e-6 m. On x86 (the Windows laptop) we have not compared output files. Only speed and
memory differ:

- **Fastest: the Mac** (Apple M5, 1.8 ms). **Slowest: the Windows laptop under WSL** (14 ms): an older 2021 Intel core,
  measured through WSL's virtualisation with other work running, on code before the `-O3` / contiguous-memory change.
  Still inside the 20 ms budget. A native Linux or Docker run is closer to the Mac: Mayank measured 3.2 MB / 4.2 ms in
  Docker (`docs/TRACK3_EVAL.md`).
- **The phone sits between them** (5 ms, 4 MB): a mid-range Android chip runs the entry at a quarter of the time
  budget. Memory is higher than on the Mac only because of the Android C/C++ runtime.
- **Memory** varies with each platform's runtime (Windows 5 MiB includes its system libraries; macOS 2.5 MB; Android
  4 MB), never with the length of the recording: the input is streamed.

**Accuracy** (from report 2): on held-out recordings ATE 3.08 m vs 3.28 m standing still; on the organisers' own
degradation script 3.40 m vs 3.66 m (7-8 % better). On the camera + IMU benchmark (report 1) Path Finder beats
physics-only and standing still in 38/38 scenarios, and on the live phone test (report 3) it was far
closer than a standard tracker in all four camera blackouts (e.g. 3.5 m against 227 m after 48 s blind, measured
against ARCore's pose when the camera came back), though in that small room simply freezing in place was closer
still (1.8 m).

Reproduce: `bash tools/benchmark_report.sh <organisers' degrade_imu.py>` (computer) and
`bash tools/android/bench_on_phone.sh <degraded folders>` (phone, USB or wireless debugging).
