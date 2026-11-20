# Benchmark reports

The current headline numbers are in the [main README](../../README.md): `src/main.py` uses 3.5 MiB of heap for the
algorithm (6.6 MiB counting everything after numpy, including compiling `main.py`) and 5.8 ms per second of IMU
([`python_budget.json`](../../results/track3/python_budget.json)), and is about 45 % below standing still on the
scored metrics ([`per_recording.md`](../../results/track3/per_recording.md)).

These three reports measure the **C++ port** of the algorithm (`src/main.cpp`, output within 3 µm of `main.py`) on
three machines, plus the live phone demo. They predate rotor-drag aiding (19 Nov):

| # | Machine | Code | What |
|---|---|---|---|
| [1](01_windows_laptop.md) | Windows laptop, Intel i7-11800H (WSL) | `main` c382494 (16 Nov) | First independent benchmark of the merged repository. Superseded: both failures it lists were fixed afterwards (see the report's last section) |
| [2](02_macbook.md) | MacBook Air, Apple M5 | b725bec (17 Nov) | Resources, accuracy on the organisers' script (5 recordings), all tests |
| [3](03_phone.md) | Samsung Galaxy M36, Exynos 1380 | b725bec | The C++ port on the phone's own CPU; the live camera-covered demo |

## The C++ port on three machines

| | Windows laptop (WSL) | MacBook Air M5 | Galaxy M36 phone | Limit |
|---|---:|---:|---:|---:|
| Peak RAM (process), worst | 5.06 MiB | **2.47 MB** | 4.04 MB | 8 |
| CPU per second of IMU, worst recording | 13.97 ms | **1.84 ms** (median of 5 runs) | 4.97 ms (best of 5 runs) | 20 |
| Output contract | 114/114 | 56 tests pass | one pose per row on all 5 recordings | |

Accuracy does not depend on the machine: the arithmetic is fixed (`-O3` without fast-math, no platform-specific
instructions). The trajectory the phone wrote for MH_05 matched the Mac's byte for byte when compared on 17 Nov.
Only speed and memory differ: the Mac is fastest (1.8 ms), the Windows laptop under WSL slowest (14 ms, measured with
other work running, on code before the `-O3` change), the phone in between (5 ms). Memory varies with each platform's
runtime, never with the length of the recording: the input is streamed.

Reproduce: `bash tools/benchmark_report.sh <organisers' degrade_imu.py>` (computer) and
`bash tools/android/bench_on_phone.sh <degraded folders>` (phone, USB or wireless debugging); for `main.py`,
`python tools/measure_python_budget.py`.
