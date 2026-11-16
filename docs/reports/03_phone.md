# Report 3: Samsung Galaxy M36 (Exynos 1380, Android 17) — commit b725bec, 17 Nov 2026

Two things measured on a real phone: the scored entry running **on the phone's own CPU**, and the live camera + IMU
demo with the camera covered.

## 1. The scored entry on the phone

The same `src/main.cpp`, cross-compiled with the Android NDK (aarch64 clang++ `-O3`, static libc++), pushed to the
phone over adb and run from a shell on the phone with `tools/android/bench_on_phone.sh` (a small `wait4` runner,
`tools/android/measure.c`, reports the process's own peak RSS and CPU; nothing installed, files removed afterwards).
Same 5 degraded recordings as the Mac report, 5 runs each. Raw: [`results/reports/phone_SM-M366B.json`](../../results/reports/phone_SM-M366B.json).

| Recording | IMU rows | Poses written | Peak RAM | CPU per s of IMU |
|---|---:|---:|---:|---:|
| EuRoC MH_05 | 21,521 | 21,521 | 4.02 MB | 4.58 ms |
| EuRoC V2_03 | 22,170 | 22,170 | 4.04 MB | 4.97 ms |
| TUM-VI room6 | 25,084 | 25,084 | 4.04 MB | 4.83 ms |
| EuRoC MH_04 | 19,120 | 19,120 | 4.04 MB | 4.59 ms |
| EuRoC V1_03 | 20,300 | 20,300 | 4.01 MB | 4.82 ms |
| **Worst** | | **one pose per row** | **4.04 MB of 8** | **4.97 ms of 20** |

On a mid-range phone chip the entry uses half the memory budget and a quarter of the time budget, so as an Android
library or APK it would run alongside a camera pipeline without strain. Memory is higher than on the Mac (2.5 MB)
because of the Android C/C++ runtime, not the estimator.

## 2. Live demo: camera covered while walking

Phone app ("Path Finder bridge", `demo_line/android/`) streams the phone's IMU and ARCore poses over adb (wireless
debugging) to the laptop; three trackers run live (`demo_line/tools/live_demo/`): ARCore alone, a standard tracker
that trusts every IMU reading, and Path Finder with its sensor checks and outage logic. Simulated faults off.

Walk: 20 s tracked, then the back camera covered with a hand while walking, then uncovered. Analysed with
`demo_line/tools/live_demo/analyze.py`:

| Camera covered for 48.3 s while walking | Error when ARCore returned |
|---|---:|
| Standard tracker | **227.25 m** |
| **Path Finder** | **3.51 m** (and it reported itself `valid = False`, ±12.6 m, during the outage) |

Total walk 36.5 m (ARCore path length); start-to-end closure 3.3 m for all three. This is the live, camera-plus-IMU
system (`demo_line/`), not the scored IMU-only entry; it shows the same idea — distrust a sensor that cannot be
believed and say how unsure you are — on real hardware.
