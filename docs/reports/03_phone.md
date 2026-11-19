# Report 3: Samsung Galaxy M36 (Exynos 1380, Android 17) — commit b725bec, 17 Nov 2026

Two things measured on a real phone: the C++ port of the entry running **on the phone's own CPU**, and the live camera + IMU
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

CPU is the best of the 5 runs for each recording (the Mac report gives the median); peak RAM is the largest of the 5.

On a mid-range phone chip the entry uses half the memory budget and a quarter of the time budget, so as an Android
library or APK it would run alongside a camera pipeline without strain. Memory is higher than on the Mac (2.5 MB)
because of the Android C/C++ runtime, not the estimator.

## 2. Live demo: camera covered while walking

Phone app ("Path Finder bridge", `demo_line/android/`) streams the phone's IMU and ARCore poses over adb (wireless
debugging) to the laptop; three trackers run live (`demo_line/tools/live_demo/`): ARCore alone, a standard tracker
that trusts every IMU reading, and Path Finder with its sensor checks and outage logic. Simulated faults off.

Walk: 20 s tracked, then the back camera covered with a hand while walking, then uncovered. Analysed with
`demo_line/tools/live_demo/analyze.py`:

Four camera blackouts in the full 14-minute session (80 m walked). Error is the distance to ARCore's pose when the
camera came back (ARCore is the reference here, not independent ground truth):

| Blackout | Standard tracker | Path Finder | Freezing at the last position | Path Finder's stated uncertainty (the analyser marks `valid` when it is under 1 m) |
|---:|---:|---:|---:|---|
| 48.3 s | 227.25 m | 3.51 m | **1.78 m** | ±12.6 m, `valid = False` |
| 13.0 s | 12.18 m | 0.47 m | **0.38 m** | ±3.4 m, `valid = False` |
| 15.0 s | 180.71 m | 10.05 m | **0.91 m** | ±3.9 m, `valid = False`: it **underestimated** its own error here |
| 1.0 s | 1.33 m | 1.17 m | **0.89 m** | ±0.5 m, `valid = True` |

Path Finder is far closer than the standard tracker every time and flags every long blackout as not valid. But the
room was small (ARCore saw the phone move within about 6 × 9 m), and simply freezing at the last position was closer
than Path Finder in all four blackouts. In the 15 s case its stated uncertainty was too small (honest uncertainty is a
stated limit). The scored entry's pull back toward the start comes from the same lesson. Start-to-end closure 2.6 m for all three trackers.
Raw session: `demo_line/runs/live/phone_test_17nov_full.jsonl` (kept locally; not in git). This is the live, camera-plus-IMU
system (`demo_line/`), not the scored IMU-only entry; it shows the same idea — distrust a sensor that cannot be
believed and say how unsure you are — on real hardware.
