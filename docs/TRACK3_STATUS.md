# Where everything is (16 Nov 2026)

## What we are trying to achieve

Samsung scores one thing: a program that reads a degraded IMU file (`imu0/data.csv`, EuRoC format), writes
`estimated_trajectory.txt` (TUM format, one pose per input row), estimates the path through the mission, keeps
predicting for 5 s after the IMU dies, and stays inside **8 MB peak RAM and 20 ms per second of IMU on one thread**.
100 points: memory 20, runtime 20, checkpoints 30, blind 5 s 20, dropouts 10. Ranked with `evo` (ATE + RPE).
The honest bar for accuracy: an IMU alone drifts without bound, so "stand still at the start" is hard to beat.

## Everything is on `main` (merged from `track3-entry`, #12 and #29, 16 Nov)

| Folder | What | Merged from |
|---|---|---|
| `src/` | **The scored entry.** `main.cpp` (submitted, C++17 + zlib), `main.py` (same logic, readable), the AirIO-style network `velnet_weights.bin` (42,716 params) and the gravity-aligned fallback `velnet_weights_world.bin` (27,020) used by the mount guard, Dockerfile | #12, #22, #27, #28 |
| `tests/stress/` | Output contract (24), review fixes, mount guard: 38 tests on the C++ entry | #21, #23, #28 |
| `tools/` | degradation stand-in, scorer, evo scorer, blind-5 s study, bundle script, trainers, fetchers | #12, #22, #25, #27 |
| `pathfinder/`, `tests/` | Nikhil's redesigned estimator (#10) and the real-vision / reproducibility workstream (#26) | #10, #26 |
| `demo_line/` | The camera + multi-IMU demo line kept whole with its **own copy** of `pathfinder/`: live phone demo, real VR headsets, saturation and long-outage work, Mayank's stress suite with its fixes, final benchmark. Run from inside the folder | #4-#11, #20, #24 |

Not scored: everything outside `src/`. It is the material for the slides, the video and the final demo (27 Nov).

## The scored entry today

| | Value |
|---|---|
| Peak RAM / compute (C++, Apple M5, 16 Nov) | **2.28 MB** of 8 / **1.96 ms** per second of IMU of 20, worst of 3 recordings (`results/track3/budget.json`); Docker on Linux 3.2 MB (Mayank); Windows 5.2 MB / 6 ms |
| Output contract | 24 tests: one pose per row through NaN, frozen, zero and missing rows, cut files, rotated gravity |
| Held-out accuracy (EuRoC MH_05 + V2_03 + TUM-VI room6, seeds 1-3; ATE / checkpoints / blind 5 s / dropouts, m) | **3.08 / 2.86 / 2.34 / 3.15** (exact mean 3.083), vs stand still 3.28 / 3.03 / 2.38 / 3.27 and the previous network 3.19 / 2.93 / 2.36 / 3.27. Re-run on the final code, raw files in `results/track3/`; reproduced independently (Codex) |
| All 16 recordings (includes training recordings), seeds 1-2 | 2.44 / 2.23 / 2.34 / 2.39 vs stand still 2.71 / 2.55 / 2.38 / 2.46; better on ATE and checkpoints in 32/32 runs, blind 25/32, dropouts 20/32 |
| **On the organisers' degradation script** (16 Nov; 5 recordings, their seed) | ATE **3.40** vs 3.66 m standing still (accel-first input) and **3.37** vs 3.66 (EuRoC order); ahead on checkpoints, blind and dropouts too (`results/track3/README.md`). The entry detects column order and units itself |
| Real `evo` on all 11 EuRoC (previous network, Mayank, `docs/TRACK3_EVAL.md`) | our scorer matches evo on ATE to 0.01 % |
| Mount guard (`docs/MOUNT_GUARD.md`) | if start-up gravity is > 25° from both training mounts (rotated IMU), switch to the gravity-aligned network. Quiet and byte-identical on all 16 normal recordings (`results/track3/mount_guard_check.json`); switches on 14/16 rotated copies (one not at rest at start-up, one rotation inside the threshold); ahead of the AirIO network on every rotated case it catches. Holds off when the device is not at rest at start-up (#29). `--mount-guard-deg 0` turns it off |
| Python = C++ | within 1e-6 m |
| CI | entry (build, 38 tests, Python reference), Docker (incl. the guard inside the image), Nikhil's suite on Ubuntu and Windows, `demo_line/` tests |

"Dropouts" is scored over each gap plus the second after it; with missing rows that is mostly the second after the
gap (standing still is scored the same way). Known and left as is: a free-falling IMU (|a| < 1 m/s²) is treated as a
bad sample, gyro included; real recordings never fall freely, and changing it would change the scored filter.

Limits we state: the gain over standing still is about 6 % on held-out data; the blind 5 s is close to a tie (its
error is set before the IMU dies); results use our degradation stand-in, not the organisers' script.

## Reviews done

- **Independent review 1** (Raghav + an AI reviewer): four faults reproduced and fixed with regression tests
  (`tests/stress/test_review_fixes.py`): NaN start with zero acceleration, lost `--extend-blind` at 200 rows, bundle
  RAM measuring the compiler, zip memory growing with archive size. Claims corrected.
- **Codex reviews** of #27 (numbers reproduced, no leakage, parity), the mount guard (two fixes made), and this
  consolidated branch (the Docker image was missing the fallback network: fixed, now tested in CI).

## Left before 18 Nov, 11:59 PM

1. Done: Nikhil reviewed and merged everything into `main` (#26, #12 with his review fixes #29).
2. Slides and video (at the end). Slide fixes: architecture "single pass, constant memory" → "streamed, bounded
   memory, one short pre-scan"; results footnote "physics only" → "original filter settings"; RPE → "1 s position
   drift"; new numbers 3.08 m and the mount guard.
3. The real `estimated_trajectory.txt`: `python tools/make_bundle.py <organisers' input>` as soon as it arrives.
4. Repo shared or public, Google Form.
