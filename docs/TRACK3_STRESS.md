# Track 3 entry: stress and scoring (Mayank)

Entry under test: `src/main.py` at `track3-entry` 6d31f9b. Machine: MacBook Air, Apple M4, macOS, Python 3.11.
Degradation: `tools/degrade_imu.py` (the team's stand-in for the organiser's script; progressive noise and bias drift,
3 x 2 s dropouts, 5 s death). Scoring: the real `evo` library (`tests/stress/track3_eval.py`), SE(3) alignment fitted on
the mission part only so the blind 5 s is a true prediction. EuRoC IMU + ground truth fetched with `tools/fetch_euroc.py`.

## Contract (`tests/stress/test_main_contract.py`, 24 tests)
23 pass. Covered: one pose per input row, seconds timestamps, unit quaternions, header variants (none, CRLF, no `#`),
low and scientific precision, no final newline, NaN / inf / zero / frozen / 1e30 rows (2 s), NaN in one channel,
2 s timestamp gap, duplicate and out-of-order timestamps, file cut mid-row, 1-3 row files, header-only file, 1.4e18 ns
timestamps, IMU frame rotated 90 deg (unknown gravity), `--extend-blind 5`, runtime.

**Fails: peak RAM.** 20.2-20.4 MiB (`/usr/bin/time -l`) in every run against the 8 MiB budget. A bare Python start is
already about 10 MiB, so this needs the planned C++ port. Runtime is fine: 1.6-2.9 ms of wall time per second of IMU
(budget 20 ms), including interpreter start-up and CSV parsing.

## Accuracy (evo, 11 recordings available, 5 used x 3 seeds, plus death and dropout variants on MH_01)
Positions in metres. "blind" = error over the 5 s after the IMU dies (mean / final). "drop" = mean error inside and 1 s after the 2 s dropouts.

| recording | ATE | blind mean (range over seeds) | drop | RPE 1 s |
|---|---:|---:|---:|---:|
| MH_01_easy | 3.3-4.0 | 4.6-8.1 | 2.1-4.4 | 0.66-0.69 |
| MH_03_medium | 5.6-6.0 | 2.3-5.1 | 2.9-4.4 | 1.46-1.61 |
| V1_01_easy | 4.0-4.2 | 1.5-3.9 | 2.8-3.7 | 0.78-0.82 |
| V1_03_difficult | 1.9-2.6 | 1.5-1.8 | 2.2-3.1 | 0.71-0.81 |
| V2_02_medium | 4.6-5.4 | 3.9-8.7 | 3.3-5.6 | 1.18-1.33 |

* Death as NaN, frozen values or missing rows gives the same numbers as zeros (the entry treats them alike); dropouts as missing rows, zeros or frozen values are within 0.1 m of each other.
* All 33 runs: exit code 0, one pose per input row (plus 1000 with `--extend-blind 5` when the death is "missing rows"), unit quaternions.
* The blind-5 s error varies 4x between seeds on the same recording (MH_01: 4.6, 7.3, 8.1 m), so a single-seed number would mislead; the error is mostly the drift already accumulated before the death, since the mission ATE is 3-6 m.
* These are IMU-only metres over 130-190 s runs; scores depend on the organiser's degradation and hidden ground truth, which we do not have.

Reproduce: `python tools/fetch_euroc.py && python tests/stress/track3_eval.py --seeds 1 2 3`
