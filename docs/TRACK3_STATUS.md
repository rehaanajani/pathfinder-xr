# Where everything is (15 Nov 2026)

## What we are trying to achieve

Samsung scores one thing: a program that reads a degraded IMU file (`imu0/data.csv`, EuRoC format), writes
`estimated_trajectory.txt` (TUM format, one pose per input row), estimates the path through the mission, keeps
predicting for 5 s after the IMU dies, and stays inside **8 MB peak RAM and 20 ms per second of IMU on one thread**.
100 points: memory 20, runtime 20, checkpoints 30, blind 5 s 20, dropouts 10. Ranked with `evo` (ATE + RPE).

Everything else we built (camera health, visual odometry, real headsets, the phone demo) is **not scored**. It is
material for the slides, the video and the final demo (27 Nov) to show depth.

The honest bar for accuracy: an IMU-only estimate in a small room is hard to make better than "stand still". Beating
stand still on held-out EuRoC and TUM-VI is the target; Debadrita's entry does it, by about 3 %.

## Two lines of work in the repo

| Line | Branches / PRs | Built on | State |
|---|---|---|---|
| **Scored entry** | `track3-entry` (#12), with Debadrita's #22 and Mayank's #23 (and #21, superseded by #23) merged in | current `main` | **Aligned.** One self-contained `src/main.py` (stdlib + numpy for the network), no dependency on `pathfinder/`. 23/24 contract tests pass |
| **Demo / innovation** | #4 → #5 → #6 → #7 → #8 → #9 → #11 → `final-benchmark`, plus Mayank's #20 on top of #11 | `main` from before Nikhil's #10 | 19 commits behind `main`. Conflicts with #10 in `pathfinder/filter.py`, `parity.py`, `pipeline.py`, `README.md`. Needs one deliberate reconcile; does not block the scored entry |

`main` = Nikhil's #10 (estimator hardening, 95 tests).

## Gap plan from the brief: status

| Gap | Status |
|---|---|
| 5 s prediction after IMU death (20 pts) | Done: dead reckoning from the last trusted state; `--extend-blind` when the file just ends. Held-out blind 5 s: 2.36 m with the network vs 2.38 m stand still (Debadrita) |
| No ground truth, unknown gravity | Done: gravity and gyro bias from the data itself; Mayank's test with the IMU rotated 90° passes |
| Exact I/O, one pose per row, through dropouts | Done: 23 contract tests (NaN, inf, frozen, gaps, duplicates, truncated files, header variants) pass |
| ≤ 20 ms per second of IMU | Done: 1.6-2.9 ms in Python |
| ≤ 8 MB peak RAM | Done with `src/main.cpp`: **2.2 MB** peak, 4 ms per second of IMU with the network; same output as `main.py` (exact without the network, within 1 µm with it); all 24 contract tests pass (`PF_ENTRY=cpp pytest tests/stress/test_main_contract.py`). Python stays as the readable reference (28 MB) |
| EuRoC data | Done on Mayank's and Debadrita's machines (all 11); still fetching on Rehaan's |
| Their degradation script | Stand-in `tools/degrade_imu.py`; organisers' script not received |
| Their scoring | Done: `tools/score_track3.py` and Mayank's `evo`-based `tests/stress/track3_eval.py` |
| **Packaging** | **Not started**: Dockerfile, `team_name.zip` (trajectory + `metadata.json` + `src/` + README), PPT `SRMIST_CaliforniaBurrito`, video ≤ 3 min, public/shared repo, Google Form |

## Open issues

#13-#19 (Mayank's stress findings) are in the streaming tracker and live server, the demo line, not the scored
entry. Fix them when that line is reconciled with `main`.

## Next, in order

1. Done: `src/main.cpp`. The network is now the default in both entries (`--model none` = old filter); with it the
   C++ entry beats stand still on MH_05 + room3 (ATE 3.81 vs 4.04 m, checkpoints 3.43 vs 3.68 m).
2. Dockerfile, bundle script, `src/README.md`, `metadata.json`.
3. PPT and video.
4. Reconcile the demo line with `main`; fix #13-#19.
