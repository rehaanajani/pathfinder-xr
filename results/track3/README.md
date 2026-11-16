# Track 3 evidence (16 Nov 2026, final code)

Raw results behind every number in the README, `docs/TRACK3_STATUS.md` and the slides. C++ entry (`src/pathfinder`),
default settings (AirIO network + mount guard), Apple M5, degraded with `tools/degrade_imu.py` (our stand-in for the
organisers' script), scored with `tools/score_track3.py` (ATE matches `evo_ape -a` to 0.01 %, `docs/TRACK3_EVAL.md`).

| File | What | Headline |
|---|---|---|
| `final_heldout_entry.json` | held-out EuRoC MH_05, V2_03, TUM-VI room6, seeds 1-3 | ATE 3.083, checkpoints 2.861, blind 2.343, dropouts 3.147 m |
| `final_heldout_still.json` | same, standing still (`--v-max 0.0000001`) | 3.284 / 3.034 / 2.383 / 3.267 |
| `final_heldout_oldnet.json` | same, the earlier gravity-aligned network | 3.194 / 2.925 / 2.364 / 3.272 |
| `final_all_entry.json` | all 11 EuRoC + TUM-VI rooms 1-4, 6, seeds 1-2 (includes training recordings) | 2.440 / 2.233 / 2.340 / 2.389; better than standing still on ATE and checkpoints in 32/32 runs, blind 25/32, dropouts 20/32 |
| `final_all_still.json` | same, standing still | 2.714 / 2.545 / 2.382 / 2.461 |
| `budget.json` | peak RSS and CPU per second of IMU, the process's own (`tools/measure_budget.py`, 3 repeats) | worst 2.28 MB of 8, 1.96 ms of 20 |
| `mount_guard_check.json` | guard on vs off on all 16 recordings, and the same files rotated 90° about x (`tools/check_mount_guard.py`) | normal: 16/16 quiet and byte-identical; rotated: 14/16 switch (MH_01: not at rest at start-up, guard holds off by design; MH_03: rotation moves gravity only ~25°, inside the threshold) |

"Dropouts" is scored over each gap plus the second after it. With the default `--dropout-mode missing` the gap has no
rows, so it is mostly the second after the gap; standing still is scored the same way.

Reproduce (needs the datasets: `python tools/fetch_euroc.py`, `python tools/fetch_tumvi.py`):

```bash
make -C src
PF_ENTRY=cpp python tools/eval_track3.py --data euroc:MH_05_difficult euroc:V2_03_difficult tumvi:room6 --seeds 1 2 3 --tag final_heldout_entry
python tools/measure_budget.py --data euroc:MH_05_difficult euroc:V2_03_difficult tumvi:room6
python tools/check_mount_guard.py --data euroc:MH_01_easy ... tumvi:room6
```
