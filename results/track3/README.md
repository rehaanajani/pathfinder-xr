# Track 3 evidence

Raw results behind every number in the main README. Default settings (AirIO network + mount guard).

## Current results (organisers' degradation script)

All runs use the organisers' `degrade_imu.py` with their seed and their documented accel-first columns
(`tools/eval_organiser.py`), scored by `tools/score_track3.py`. Every run wrote exactly one pose per input row. The
main table comes from `main.py`; the C++ port gives the same scores (file below).

| File | What | Headline |
|---|---|---|
| [`per_recording.md`](per_recording.md) | all 11 EuRoC + 6 TUM-VI recordings, every metric, Path Finder (`main.py`) vs standing still, per recording | better on 17/17 for ATE, checkpoints, dropouts and drift; blind 13/17 |
| [`organiser_per_recording.json`](organiser_per_recording.json) | the raw numbers behind that table, plus the 8 Monado headset recordings | means: ATE 1.96 vs 2.62 m, checkpoints 1.84 vs 2.45, blind 1.75 vs 2.28, dropouts 1.80 vs 2.42, drift 0.47 vs 0.77; Monado 0.57 vs 0.83 m (8/8, measured before dropout bridging) |
| [`organiser_per_recording_cpp.json`](organiser_per_recording_cpp.json) | the 17 recordings scored with the C++ port (`--entry cpp`) | identical to the file above within 1e-6 m on every metric |
| [`python_budget.json`](python_budget.json) | memory and CPU of `src/main.py`, 5 recordings (`tools/measure_python_budget.py`) | worst: algorithm 1.47 MiB; 5.87 MiB counting everything after numpy, including compiling `main.py`; 4.9 ms per second of IMU |
| [`gap_bridge_rotor.md`](gap_bridge_rotor.md) | bridging dropouts with interpolated readings, on top of rotor drag (the last change kept) | ATE 2.03 -> 1.96 m, blind 1.85 -> 1.75 m; better on every scored metric at seeds 42, 1, 7, 2026 |
| [`competitors.md`](competitors.md) | every public Track 3 entry run on our damaged files and scored the same way | second on blind 5 s, third on checkpoints and ATE |
| [`rotor_drag.md`](rotor_drag.md) | rotor-drag aiding and the gentler pull on the drone: fit, settings sweep, the smoother that did not help | ATE 2.31 -> 2.03 m, blind 2.24 -> 1.85 m; TUM-VI and Monado unchanged |
| [`coupling_sweep.md`](coupling_sweep.md) | how the position coupling was chosen before rotor drag: 9 settings × 25 recordings | the setting that beats standing still on all 17 |
| [`retrain_organiser.md`](retrain_organiser.md) | retraining on the organisers' damage, a 4× larger network, a 2 s window | no gain beyond seed noise; the current network stays |
| [`mount_guard_check.json`](mount_guard_check.json) | mount guard on vs off on 16 recordings, and the same files rotated 90° (`tools/check_mount_guard.py`) | normal: 16/16 quiet and byte-identical; rotated: 14/16 switch networks |

**What the metrics measure** (`tools/score_track3.py`). ATE: RMSE after one SE(3) (Umeyama) alignment of the whole
trajectory, as `evo_ape -a`. Checkpoints, blind 5 s, dropouts and drift are scored after an alignment fitted on the
mission part only (before the IMU dies), so the blind 5 s gets no help from the alignment. Checkpoints: 10 evenly
spaced mission times (the organisers' own checkpoints are hidden). Dropouts: each gap plus the second after it; the
organisers' script removes rows, so there are no poses inside a gap and this mostly scores the recovery second.
Drift: error of the position change over 1 s, position only (not evo's pose RPE, which includes rotation).
Standing still: the start position at every row; after alignment it sits at the centre of the path, which makes it a
strong baseline in small spaces.

## Earlier results (16 Nov)

Kept for the record. These used our own stand-in for the organisers' script (`tools/degrade_imu.py`, seeds 1-3), or
the C++ port, before the organisers' script was available and before the final settings.

| File | What | Headline (ATE / checkpoints / blind 5 s / dropouts, m) |
|---|---|---|
| `final_heldout_entry.json` | held-out MH_05, V2_03, room6, seeds 1-3, stand-in damage | 3.083 / 2.861 / 2.343 / 3.147 |
| `final_heldout_still.json` | same, standing still | 3.284 / 3.034 / 2.383 / 3.267 |
| `final_heldout_oldnet.json` | same, the earlier gravity-aligned network | 3.194 / 2.925 / 2.364 / 3.272 |
| `final_all_entry.json`, `final_all_still.json` | 16 recordings, seeds 1-2, stand-in damage | 2.440 / 2.233 / 2.340 / 2.389 vs 2.714 / 2.545 / 2.382 / 2.461 |
| `final_org_accel_first_*.json`, `final_org_euroc_*.json` | first runs on the organisers' script, 5 recordings, both column orders | 3.395 / 3.216 / 2.468 / 3.206 vs 3.663 / 3.439 / 2.511 / 3.442 standing still |
| `budget.json` | the C++ port's peak RSS and CPU (`tools/measure_budget.py`) | 2.28 MB, 1.96 ms per second of IMU |

## Reproduce

Needs the datasets (`python tools/fetch_euroc.py`, `python tools/fetch_tumvi.py`) and the organisers'
`degrade_imu.py`:

```bash
python tools/eval_organiser.py --script degrade_imu.py --data euroc:MH_01_easy ... tumvi:room6
python tools/measure_python_budget.py --script degrade_imu.py --data euroc:MH_04_difficult euroc:V1_03_difficult euroc:MH_05_difficult euroc:V2_03_difficult tumvi:room6
python tools/check_mount_guard.py --data euroc:MH_01_easy ... tumvi:room6
```
