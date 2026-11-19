# Ensemble of three training seeds (20 Nov 2026): not kept

Idea: average three training seeds of the current network and add their disagreement to the predicted variance
(mean of the members' variances + variance of their means), to remove the seed lottery. Members: the shipped network
(`r0_s2`) and the two other seeds of the same configuration trained with it (`r0_s0`, `r0_s1`), same data.
Prototype in `main.py` (`--model a+b+c`), validation only.

Organisers' `degrade_imu.py` (their seed 42 unless stated), accel-first input, C++ entry (`tools/eval_organiser.py`;
same output as `main.py` within 1e-6 m), metres. Choices on training and validation recordings only; the held-out
recordings (MH_05, V2_03, room6) and the fresh seeds 1, 7, 2026 were not used. Raw results:
[`network/`](network/); tables generated with `python tools/network_tables.py`.

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| shipped seed (r0_s2) | 2.404 | 2.401 | 1.518 | 2.301 |
| seed r0_s0 | 2.459 | 2.438 | 1.551 | 2.351 |
| seed r0_s1 | 2.411 | 2.397 | 1.650 | 2.311 |
| ensemble of the three | 2.402 | 2.399 | 1.572 | 2.301 |

| Recording | shipped seed (r0_s2) | seed r0_s0 | seed r0_s1 | ensemble of the three |
|---|---:|---:|---:|---:|
| euroc:MH_04_difficult | 4.99 / 5.02 | 4.87 / 4.86 | 4.96 / 4.95 | 4.94 / 4.94 |
| euroc:V1_03_difficult | 1.31 / 1.33 | 1.58 / 1.55 | 1.32 / 1.30 | 1.35 / 1.36 |
| tumvi:room5 | 0.91 / 0.86 | 0.93 / 0.90 | 0.96 / 0.94 | 0.92 / 0.90 |

Per-recording cells: ATE / checkpoints.

The mean of the three members alone is ATE 2.425, checkpoints 2.412, blind 1.573, dropouts 2.321: the ensemble removes
the seed noise as expected (ATE 2.402 vs 2.425, blind equal). But the shipped seed was itself chosen as the best of
these three on MH_04 + V1_03 (with our earlier degradation), so against it the ensemble is level on ATE and worse on
blind 5 s (1.57 vs 1.52), at three times the network cost. **Not kept**; the prototype was not committed.
