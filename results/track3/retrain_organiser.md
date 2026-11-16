# Retraining the network on the organisers' damage, 17 Nov 2026

Question: does the velocity network improve if it is trained on copies damaged by the organisers' own
`degrade_imu.py` (spikes, stuck runs, dropped rows and bursts, duplicates, ramped noise and drift)?

Setup: `tools/train_airio.py` with `PFX_ORG_SCRIPT` set adds 6 organiser-damaged copies per training recording
(seeds 1000-1005, final severity 0.7-1.6; their scoring seed is 42) to the usual 13. Choice made on validation only
(MH_04, V1_03, room5), organisers' script with their seed, C++ entry, default settings. Held-out not looked at.

| Network | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| current `src/velnet_weights.bin` | 2.86 | 2.74 | 1.91 | 2.70 |
| retrained, our damage only (control, reproduces the current one) | 2.86 | 2.73 | 1.86 | 2.71 |
| our damage + organiser damage | 2.88 | 2.81 | 1.94 | 2.72 |
| organiser damage only | 2.89 | 2.81 | 1.88 | 2.73 |
| 4x wider network (164k parameters), seed 2 | 2.90 | 2.78 | 1.90 | 2.73 |
| 4x wider network, seed 3 | 2.81 | 2.72 | 1.82 | 2.63 |
| 2 s window | 2.97 | 2.91 | 2.03 | 2.79 |

The two seeds of the same wide network differ by 0.09 m, as much as any change here: no candidate is better than
the current network beyond seed noise, so the current network stays.

**Ceiling** (`tools/oracle_ceiling.py`, ground-truth velocity in place of the network, same three recordings, our
degradation seed 1): ATE 2.56 with the pull to the start at 3 s, **1.95 at 10 s**, 2.00 at 30 s, 3.07 with no pull.
A much better network would pay off (and would allow a weaker pull), but ours removes only about 35 % of the
velocity error of predicting zero; the limit is the 11 training recordings, not the kind of damage or the model size.
