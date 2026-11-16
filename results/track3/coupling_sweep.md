# Coupling sweep, 17 Nov 2026

Organisers' degradation script (seed 42, accel-first input), C++ entry, all 11 EuRoC + 6 TUM-VI recordings.
Mean ATE in metres (lower is better). The 8 Monado headset recordings are not listed: the mount guard switches them
to the gravity-aligned network and its own coupling, so every setting gives the same 0.569 m there.

| Setting | EuRoC (11) | TUM-VI (6) | Worst recording vs standing still | Recordings worse than standing still |
|---|---:|---:|---:|---|
| standing still (pull with 0.01 s) | 3.45 | 1.09 |  |  |
| **chosen**: pull 3 s, network steps not trusted | 3.17 | 0.75 | 0.98x | none |
| pull 5 s | 3.02 | 0.74 | 1.04x | room5 |
| pull 7 s | 2.90 | 0.75 | 1.10x | room5, room6 |
| pull 7 s + firm pull after 1 s blind | 2.90 | 0.75 | 1.10x | room5, room6 |
| pull 10 s + firm blind pull | 2.77 | 0.79 | 1.17x | room5, room6 |
| pull 7 s + network steps x0.3 | 2.74 | 0.80 | 1.28x | room5, room6 |
| pull 7 s + network steps (ref 0.1) | 2.56 | 0.92 | 1.64x | room5, room6 |
| tuned 17 Nov candidate: pull 7 s, network steps (ref 0.3), firm blind pull | 2.35 | 1.05 | 1.87x | room5, room6 |

Stronger settings win on EuRoC (down to 2.35 m) but lose on TUM-VI rooms 5 and 6, which were not used to tune
them. The chosen setting is the only one that is never worse than standing still, so it is the default.
Reproduce: `tools/eval_organiser.py --script <degrade_imu.py> --data ... -- --pos-tau 7 ...` per row.
