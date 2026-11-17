# Coupling sweep, 17 Nov 2026

How the default position coupling was chosen. Organisers' degradation script (seed 42, accel-first input), C++
entry, all 11 EuRoC + 6 TUM-VI recordings; mean ATE in metres (lower is better). The 8 Monado headset recordings
use the gravity-aligned network and its own coupling (mount guard): 0.57 m for every setting, against 0.84 m
standing still.

| Setting | EuRoC (11) | TUM-VI (6) | Recordings better than standing still (ATE) |
|---|---:|---:|---:|
| standing still | 3.45 | 1.09 |  |
| **chosen**: pull 3 s, network steps not trusted | 3.17 | 0.75 | 17 / 17 |
| pull 5 s | 3.02 | 0.74 | 16 / 17 |
| pull 7 s | 2.90 | 0.75 | 15 / 17 |
| pull 7 s + firm pull after 1 s blind | 2.90 | 0.75 | 15 / 17 |
| pull 10 s + firm blind pull | 2.77 | 0.79 | 15 / 17 |
| pull 7 s + network steps x0.3 | 2.74 | 0.80 | 15 / 17 |
| pull 7 s + network steps (ref 0.1) | 2.56 | 0.92 | 15 / 17 |
| pull 7 s, network steps (ref 0.3), firm blind pull | 2.35 | 1.05 | 15 / 17 |

The chosen setting is the one that beats standing still on all 17 recordings. Settings that score higher on
EuRoC do so by giving up TUM-VI rooms 5 and 6, which were not used to tune them, so they were not taken: the
default is the setting that generalises.
Reproduce: `tools/eval_organiser.py --script <degrade_imu.py> --data ... -- --pos-tau 7 ...` per row.
