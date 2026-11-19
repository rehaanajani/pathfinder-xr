# Rotor-drag aiding, 19 Nov 2026

**Idea.** A multirotor's sideways specific force is proportional to its sideways velocity (rotor drag): in the body
frame, f_perp = k · v_perp. The accelerometer therefore measures the drone's sideways velocity directly, and in its own
frame, so the measurement needs no heading. It applies only to a flying multirotor, so `main.py` turns it on only when
the gravity measured at start-up is within 15° of the EuRoC airframe's thrust axis (in the IMU frame), and stays off for
handheld recordings (TUM-VI) and other mounts (Monado headsets).

**Fit.** Against motion capture on the EuRoC training flights (MH_01-03, V1_01-02, V2_01-02), the low-passed sideways
specific force follows the sideways body velocity (correlation -0.92 on MH_03, -0.86 / -0.88 on V1_02; weak on MH_01-02,
which start hand-carried and on the floor). k = -0.19 and -0.17 1/s on the two sideways axes. Readings far from the
current estimate get a Cauchy weight (scale 2 m/s), so carried or grounded phases are discounted.

**Coupling.** With velocity measured, the pull back to the start point can be gentler on the drone: 12 s instead of
3 s, only while drag is on. TUM-VI and Monado keep the old settings, so their results are unchanged.

**Results** (organisers' script, seed 42, accel-first; all 17 EuRoC + TUM-VI recordings; `main.py`):

| Setting | EuRoC ATE (11) | TUM-VI ATE (6) | Means: ATE / checkpoints / blind / dropouts / drift | Recordings better than standing still (same order) |
|---|---:|---:|---|---|
| before (no drag, pull 3 s everywhere) | 3.17 | 0.75 | 2.31 / 2.13 / 2.24 / 2.13 / 0.57 | 17 / 17 / 12 / 17 / 17 |
| drag only, pull 3 s | 3.16 | 0.75 | 2.31 / 2.14 / 2.18 / 2.12 / 0.57 | 17 / 17 / 14 / 17 / 17 |
| pull 7 s on the drone, no drag | 2.90 | 0.75 | 2.14 / 1.96 / 2.18 / 1.97 / 0.54 | 17 / 17 / 13 / 17 / 17 |
| pull 10 s on the drone, no drag | 2.77 | 0.75 | 2.06 / 1.89 / 2.13 / 1.90 / 0.53 | 17 / 17 / 12 / 17 / 17 |
| drag + pull 10 s | 2.78 | 0.75 | 2.06 / 1.93 / 1.91 / 1.90 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| **drag + pull 12 s (default)** | 2.73 | 0.75 | 2.03 / 1.90 / 1.85 / 1.86 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| drag + pull 15 s | 2.67 | 0.75 | 1.99 / 1.86 / 1.80 / 1.83 / 0.52 | 17 / 17 / 14 / 17 / 17 |
| drag (blend 1 s, robust 1 m/s) + pull 20 s | 2.57 | 0.75 | 1.93 / 1.78 / 1.93 / 1.78 / 0.51 | 17 / 16 / 13 / 16 / 17 |
| drag (blend 1 s, robust 1 m/s) + no pull on the drone | 2.56 | 0.75 | 1.92 / 1.67 / 2.74 / 1.75 / 0.52 | 15 / 15 / 6 / 15 / 17 |

Drag alone barely moves the score while the pull to the start stays at 3 s; the gain comes from the gentler pull on
the drone, and drag's own contribution shows in the blind 5 s (pull 10 s: 2.13 m without drag, 1.91 m with it) and in the blind win count.
Longer pulls (15-20 s) or none at all lower some means further but lose recordings to standing still, so 12 s is the
default. Standing still: ATE 2.62, checkpoints 2.45, blind 2.28, dropouts 2.42, drift 0.77 m.

**Drag and the pull, separately** (seed 42): the gentler pull on the drone gives most of the ATE and checkpoint gain;
drag gives the blind-window gain (2.11 -> 1.86 m) and two more blind wins, at a small checkpoint cost.

| Setting | EuRoC ATE (11) | TUM-VI ATE (6) | Means: ATE / checkpoints / blind / dropouts / drift | Recordings better than standing still (same order) |
|---|---:|---:|---|---|
| drag off, pull 3 s (before) | 3.17 | 0.75 | 2.31 / 2.13 / 2.24 / 2.13 / 0.57 | 17 / 17 / 12 / 17 / 17 |
| drag on, pull 3 s | 3.16 | 0.75 | 2.31 / 2.14 / 2.18 / 2.12 / 0.57 | 17 / 17 / 14 / 17 / 17 |
| drag off, pull 12 s on the drone | 2.71 | 0.75 | 2.02 / 1.85 / 2.11 / 1.86 / 0.53 | 17 / 17 / 13 / 17 / 17 |
| **drag on, pull 12 s (default)** | 2.73 | 0.75 | 2.03 / 1.90 / 1.85 / 1.86 / 0.52 | 17 / 17 / 15 / 17 / 17 |

**Blackout strategy, same state** (only the velocity decay after the IMU dies changes; drag on, pull 12 s). A trade-off
within about 2 cm: faster decay wins one more recording, slower decay lowers the mean. The default (0.7 s) is kept rather
than tuned further on the same recordings.

| Setting | EuRoC ATE (11) | TUM-VI ATE (6) | Means: ATE / checkpoints / blind / dropouts / drift | Recordings better than standing still (same order) |
|---|---:|---:|---|---|
| decay 0.05 s (near freeze) | 2.73 | 0.75 | 2.03 / 1.90 / 1.88 / 1.87 / 0.52 | 17 / 17 / 16 / 17 / 17 |
| decay 0.2 s | 2.73 | 0.75 | 2.03 / 1.90 / 1.87 / 1.87 / 0.52 | 17 / 17 / 16 / 17 / 17 |
| decay 0.7 s (default) | 2.73 | 0.75 | 2.03 / 1.90 / 1.85 / 1.86 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| decay 1.5 s | 2.72 | 0.75 | 2.03 / 1.90 / 1.84 / 1.86 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| decay 3 s | 2.72 | 0.75 | 2.03 / 1.90 / 1.84 / 1.86 / 0.52 | 17 / 17 / 14 / 17 / 17 |

**Fresh corruption seeds.** The organisers' script with only its fixed seed changed (1, 7, 2026), which none of these
settings saw. The gain holds on every seed: ATE about 0.3 m lower, blind about 0.35 m lower, and better than standing
still on all 17 recordings for ATE, checkpoints, dropouts and drift. Each seed is compared with standing still on the
same seed.

| Setting | EuRoC ATE (11) | TUM-VI ATE (6) | Means: ATE / checkpoints / blind / dropouts / drift | Recordings better than standing still (same order) |
|---|---:|---:|---|---|
| seed 1, before | 3.17 | 0.75 | 2.31 / 2.12 / 2.22 / 2.14 / 0.57 | 17 / 17 / 14 / 17 / 17 |
| seed 1, with drag | 2.74 | 0.75 | 2.04 / 1.91 / 1.88 / 1.88 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| seed 7, before | 3.15 | 0.75 | 2.30 / 2.11 / 2.23 / 2.13 / 0.56 | 17 / 17 / 12 / 17 / 17 |
| seed 7, with drag | 2.69 | 0.75 | 2.00 / 1.87 / 1.92 / 1.85 / 0.51 | 17 / 17 / 13 / 17 / 17 |
| seed 2026, before | 3.15 | 0.75 | 2.30 / 2.11 / 2.22 / 2.14 / 0.57 | 17 / 17 / 13 / 17 / 17 |
| seed 2026, with drag | 2.69 | 0.75 | 2.01 / 1.88 / 1.87 / 1.86 / 0.51 | 17 / 17 / 14 / 17 / 17 |

**Tried and dropped: offline position smoothing** (zero-phase moving average of the position, bounded memory):

| Setting | EuRoC ATE (11) | TUM-VI ATE (6) | Means: ATE / checkpoints / blind / dropouts / drift | Recordings better than standing still (same order) |
|---|---:|---:|---|---|
| drag + pull 12 s (default) | 2.73 | 0.75 | 2.03 / 1.90 / 1.85 / 1.86 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| + smoothing window 0.4 s | 2.73 | 0.75 | 2.03 / 1.90 / 1.86 / 1.87 / 0.52 | 17 / 17 / 15 / 17 / 17 |
| + 1 s | 2.73 | 0.75 | 2.03 / 1.90 / 1.86 / 1.87 / 0.53 | 17 / 17 / 15 / 17 / 17 |
| + 2 s | 2.73 | 0.77 | 2.04 / 1.91 / 1.86 / 1.88 / 0.55 | 17 / 17 / 15 / 17 / 17 |
| + 4 s | 2.76 | 0.85 | 2.08 / 1.94 / 1.87 / 1.93 / 0.61 | 17 / 16 / 15 / 17 / 17 |

It does not help: the remaining error is slow drift, which averaging cannot remove, and longer windows hurt TUM-VI.

Cost: about 10 % more CPU when measured side by side; heap unchanged. Final figures, measured on an idle machine: 4.0 ms per
second of IMU ([`python_budget.json`](python_budget.json)). The C++ port has
the same drag code and gives the same scores within 1e-6 m.
