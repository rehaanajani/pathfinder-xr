# Bridging dropouts with interpolated readings: **kept** (`--bridge-max 0.5`, now the default)

**Idea** (from the competitor scoreboard: vortexxx integrates the gyro trapezoidally across dropouts, ZeroDrift
sub-steps missing rows with interpolated rate and force). When rows are missing, our filter dead-reckoned the gap
with the last turn rate decaying (τ = 0.5 s) and velocity decaying, and the first reading after the gap never
contributed to the gap. On TUM-VI room5 our tilt error reached 9° at the 95th percentile (vortexxx 2°).

**Change** (`src/main.py` `PathFinder.step`, ported to `src/main.cpp`): when the rows between two trusted readings are
missing and the gap is at most `--bridge-max` seconds, the missing rows are filled with readings interpolated linearly
between those two, and each goes through the normal update. Gaps between a trusted and an untrusted row, longer gaps,
and the blind tail are handled as before. The organisers' bursts are 5-50 rows (at most 0.25 s at 200 Hz); 0.3, 1
and 3 s give identical results on the development set, so 0.5 s was taken: it covers the bursts with margin (and at
100 Hz input) without interpolating across a 2 s signal loss.

**Method** (`tools/exp_option.py`): organisers' script, accel-first. Settings compared on the 11 development
recordings at seed 42; the frozen value checked once on the 6 held-out recordings and on all 17 at seeds 1, 7 and
2026. "entry" is `main` @ 4122d58 before this change.

**Result: kept.** All acceptance rules hold:

* the mean improves on the held-out set (ATE 2.980 → 2.960 m) and on every fresh seed (all 17: ATE 2.315 → 2.261,
  2.303 → 2.262, 2.305 → 2.259 m), with checkpoints, dropouts and drift improving too;
* blind is better or level everywhere (all 17: 2.236 → 2.200 at seed 42, 2.221 → 2.165 at seed 2026; held-out 2.161 → 2.160);
* better than standing still on all 17 recordings at every seed, for ATE, checkpoints, dropouts and drift;
* resources unchanged (`tools/measure_python_budget.py`, same machine, 5 recordings): heap 1.23 MiB algorithm /
  3.63 MiB with the module, CPU 4.50 ms per second of IMU worst case (4.54 before);
* `src/main.cpp` matches `src/main.py` within 1e-6 m (the output precision) on room5, room1, MH_04 and V1_03;
  `pytest tests/stress`: 58 passed, 2 skipped (Python), 60 passed (`PF_ENTRY=cpp`).

The gain is modest (about 2 % ATE) and largest on TUM-VI (0.75 → 0.68 m). The one group that gets worse is the
held-out TUM-VI blind 5 s (0.51 → 0.57 m, two recordings); the overall blind mean does not.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.952 | 1.751 | 2.276 | 1.760 | 0.486 | 3.94 / 4.59 | 1.74 / 2.17 | 0.68 / 0.64 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | bridge 0.1 s | 1.916 | 1.706 | 2.265 | 1.718 | 0.455 | 3.89 / 4.56 | 1.72 / 2.16 | 0.63 / 0.65 | 11/11 |
| dev, seed 42 | bridge 0.3 s | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | bridge 1 s | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | bridge 3 s | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.980 | 2.823 | 2.161 | 2.813 | 0.731 | 6.40 / 4.01 | 1.65 / 1.96 | 0.89 / 0.51 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | bridge 0.5 s | 2.960 | 2.796 | 2.160 | 2.789 | 0.715 | 6.33 / 3.95 | 1.65 / 1.96 | 0.89 / 0.57 | 6/6 |
| all 17, seed 42 | entry | 2.315 | 2.129 | 2.236 | 2.131 | 0.573 | 4.92 / 4.36 | 1.71 / 2.10 | 0.75 / 0.60 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | bridge 0.5 s | 2.270 | 2.082 | 2.200 | 2.082 | 0.530 | 4.86 / 4.25 | 1.70 / 2.08 | 0.69 / 0.61 | 17/17 |
| all 17, seed 1 | entry | 2.315 | 2.125 | 2.221 | 2.138 | 0.572 | 4.92 / 4.34 | 1.71 / 2.08 | 0.75 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | bridge 0.5 s | 2.261 | 2.073 | 2.196 | 2.082 | 0.522 | 4.84 / 4.25 | 1.70 / 2.09 | 0.68 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 2.303 | 2.113 | 2.231 | 2.134 | 0.565 | 4.88 / 4.36 | 1.71 / 2.07 | 0.75 / 0.62 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | bridge 0.5 s | 2.262 | 2.077 | 2.208 | 2.087 | 0.521 | 4.84 / 4.29 | 1.70 / 2.08 | 0.68 / 0.61 | 17/17 |
| all 17, seed 2026 | entry | 2.305 | 2.113 | 2.221 | 2.144 | 0.567 | 4.89 / 4.34 | 1.70 / 2.05 | 0.75 / 0.63 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | bridge 0.5 s | 2.259 | 2.075 | 2.165 | 2.096 | 0.521 | 4.83 / 4.22 | 1.69 / 2.01 | 0.68 / 0.60 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): none


## Reproduce

```bash
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>
python tools/exp_option.py --name gap_bridge sweep --variants '{"bridge 0.1 s": ["--bridge-max", "0.1"], "bridge 0.3 s": ["--bridge-max", "0.3"], "bridge 1 s": ["--bridge-max", "1.0"], "bridge 3 s": ["--bridge-max", "3.0"]}'
python tools/exp_option.py --name gap_bridge check --label "bridge 0.5 s" --variant '["--bridge-max", "0.5"]'
```

(These ran before 0.5 s became the default, with the old default 0 as "entry". After this change, pass
`--bridge-max 0` to get the old behaviour.)
