# Bridging dropouts with interpolated readings, on top of rotor drag: **kept** (`--bridge-max 0.5`, default)

**Idea** (from the competitor scoreboard, `competitors.md`: vortexxx integrates the gyro trapezoidally across
dropouts, ZeroDrift sub-steps missing rows with interpolated rate and force; our own implementation). When rows are
missing, the filter dead-reckoned the gap with the last turn rate decaying (τ = 0.5 s), and the first reading after
the gap never contributed to it. On TUM-VI room5 our tilt error reached 9° at the 95th percentile (vortexxx 2°).

**Change** (`PathFinder.step` in `src/main.py`, ported to `src/main.cpp`): when the rows between two trusted readings
are missing and the gap is at most `--bridge-max` seconds, the missing rows are filled with readings interpolated
linearly between those two, each through the normal update. Gaps next to an untrusted row, longer gaps and the blind
tail are handled as before. The organisers' bursts are 5-50 rows (at most 0.25 s at 200 Hz); 0.3, 0.5 and 1 s give
identical results, and 0.5 s covers the bursts with margin (and 100 Hz input) without interpolating across a 2 s
signal loss. This was first measured on `main` before rotor drag (`gap_bridge.md`); this page re-measures it on top
of rotor drag (`85b3a30`), which is what will ship.

**Method** (`tools/exp_option.py`): organisers' script, accel-first; development recordings at seed 42 for the
choice, then one check on the 6 held-out recordings (network-training holdouts that also informed earlier filter
settings, so not untouched tests) and on all 17 at seeds 1, 7 and 2026. "entry" = rotor drag + bridge;
"rotor drag, no bridge" = `--bridge-max 0`.

**Result: kept.** Every scored metric improves on every input set:

* all 17, seed 42: ATE 2.027 → 1.961 m, checkpoints 1.896 → 1.842, **blind 1.855 → 1.752**, dropouts 1.865 → 1.800,
  drift 0.520 → 0.468;
* held-out 6: ATE 2.525 → 2.471, checkpoints 2.452 → 2.389, blind 1.758 → 1.688, dropouts 2.397 → 2.336;
* seeds 1 / 7 / 2026 (all 17): ATE 2.038 / 2.002 / 2.007 → 1.946 / 1.954 / 1.945; blind 1.876 / 1.916 / 1.867 →
  1.805 / 1.850 / 1.690;
* better than standing still on all 17 at every seed for ATE, checkpoints, dropouts and drift;
* the largest gains: Machine Hall blind (3.50 → 3.13 m at seed 42) and TUM-VI ATE (0.75 → 0.69 m). The one group
  that is worse: held-out TUM-VI blind (0.51 → 0.57 m, room5/room6).

**Resources** (`tools/measure_python_budget.py`, 5 recordings, this machine): algorithm heap
1.47 MiB (limit 3), 5.87 MiB including compiling `main.py`
(limit 7), worst CPU 5.0 ms per second of IMU (limit 10).

**C++ port**: matches `main.py` within 2e-6 m on room5, room1, MH_01, MH_04 and V1_03; rotor drag alone already
differs by 2e-6 m on MH_01 and MH_04 (last digit of the 6-decimal output), and the bridge does not make it larger.

**Tests**: `pytest tests/stress` 60 passed, 2 skipped; `PF_ENTRY=cpp` 62 passed.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.684 | 1.543 | 1.786 | 1.507 | 0.374 | 3.55 / 3.44 | 1.38 / 1.70 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | rotor drag, no bridge | 1.756 | 1.592 | 1.908 | 1.575 | 0.441 | 3.65 / 3.87 | 1.41 / 1.70 | 0.68 / 0.64 | 11/11 |
| dev, seed 42 | bridge 0.3 s | 1.684 | 1.543 | 1.786 | 1.507 | 0.374 | 3.55 / 3.44 | 1.38 / 1.70 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | bridge 1 s | 1.684 | 1.543 | 1.786 | 1.507 | 0.374 | 3.55 / 3.44 | 1.38 / 1.70 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.471 | 2.389 | 1.688 | 2.336 | 0.640 | 4.83 / 2.66 | 1.69 / 1.83 | 0.89 / 0.57 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | rotor drag, no bridge | 2.525 | 2.452 | 1.758 | 2.397 | 0.664 | 5.04 / 2.95 | 1.64 / 1.82 | 0.89 / 0.51 | 6/6 |
| all 17, seed 42 | entry | 1.961 | 1.842 | 1.752 | 1.800 | 0.468 | 4.07 / 3.13 | 1.48 / 1.74 | 0.69 / 0.61 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | rotor drag, no bridge | 2.027 | 1.896 | 1.855 | 1.865 | 0.520 | 4.21 / 3.50 | 1.49 / 1.74 | 0.75 / 0.60 | 17/17 |
| all 17, seed 1 | entry | 1.946 | 1.827 | 1.805 | 1.785 | 0.458 | 4.01 / 3.23 | 1.49 / 1.83 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | rotor drag, no bridge | 2.038 | 1.911 | 1.876 | 1.878 | 0.521 | 4.22 / 3.51 | 1.51 / 1.79 | 0.75 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 1.954 | 1.832 | 1.850 | 1.789 | 0.456 | 4.01 / 3.35 | 1.52 / 1.84 | 0.68 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | rotor drag, no bridge | 2.002 | 1.874 | 1.916 | 1.848 | 0.509 | 4.10 / 3.65 | 1.51 / 1.77 | 0.75 / 0.62 | 17/17 |
| all 17, seed 2026 | entry | 1.945 | 1.833 | 1.690 | 1.798 | 0.456 | 4.00 / 3.07 | 1.49 / 1.63 | 0.68 / 0.60 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | rotor drag, no bridge | 2.007 | 1.882 | 1.867 | 1.860 | 0.512 | 4.13 / 3.52 | 1.49 / 1.73 | 0.75 / 0.63 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): none


## Reproduce

```bash
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>  # + seed 2026
python tools/exp_option.py --name gap_bridge_rotor sweep --variants '{"rotor drag, no bridge": ["--bridge-max", "0"], "bridge 0.3 s": ["--bridge-max", "0.3"], "bridge 1 s": ["--bridge-max", "1.0"]}'
python tools/exp_option.py --name gap_bridge_rotor check --label "rotor drag, no bridge" --variant '["--bridge-max", "0"]'
```

`tools/exp_option.py` reads the damaged inputs that `tools/competitor_scoreboard.py inputs` writes (PR #38).
