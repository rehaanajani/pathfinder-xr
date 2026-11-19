# Blackout start velocity from a short average: tested, **not kept**

**Idea** (brief item B6; VIO-Edge averages the velocity over the last 0.5 s before the blind window): when the IMU
becomes unusable, coast from the mean velocity over the last `--blind-v-avg` seconds of trusted motion instead of the
last value, which comes from the most damaged stretch of the file. Implementation in this branch's history (commit
"Added an optional blackout start velocity ...", reverted in the next commit). Baseline "entry" = rotor drag + dropout
bridging (`gap_bridge_rotor.md`).

**Result: not kept.** On the development set every window gives a tiny blind gain (1.786 → 1.776-1.780 m) and nothing
else moves by more than 0.002 m. The one-time check of 1 s does not hold it: held-out blind gets worse (1.688 →
1.707 m), and on all 17 recordings at seeds 42 / 1 / 7 / 2026 blind moves by 0.000 / −0.014 / −0.003 / +0.001 m.
This matches what the earlier experiments showed: the blind 5 s error is set by the position at the moment of
failure, not by the velocity the coast starts from.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.684 | 1.543 | 1.786 | 1.507 | 0.374 | 3.55 / 3.44 | 1.38 / 1.70 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | v-avg 0.25 s | 1.683 | 1.543 | 1.780 | 1.505 | 0.374 | 3.55 / 3.44 | 1.37 / 1.68 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | v-avg 0.5 s | 1.683 | 1.543 | 1.780 | 1.506 | 0.375 | 3.55 / 3.45 | 1.37 / 1.68 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | v-avg 1 s | 1.683 | 1.542 | 1.776 | 1.506 | 0.375 | 3.55 / 3.44 | 1.37 / 1.67 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | v-avg 2 s | 1.682 | 1.542 | 1.778 | 1.505 | 0.375 | 3.55 / 3.44 | 1.37 / 1.68 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.471 | 2.389 | 1.688 | 2.336 | 0.640 | 4.83 / 2.66 | 1.69 / 1.83 | 0.89 / 0.57 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | v-avg 1 s | 2.472 | 2.387 | 1.707 | 2.337 | 0.640 | 4.83 / 2.66 | 1.69 / 1.89 | 0.89 / 0.57 | 6/6 |
| all 17, seed 42 | entry | 1.961 | 1.842 | 1.752 | 1.800 | 0.468 | 4.07 / 3.13 | 1.48 / 1.74 | 0.69 / 0.61 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | v-avg 1 s | 1.961 | 1.840 | 1.752 | 1.799 | 0.468 | 4.07 / 3.13 | 1.48 / 1.75 | 0.69 / 0.61 | 17/17 |
| all 17, seed 1 | entry | 1.946 | 1.827 | 1.805 | 1.785 | 0.458 | 4.01 / 3.23 | 1.49 / 1.83 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | v-avg 1 s | 1.946 | 1.828 | 1.791 | 1.784 | 0.459 | 4.01 / 3.22 | 1.49 / 1.80 | 0.68 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 1.954 | 1.832 | 1.850 | 1.789 | 0.456 | 4.01 / 3.35 | 1.52 / 1.84 | 0.68 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | v-avg 1 s | 1.954 | 1.833 | 1.847 | 1.789 | 0.458 | 4.01 / 3.36 | 1.52 / 1.83 | 0.67 / 0.60 | 17/17 |
| all 17, seed 2026 | entry | 1.945 | 1.833 | 1.690 | 1.798 | 0.456 | 4.00 / 3.07 | 1.49 / 1.63 | 0.68 / 0.60 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | v-avg 1 s | 1.945 | 1.833 | 1.691 | 1.797 | 0.458 | 4.00 / 3.05 | 1.49 / 1.65 | 0.68 / 0.60 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s7_accel_first euroc:V1_03_difficult checkpoints_m 1.52 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.50 vs 1.42


## Reproduce

```bash
python tools/exp_option.py --name blackout_velocity sweep --variants '{"v-avg 0.25 s": ["--blind-v-avg", "0.25"], "v-avg 0.5 s": ["--blind-v-avg", "0.5"], "v-avg 1 s": ["--blind-v-avg", "1.0"], "v-avg 2 s": ["--blind-v-avg", "2.0"]}'
python tools/exp_option.py --name blackout_velocity check --label "v-avg 1 s" --variant '["--blind-v-avg", "1.0"]'
```
(on the commit that adds the option)
