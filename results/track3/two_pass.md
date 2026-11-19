# Offline two-pass start: tested, **not kept** (near miss)

**Idea** (brief item B5a; ZeroDrift runs its filter twice): the rules allow reading the whole file. A silent first pass
over the first `--two-pass` seconds learns the gyro bias (and optionally the slow acceleration residual); the real
pass then starts from those values instead of learning them during its own first seconds. Implementation in this
branch's history (commit "Added an optional offline two-pass start ...", reverted in the next commit). Baseline
"entry" = rotor drag + dropout bridging (`gap_bridge_rotor.md`).

**Result: not kept, though every mean improves.** Development: every metric slightly better with the bias alone
(ATE 1.684 → 1.675, checkpoints 1.543 → 1.538, blind 1.786 → 1.778, dropouts 1.507 → 1.501); 10, 30 and 60 s
are the same, the residual adds nothing. The one-time check of 10 s improves every mean on every set (held-out ATE
2.471 → 2.452 and blind 1.688 → 1.641; all 17 at seeds 42 / 1 / 7 / 2026: ATE −0.012 to −0.014 m, blind −0.022 to
−0.031 m). But at seed 7 it makes **V1_03 worse than standing still** on checkpoints (1.54 vs 1.50 m) and dropouts
(1.51 vs 1.42 m), which the entry is not, so it fails the "better than standing still on all 17" rule. CPU cost: the
first 10 s are run twice (a few percent).

The gain comes from the Vicon rooms (ATE 1.48 → 1.44, blind 1.74 → 1.68 at seed 42). If the team decides the
all-17 rule can bend for a single recording at a single seed, this is the next candidate; it is otherwise ready (one
option, ~20 lines, C++ port not written).


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.684 | 1.543 | 1.786 | 1.507 | 0.374 | 3.55 / 3.44 | 1.38 / 1.70 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | two-pass 10 s, bias | 1.675 | 1.538 | 1.778 | 1.501 | 0.368 | 3.55 / 3.44 | 1.35 / 1.67 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | two-pass 30 s, bias | 1.675 | 1.538 | 1.778 | 1.501 | 0.368 | 3.55 / 3.44 | 1.35 / 1.67 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | two-pass 30 s, bias + residual | 1.676 | 1.539 | 1.778 | 1.501 | 0.368 | 3.55 / 3.44 | 1.36 / 1.67 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | two-pass 60 s, bias + residual | 1.676 | 1.539 | 1.779 | 1.502 | 0.368 | 3.56 / 3.44 | 1.35 / 1.68 | 0.59 / 0.63 | 11/11 |

### Check of the frozen variant (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.471 | 2.389 | 1.688 | 2.336 | 0.640 | 4.83 / 2.66 | 1.69 / 1.83 | 0.89 / 0.57 | 6/6 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 | 6.84 / 4.12 | 1.82 / 1.98 | 0.97 / 0.51 | 6/6 |
| held-out 6, seed 42 | two-pass 10 s, bias | 2.452 | 2.369 | 1.641 | 2.317 | 0.635 | 4.83 / 2.66 | 1.63 / 1.69 | 0.89 / 0.58 | 6/6 |
| all 17, seed 42 | entry | 1.961 | 1.842 | 1.752 | 1.800 | 0.468 | 4.07 / 3.13 | 1.48 / 1.74 | 0.69 / 0.61 | 17/17 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 | 5.25 / 4.46 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 42 | two-pass 10 s, bias | 1.949 | 1.832 | 1.730 | 1.789 | 0.462 | 4.07 / 3.13 | 1.44 / 1.68 | 0.69 / 0.62 | 17/17 |
| all 17, seed 1 | entry | 1.946 | 1.827 | 1.805 | 1.785 | 0.458 | 4.01 / 3.23 | 1.49 / 1.83 | 0.68 / 0.59 | 17/17 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 | 5.25 / 4.44 | 1.95 / 2.14 | 1.09 / 0.61 | 17/17 |
| all 17, seed 1 | two-pass 10 s, bias | 1.933 | 1.817 | 1.774 | 1.774 | 0.452 | 4.01 / 3.23 | 1.46 / 1.74 | 0.68 / 0.59 | 17/17 |
| all 17, seed 7 | entry | 1.954 | 1.832 | 1.850 | 1.789 | 0.456 | 4.01 / 3.35 | 1.52 / 1.84 | 0.68 / 0.61 | 17/17 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 7 | two-pass 10 s, bias | 1.940 | 1.820 | 1.821 | 1.778 | 0.450 | 4.01 / 3.35 | 1.48 / 1.76 | 0.67 / 0.61 | 17/17 |
| all 17, seed 2026 | entry | 1.945 | 1.833 | 1.690 | 1.798 | 0.456 | 4.00 / 3.07 | 1.49 / 1.63 | 0.68 / 0.60 | 17/17 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 | 5.25 / 4.45 | 1.95 / 2.13 | 1.09 / 0.61 | 17/17 |
| all 17, seed 2026 | two-pass 10 s, bias | 1.931 | 1.821 | 1.664 | 1.786 | 0.450 | 4.00 / 3.07 | 1.45 / 1.55 | 0.68 / 0.61 | 17/17 |

Not better than standing still (ATE, checkpoints, dropouts, drift): s7_accel_first euroc:V1_03_difficult checkpoints_m 1.54 vs 1.50; s7_accel_first euroc:V1_03_difficult dropout_m 1.51 vs 1.42


## Reproduce

```bash
python tools/exp_option.py --name two_pass sweep --variants '{"two-pass 10 s, bias": ["--two-pass", "10"], "two-pass 30 s, bias": ["--two-pass", "30"], "two-pass 30 s, bias + residual": ["--two-pass", "30", "--two-pass-res", "1"], "two-pass 60 s, bias + residual": ["--two-pass", "60", "--two-pass-res", "1"]}'
python tools/exp_option.py --name two_pass check --label "two-pass 10 s, bias" --variant '["--two-pass", "10"]'
```
(on the commit that adds the option)
