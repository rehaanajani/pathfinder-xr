# Zero-phase pull back to the start: tested, **not kept**

**Idea** (from the competitor scoreboard, `competitors.md`): both teams ahead of us smooth offline. ZeroDrift removes
slow position drift with a zero-phase filter whose cut-off shortens as the damage grows; vortexxx runs an RTS smoother
with a workspace state. Our pull back to the start (`--pos-tau 3`) is the causal version of the same thing: a
first-order high-pass of position, so it lags. Here it is replaced by a forward-backward (zero-phase) first-order
high-pass, with a time constant that is constant or follows the organisers' damage ramp, `τ(t) = τ0 + (τ1 − τ0)(t/T)^γ`.

**Method** (`tools/exp_zero_phase.py`): `main.py` runs with the pull off (`--pos-tau 0 --pos-tau-nonet 0`) and the
position is filtered afterwards. Applying the shipped causal filter that way reproduces the entry exactly (rows
"entry" and "causal 3 s (= entry)" below), so the comparison is like for like. Inputs: the organisers' script,
accel-first, from `tools/competitor_scoreboard.py`. Settings were chosen on the 11 development recordings at seed 42
only; the frozen setting (48 → 3 s, γ = 1.7, the script's own exponent, best blind on the flat part of the curve) was
then checked once on the 6 held-out recordings and on seeds 1, 7 and 2026. Ours here is `main` @ 4122d58, before
rotor drag.

**Result: dropped.** It improves every mean, and a lot: on all 17 at seed 42, ATE 2.315 → 1.873 m, checkpoints
2.129 → 1.692, dropouts 2.131 → 1.716, drift 0.573 → 0.507, blind 2.236 → 2.214; held-out ATE 2.980 → 2.499. But it
breaks two acceptance rules:

1. It is **worse than standing still on three held-out recordings**, on every seed, for ATE, checkpoints and
   dropouts: TUM-VI room5 (0.91 → about 1.3 m, standing still 0.93), room6 (0.87 → about 1.35, standing still
   1.00) and V1_03 (1.32 → about 1.6, standing still 1.58). The development TUM-VI rooms 1-4 had improved, so the
   failure does not show on development data. The long time constant lets slow drift through where the raw,
   un-pulled position wanders far; the causal 3 s pull hides that.
2. **Blind gets slightly worse on seed 7** (2.231 → 2.245 m).

Following the rules, nothing was re-tuned after the held-out check. What it does show: the mean gain from offline
smoothing is large (about 0.4 m ATE), but a fixed offline filter of our current position is not robust enough. A
smoother that also models the position (an RTS smoother with a workspace state, as vortexxx does) is the next thing to
try, chosen on development data and again checked once.

### Development sweep (seed 42, 11 development recordings; mean metres)

| setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s |
|---|---:|---:|---:|---:|---:|
| entry (causal 3 s) | 1.952 | 1.751 | 2.276 | 1.760 | 0.486 |
| causal 3 s (= entry) | 1.952 | 1.751 | 2.276 | 1.760 | 0.486 |
| no pull | 1.744 | 1.407 | 3.411 | 1.540 | 0.438 |
| zero-phase 1.5 s | 2.095 | 1.953 | 2.318 | 1.906 | 0.552 |
| zero-phase 3 s | 1.955 | 1.801 | 2.307 | 1.772 | 0.478 |
| zero-phase 6 s | 1.819 | 1.646 | 2.290 | 1.645 | 0.438 |
| zero-phase 12 s | 1.699 | 1.509 | 2.280 | 1.528 | 0.424 |
| zero-phase 24 s | 1.609 | 1.404 | 2.345 | 1.434 | 0.421 |
| zero-phase 48 s | 1.597 | 1.359 | 2.609 | 1.411 | 0.424 |
| zero-phase 96 s | 1.654 | 1.371 | 2.985 | 1.461 | 0.430 |
| zero-phase 24→3 s, γ 1.7 | 1.622 | 1.427 | 2.283 | 1.455 | 0.423 |
| zero-phase 24→6 s, γ 1.7 | 1.616 | 1.420 | 2.276 | 1.448 | 0.422 |
| zero-phase 24→12 s, γ 1.7 | 1.609 | 1.411 | 2.286 | 1.439 | 0.421 |
| zero-phase 48→3 s, γ 1.7 | 1.531 | 1.333 | 2.257 | 1.368 | 0.421 |
| zero-phase 48→6 s, γ 1.7 | 1.533 | 1.333 | 2.275 | 1.368 | 0.421 |
| zero-phase 48→12 s, γ 1.7 | 1.540 | 1.335 | 2.328 | 1.372 | 0.421 |
| zero-phase 72→3 s, γ 1.7 | 1.506 | 1.305 | 2.268 | 1.346 | 0.422 |
| zero-phase 96→3 s, γ 1.7 | 1.505 | 1.299 | 2.306 | 1.346 | 0.423 |
| zero-phase 72→2 s, γ 1.7 | 1.504 | 1.305 | 2.255 | 1.344 | 0.422 |
| zero-phase 48→3 s, γ 1.0 | 1.556 | 1.360 | 2.272 | 1.393 | 0.422 |
| zero-phase 48→3 s, γ 3.0 | 1.525 | 1.324 | 2.268 | 1.361 | 0.421 |
| zero-phase 72→3 s, γ 3.0 | 1.517 | 1.307 | 2.332 | 1.354 | 0.422 |
| standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 |

### Check of the frozen setting (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s |
|---|---|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry (causal 3 s) | 2.980 | 2.823 | 2.161 | 2.813 | 0.731 |
| held-out 6, seed 42 | zero-phase 48→3 s, γ 1.7 | 2.499 | 2.349 | 2.136 | 2.353 | 0.664 |
| held-out 6, seed 42 | standing still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 |
| all 17, seed 42 | entry (causal 3 s) | 2.315 | 2.129 | 2.236 | 2.131 | 0.573 |
| all 17, seed 42 | zero-phase 48→3 s, γ 1.7 | 1.873 | 1.692 | 2.214 | 1.716 | 0.507 |
| all 17, seed 42 | standing still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 |
| all 17, seed 1 | entry (causal 3 s) | 2.315 | 2.125 | 2.221 | 2.138 | 0.572 |
| all 17, seed 1 | zero-phase 48→3 s, γ 1.7 | 1.928 | 1.730 | 2.218 | 1.756 | 0.516 |
| all 17, seed 1 | standing still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 |
| all 17, seed 7 | entry (causal 3 s) | 2.303 | 2.113 | 2.231 | 2.134 | 0.565 |
| all 17, seed 7 | zero-phase 48→3 s, γ 1.7 | 1.845 | 1.666 | 2.245 | 1.694 | 0.495 |
| all 17, seed 7 | standing still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 |
| all 17, seed 2026 | entry (causal 3 s) | 2.305 | 2.113 | 2.221 | 2.144 | 0.567 |
| all 17, seed 2026 | zero-phase 48→3 s, γ 1.7 | 1.855 | 1.669 | 2.202 | 1.705 | 0.502 |
| all 17, seed 2026 | standing still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 |

### Recordings where the frozen setting is not better than standing still

| inputs | recording | metric | frozen | entry | standing still |
|---|---|---|---:|---:|---:|
| seed 42 | euroc:V1_03_difficult | ATE | 1.62 | 1.32 | 1.58 |
| seed 42 | euroc:V1_03_difficult | checkpoints | 1.57 | 1.21 | 1.50 |
| seed 42 | euroc:V1_03_difficult | dropouts | 1.47 | 1.20 | 1.35 |
| seed 42 | tumvi:room5 | ATE | 1.29 | 0.91 | 0.93 |
| seed 42 | tumvi:room5 | checkpoints | 1.30 | 0.86 | 0.89 |
| seed 42 | tumvi:room5 | dropouts | 1.20 | 0.85 | 0.91 |
| seed 42 | tumvi:room6 | ATE | 1.34 | 0.87 | 1.00 |
| seed 42 | tumvi:room6 | checkpoints | 1.35 | 1.02 | 1.08 |
| seed 42 | tumvi:room6 | dropouts | 1.14 | 0.79 | 0.96 |
| seed 1 | euroc:V1_03_difficult | ATE | 1.60 | 1.32 | 1.58 |
| seed 1 | euroc:V1_03_difficult | checkpoints | 1.56 | 1.23 | 1.50 |
| seed 1 | euroc:V1_03_difficult | dropouts | 1.47 | 1.21 | 1.41 |
| seed 1 | tumvi:room5 | ATE | 1.33 | 0.91 | 0.93 |
| seed 1 | tumvi:room5 | checkpoints | 1.34 | 0.84 | 0.89 |
| seed 1 | tumvi:room5 | dropouts | 1.23 | 0.82 | 0.90 |
| seed 1 | tumvi:room6 | ATE | 1.32 | 0.88 | 1.00 |
| seed 1 | tumvi:room6 | checkpoints | 1.35 | 1.03 | 1.08 |
| seed 1 | tumvi:room6 | dropouts | 1.16 | 0.80 | 0.99 |
| seed 7 | euroc:V1_03_difficult | ATE | 1.66 | 1.35 | 1.58 |
| seed 7 | euroc:V1_03_difficult | checkpoints | 1.62 | 1.23 | 1.50 |
| seed 7 | euroc:V1_03_difficult | dropouts | 1.56 | 1.26 | 1.42 |
| seed 7 | tumvi:room5 | ATE | 1.26 | 0.89 | 0.93 |
| seed 7 | tumvi:room5 | checkpoints | 1.31 | 0.85 | 0.89 |
| seed 7 | tumvi:room5 | dropouts | 1.17 | 0.83 | 0.90 |
| seed 7 | tumvi:room6 | ATE | 1.37 | 0.88 | 1.00 |
| seed 7 | tumvi:room6 | checkpoints | 1.41 | 1.02 | 1.08 |
| seed 7 | tumvi:room6 | dropouts | 1.25 | 0.82 | 0.98 |
| seed 2026 | euroc:V1_03_difficult | ATE | 1.60 | 1.32 | 1.58 |
| seed 2026 | euroc:V1_03_difficult | checkpoints | 1.53 | 1.22 | 1.50 |
| seed 2026 | euroc:V1_03_difficult | dropouts | 1.39 | 1.20 | 1.34 |
| seed 2026 | tumvi:room5 | ATE | 1.29 | 0.90 | 0.93 |
| seed 2026 | tumvi:room5 | checkpoints | 1.30 | 0.82 | 0.89 |
| seed 2026 | tumvi:room5 | dropouts | 1.21 | 0.84 | 0.91 |
| seed 2026 | tumvi:room6 | ATE | 1.34 | 0.88 | 1.00 |
| seed 2026 | tumvi:room6 | checkpoints | 1.39 | 1.00 | 1.08 |
| seed 2026 | tumvi:room6 | dropouts | 1.12 | 0.82 | 0.99 |


## Reproduce

```bash
python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>
# plus seed 2026: the same with SETS = {"s2026_accel_first": (2026, "accel_first")}
python tools/exp_zero_phase.py sweep
python tools/exp_zero_phase.py check
python tools/exp_zero_phase.py tables
```
