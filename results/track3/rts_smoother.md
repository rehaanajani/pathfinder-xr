# Offline RTS smoother over position: tested, **not kept**

**Idea** (vortexxx's translation smoother, written from scratch; see `competitors.md`): replace the causal pull back
to the start with a Kalman filter + RTS smoother over our own position change. Per axis the state is position, a
slowly varying velocity error and the workspace centre; every block (0.2 s) a soft prior keeps the position within
σ of the centre. Variants: the velocity-error noise and the displacement noise grow with the organisers' damage ramp,
the prior tightens toward the end, and a hand-back to the causal estimate before the IMU dies (`tools/exp_smoother.py`).
`main.py` runs with the pull off; nothing in `src/` changes. Ours here is `main` @ 4122d58, before rotor drag.

**Result: dropped.** On development data the best setting (σ 5 m, no ramps) gains about 12 % on ATE, checkpoints and
dropouts in every group (Machine Hall 3.94 → 3.53, Vicon 1.74 → 1.46, TUM-VI 0.68 → 0.58 m ATE), but **no setting
keeps blind as good as the entry** (best 2.300 vs 2.276 m), and the one-time check finds it **worse than standing
still on TUM-VI room5 and room6** at every seed, the same two held-out rooms that defeated the zero-phase pull
(`zero_phase_pull.md`). Handing back to the causal estimate before the failure makes blind worse, not better.

**What it taught us.** Blind 5 s error is almost entirely the error at the moment of failure (about 2.2 m against
0.35 m of motion afterwards). EuRoC Machine Hall flights land within 0.3-0.9 m of their take-off point, 3-5 m from
the centre of the path. Our trajectory is pulled so hard to the start that, after alignment, the landing point sits at
the centre; vortexxx's has real extent (rotor-drag velocity), so its landing lands right (0.2-0.7 m on MH_03-05).
That gap needs real velocity information (rotor drag), not a different pull. Room5's raw velocity also suffered from
tilt lost in dropouts, which `gap_bridge.md` addresses.

### Development sweep (seed 42, 11 development recordings; mean metres)

| setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | worst ATE ratio to standing still |
|---|---:|---:|---:|---:|---:|---:|
| entry | 1.952 | 1.751 | 2.276 | 1.760 | 0.486 | 0.98 |
| still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 1.00 |
| sigma_h 1.5, qb0 0.0001, a 0, qp 0.001 | 1.794 | 1.607 | 2.318 | 1.617 | 0.423 | 0.96 |
| sigma_h 1.5, qb0 0.0001, a 0, qp 0.01 | 1.846 | 1.671 | 2.317 | 1.668 | 0.437 | 0.97 |
| sigma_h 1.5, qb0 0.0001, a 30, qp 0.001 | 1.878 | 1.694 | 2.317 | 1.698 | 0.439 | 0.98 |
| sigma_h 1.5, qb0 0.0001, a 30, qp 0.01 | 1.905 | 1.730 | 2.317 | 1.724 | 0.449 | 0.98 |
| sigma_h 1.5, qb0 0.001, a 0, qp 0.001 | 1.889 | 1.715 | 2.315 | 1.705 | 0.439 | 0.98 |
| sigma_h 1.5, qb0 0.001, a 0, qp 0.01 | 1.913 | 1.745 | 2.316 | 1.728 | 0.449 | 0.98 |
| sigma_h 1.5, qb0 0.001, a 30, qp 0.001 | 1.998 | 1.833 | 2.319 | 1.808 | 0.477 | 1.00 |
| sigma_h 1.5, qb0 0.001, a 30, qp 0.01 | 2.007 | 1.845 | 2.320 | 1.817 | 0.483 | 1.00 |
| sigma_h 1.5, qb0 0.01, a 0, qp 0.001 | 2.010 | 1.853 | 2.317 | 1.813 | 0.479 | 0.99 |
| sigma_h 1.5, qb0 0.01, a 0, qp 0.01 | 2.018 | 1.863 | 2.318 | 1.822 | 0.484 | 0.99 |
| sigma_h 1.5, qb0 0.01, a 30, qp 0.001 | 2.126 | 1.977 | 2.324 | 1.931 | 0.555 | 1.00 |
| sigma_h 1.5, qb0 0.01, a 30, qp 0.01 | 2.128 | 1.980 | 2.324 | 1.934 | 0.557 | 1.00 |
| sigma_h 3, qb0 0.0001, a 0, qp 0.001 | 1.743 | 1.545 | 2.338 | 1.562 | 0.419 | 0.94 |
| sigma_h 3, qb0 0.0001, a 0, qp 0.01 | 1.766 | 1.573 | 2.328 | 1.588 | 0.423 | 0.95 |
| sigma_h 3, qb0 0.0001, a 30, qp 0.001 | 1.816 | 1.623 | 2.316 | 1.638 | 0.426 | 0.97 |
| sigma_h 3, qb0 0.0001, a 30, qp 0.01 | 1.827 | 1.638 | 2.317 | 1.650 | 0.430 | 0.97 |
| sigma_h 3, qb0 0.001, a 0, qp 0.001 | 1.824 | 1.641 | 2.313 | 1.645 | 0.426 | 0.97 |
| sigma_h 3, qb0 0.001, a 0, qp 0.01 | 1.835 | 1.655 | 2.314 | 1.657 | 0.430 | 0.97 |
| sigma_h 3, qb0 0.001, a 30, qp 0.001 | 1.922 | 1.744 | 2.319 | 1.740 | 0.450 | 0.99 |
| sigma_h 3, qb0 0.001, a 30, qp 0.01 | 1.927 | 1.751 | 2.319 | 1.744 | 0.452 | 0.99 |
| sigma_h 3, qb0 0.01, a 0, qp 0.001 | 1.935 | 1.767 | 2.318 | 1.746 | 0.451 | 0.99 |
| sigma_h 3, qb0 0.01, a 0, qp 0.01 | 1.940 | 1.772 | 2.318 | 1.750 | 0.453 | 0.99 |
| sigma_h 3, qb0 0.01, a 30, qp 0.001 | 2.048 | 1.890 | 2.322 | 1.855 | 0.504 | 1.00 |
| sigma_h 3, qb0 0.01, a 30, qp 0.01 | 2.050 | 1.892 | 2.322 | 1.856 | 0.505 | 1.00 |
| sigma_h 5, qb0 0.0001, a 0, qp 0.001 | 1.706 | 1.499 | 2.341 | 1.520 | 0.418 | 0.93 |
| sigma_h 5, qb0 0.0001, a 0, qp 0.01 | 1.719 | 1.516 | 2.330 | 1.536 | 0.420 | 0.93 |
| sigma_h 5, qb0 0.0001, a 30, qp 0.001 | 1.777 | 1.581 | 2.317 | 1.599 | 0.422 | 0.95 |
| sigma_h 5, qb0 0.0001, a 30, qp 0.01 | 1.783 | 1.588 | 2.318 | 1.606 | 0.423 | 0.96 |
| sigma_h 5, qb0 0.001, a 0, qp 0.001 | 1.784 | 1.595 | 2.320 | 1.607 | 0.422 | 0.96 |
| sigma_h 5, qb0 0.001, a 0, qp 0.01 | 1.790 | 1.602 | 2.320 | 1.613 | 0.423 | 0.96 |
| sigma_h 5, qb0 0.001, a 30, qp 0.001 | 1.870 | 1.684 | 2.320 | 1.692 | 0.437 | 0.98 |
| sigma_h 5, qb0 0.001, a 30, qp 0.01 | 1.873 | 1.688 | 2.320 | 1.694 | 0.438 | 0.98 |
| sigma_h 5, qb0 0.01, a 0, qp 0.001 | 1.881 | 1.706 | 2.317 | 1.699 | 0.437 | 0.98 |
| sigma_h 5, qb0 0.01, a 0, qp 0.01 | 1.884 | 1.709 | 2.318 | 1.701 | 0.439 | 0.98 |
| sigma_h 5, qb0 0.01, a 30, qp 0.001 | 1.991 | 1.825 | 2.320 | 1.803 | 0.476 | 0.99 |
| sigma_h 5, qb0 0.01, a 30, qp 0.01 | 1.992 | 1.826 | 2.320 | 1.804 | 0.476 | 0.99 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0 | 1.743 | 1.545 | 2.338 | 1.562 | 0.419 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, sigma_end 2 | 1.750 | 1.553 | 2.326 | 1.569 | 0.420 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, sigma_end 1 | 1.758 | 1.562 | 2.318 | 1.579 | 0.421 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, sigma_end 0.5 | 1.764 | 1.568 | 2.317 | 1.585 | 0.422 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 30 | 1.767 | 1.570 | 2.324 | 1.588 | 0.424 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 30, sigma_end 2 | 1.779 | 1.581 | 2.321 | 1.600 | 0.427 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 30, sigma_end 1 | 1.795 | 1.596 | 2.320 | 1.616 | 0.431 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 30, sigma_end 0.5 | 1.806 | 1.606 | 2.321 | 1.627 | 0.435 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100 | 1.806 | 1.610 | 2.314 | 1.630 | 0.435 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, sigma_end 2 | 1.823 | 1.626 | 2.318 | 1.646 | 0.440 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, sigma_end 1 | 1.845 | 1.645 | 2.321 | 1.667 | 0.448 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, sigma_end 0.5 | 1.859 | 1.658 | 2.323 | 1.681 | 0.455 | 0.97 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 300 | 1.872 | 1.683 | 2.312 | 1.698 | 0.456 | 0.97 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 300, sigma_end 2 | 1.892 | 1.702 | 2.318 | 1.717 | 0.465 | 0.97 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 300, sigma_end 1 | 1.917 | 1.725 | 2.322 | 1.741 | 0.478 | 0.98 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 300, sigma_end 0.5 | 1.931 | 1.738 | 2.324 | 1.755 | 0.488 | 0.98 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0 | 1.706 | 1.499 | 2.341 | 1.520 | 0.418 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, sigma_end 2 | 1.718 | 1.512 | 2.326 | 1.534 | 0.419 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, sigma_end 1 | 1.723 | 1.518 | 2.320 | 1.540 | 0.419 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, sigma_end 0.5 | 1.727 | 1.521 | 2.318 | 1.543 | 0.420 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 30 | 1.719 | 1.514 | 2.324 | 1.536 | 0.420 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 30, sigma_end 2 | 1.739 | 1.533 | 2.321 | 1.557 | 0.423 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 30, sigma_end 1 | 1.749 | 1.542 | 2.320 | 1.567 | 0.426 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 30, sigma_end 0.5 | 1.755 | 1.548 | 2.320 | 1.574 | 0.427 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100 | 1.745 | 1.542 | 2.308 | 1.565 | 0.425 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, sigma_end 2 | 1.773 | 1.568 | 2.318 | 1.594 | 0.432 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, sigma_end 1 | 1.787 | 1.580 | 2.321 | 1.607 | 0.436 | 0.95 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, sigma_end 0.5 | 1.796 | 1.587 | 2.322 | 1.615 | 0.440 | 0.95 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 300 | 1.794 | 1.595 | 2.300 | 1.619 | 0.436 | 0.95 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 300, sigma_end 2 | 1.831 | 1.627 | 2.318 | 1.654 | 0.449 | 0.96 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 300, sigma_end 1 | 1.847 | 1.642 | 2.322 | 1.670 | 0.457 | 0.96 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 300, sigma_end 0.5 | 1.857 | 1.650 | 2.324 | 1.679 | 0.462 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, taper 0 | 1.750 | 1.545 | 2.414 | 1.562 | 0.421 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, taper 10 | 1.761 | 1.548 | 2.407 | 1.573 | 0.419 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, taper 30 | 1.771 | 1.562 | 2.396 | 1.584 | 0.422 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 0, taper 60 | 1.784 | 1.582 | 2.380 | 1.602 | 0.426 | 0.94 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, taper 0 | 1.812 | 1.610 | 2.397 | 1.630 | 0.436 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, taper 10 | 1.821 | 1.613 | 2.389 | 1.638 | 0.435 | 0.96 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, taper 30 | 1.826 | 1.622 | 2.381 | 1.645 | 0.436 | 0.95 |
| sigma_h 3, qb0 0.0001, qp 0.001, ap 100, taper 60 | 1.833 | 1.635 | 2.371 | 1.654 | 0.438 | 0.96 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, taper 0 | 1.717 | 1.499 | 2.503 | 1.521 | 0.423 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, taper 10 | 1.734 | 1.503 | 2.487 | 1.538 | 0.419 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, taper 30 | 1.744 | 1.520 | 2.462 | 1.552 | 0.421 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 0, taper 60 | 1.749 | 1.534 | 2.426 | 1.561 | 0.426 | 0.93 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, taper 0 | 1.755 | 1.542 | 2.473 | 1.566 | 0.428 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, taper 10 | 1.770 | 1.546 | 2.459 | 1.582 | 0.426 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, taper 30 | 1.779 | 1.561 | 2.439 | 1.593 | 0.428 | 0.94 |
| sigma_h 5, qb0 0.0001, qp 0.001, ap 100, taper 60 | 1.783 | 1.573 | 2.411 | 1.600 | 0.431 | 0.94 |

### Check of `sigma_h 5, qb0 0.0001, qp 0.001` (mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s |
|---|---|---:|---:|---:|---:|---:|
| held-out 6, seed 42 | entry | 2.980 | 2.823 | 2.161 | 2.813 | 0.731 |
| held-out 6, seed 42 | sigma_h 5, qb0 0.0001, qp 0.001 | 2.462 | 2.337 | 2.151 | 2.329 | 0.655 |
| held-out 6, seed 42 | still | 3.209 | 3.015 | 2.202 | 3.020 | 0.838 |
| all 17, seed 42 | entry | 2.315 | 2.129 | 2.236 | 2.131 | 0.573 |
| all 17, seed 42 | sigma_h 5, qb0 0.0001, qp 0.001 | 1.973 | 1.795 | 2.274 | 1.806 | 0.502 |
| all 17, seed 42 | still | 2.618 | 2.448 | 2.281 | 2.421 | 0.775 |
| all 17, seed 1 | entry | 2.315 | 2.125 | 2.221 | 2.138 | 0.572 |
| all 17, seed 1 | sigma_h 5, qb0 0.0001, qp 0.001 | 2.008 | 1.822 | 2.234 | 1.834 | 0.509 |
| all 17, seed 1 | still | 2.617 | 2.448 | 2.277 | 2.439 | 0.774 |
| all 17, seed 7 | entry | 2.303 | 2.113 | 2.231 | 2.134 | 0.565 |
| all 17, seed 7 | sigma_h 5, qb0 0.0001, qp 0.001 | 1.946 | 1.771 | 2.284 | 1.785 | 0.489 |
| all 17, seed 7 | still | 2.617 | 2.448 | 2.278 | 2.441 | 0.775 |
| all 17, seed 2026 | entry | 2.305 | 2.113 | 2.221 | 2.144 | 0.567 |
| all 17, seed 2026 | sigma_h 5, qb0 0.0001, qp 0.001 | 1.959 | 1.782 | 2.262 | 1.803 | 0.497 |
| all 17, seed 2026 | still | 2.616 | 2.448 | 2.279 | 2.452 | 0.775 |

Not better than standing still: s42_accel_first tumvi:room5 ate_rmse_m 1.04 vs 0.93; s42_accel_first tumvi:room5 checkpoints_m 1.10 vs 0.89; s42_accel_first tumvi:room5 dropout_m 0.97 vs 0.91; s42_accel_first tumvi:room6 ate_rmse_m 1.03 vs 1.00; s42_accel_first tumvi:room6 checkpoints_m 1.09 vs 1.08; s1_accel_first tumvi:room5 ate_rmse_m 1.06 vs 0.93; s1_accel_first tumvi:room5 checkpoints_m 1.10 vs 0.89; s1_accel_first tumvi:room5 dropout_m 0.98 vs 0.90; s1_accel_first tumvi:room6 ate_rmse_m 1.04 vs 1.00; s1_accel_first tumvi:room6 checkpoints_m 1.11 vs 1.08; s7_accel_first tumvi:room5 ate_rmse_m 1.05 vs 0.93; s7_accel_first tumvi:room5 checkpoints_m 1.13 vs 0.89; s7_accel_first tumvi:room5 dropout_m 0.98 vs 0.90; s7_accel_first tumvi:room6 ate_rmse_m 1.04 vs 1.00; s7_accel_first tumvi:room6 checkpoints_m 1.13 vs 1.08; s2026_accel_first tumvi:room5 ate_rmse_m 1.05 vs 0.93; s2026_accel_first tumvi:room5 checkpoints_m 1.12 vs 0.89; s2026_accel_first tumvi:room5 dropout_m 0.98 vs 0.91; s2026_accel_first tumvi:room6 ate_rmse_m 1.05 vs 1.00; s2026_accel_first tumvi:room6 checkpoints_m 1.11 vs 1.08.

## Reproduce

```bash
python tools/exp_smoother.py sweep && python tools/exp_smoother.py sweep2 && python tools/exp_smoother.py sweep3
python tools/exp_smoother.py check --frozen '{"sigma_h": 5.0, "qb0": 0.0001, "qp": 0.001}'
```
