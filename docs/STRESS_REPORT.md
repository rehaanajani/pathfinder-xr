# Stress report (Phase 1: break the code)

Branch `stress-mayank`, tests in `tests/stress/`. Nothing in `pathfinder/` was changed or tuned.
Machine: MacBook Air, Apple M4, 16 GB, macOS (Darwin 27), Python 3.11 venv from `requirements.lock`.
Baseline: `pytest tests` gives **43 passed, 2 skipped** (the two Monado tests need data we did not download), not 45.

Run: `python -m pytest -q tests/stress` (fast tier, about 1 min) or `STRESS_SLOW=1 python -m pytest -q tests/stress` (about 15 min).
Tests that expose a bug are `xfail(strict=True)` with the issue number, so they flip to a failure when the bug is fixed.

## Bugs found (one issue each)

| # | Issue | Severity | Evidence |
|---|---|---|---|
| 1 | [#13](https://github.com/rehaanajani/pathfinder-xr/issues/13) NaN/inf/1e30 IMU sample raises `ValueError`; acc NaN poisons the state, every later push raises | high | 1 or 2 IMUs and the raw streaming API; 3+ IMUs are handled by parity |
| 2 | [#14](https://github.com/rehaanajani/pathfinder-xr/issues/14) NaN camera health / repeated inf fix raises; NaN health re-raises on every later push (`pending_cam` never cleared) | high | repro in the issue |
| 3 | [#15](https://github.com/rehaanajani/pathfinder-xr/issues/15) Uncertainty floor never applies until a healthy fix was accepted once: `valid=True` at 10-40 m error | medium | real TLIO val: blind from 0 s, error 5.6-11 m at 10 s while claiming 0.51 m |
| 4 | [#16](https://github.com/rehaanajani/pathfinder-xr/issues/16) Stream != run with saturation: window 2 samples short; offline mask non-causal in the first 2 s | medium | 3 of 600 fuzz cases; fix verified locally by lengthening the buffer |
| 5 | [#17](https://github.com/rehaanajani/pathfinder-xr/issues/17) Stream != run at nominal rate (offline dt = median diff, one ULP off) | low | up to 0.17 m; 5 of 48 early fuzz cases |
| 6 | [#18](https://github.com/rehaanajani/pathfinder-xr/issues/18) Live server / analyze crash on truncated or malformed line | medium | real session B with its last line cut |
| 7 | [#19](https://github.com/rehaanajani/pathfinder-xr/issues/19) README says 10 s hold-out (code and measurement: 30 s); 1-sample `prepare` crash | low | |

Fuzz equality (`stream == run`, 1e-9) holds only under a **matched dt** (`rate = 1/seq.dt`); at the nominal rate it fails (#17). The harness uses matched dt and says so.

## What held up

* **Fuzz, 600 random cases** (1-6 IMUs, up to 5 overlapping faults including at t = 0 and the last sample, camera drop 0-100%): no exception, finite output, covariance symmetric and PSD (1e-9 relative), faster than real time. The only violations are the 3 saturation cases (#16). 60 more cases with the real learned model running: clean.
* 10-minute run (120,000 samples, camera blackout and IMU bias): all invariants hold.
* Rates 100/400/1000 Hz: invariants hold. Real-time factor (3 IMUs, stream loop with checks): 105x, 42x, 15x, **3.2x** at 1000 Hz (per-sample `np.array(self.raw)` is O(window); fine at 200 Hz, thin at 1 kHz). Error during a 5 s blackout rises with rate (0.66 to 1.29 m), but the synthetic IMU noise is per sample, so noise density grows with rate: confounded, not claimed as a bug.
* NaN bursts on 3+ IMUs: handled (error 0.016 m final).
* Determinism: same seed gives bit-identical output, also after another run crashed in between. `cfg.use_gate` is toggled without `try/finally` in the re-anchor path; we could not make it raise, so it is a code-review note, not a bug.
* Real phone sessions A and B reproduce the published numbers exactly (1.96 / 0.50 m and 0.60 m, `valid=False` with claimed +-8.3 / 49.9 / 15.9 m).
* Realistic link problems on both real sessions (20% duplicated lines, 30% IMU lines dropped, 20 s of ARCore dropped, +10 s camera clock, +10 s IMU clock): no crash, and fewer than 1% of frames are `valid` yet more than 2 m from ARCore where the baseline agreed.
* The 61 s IMU hole inside session A (Wi-Fi drop): the first frame after it is 6 m off and honestly reported (`std` 15.9 m, `valid=False`).
* A +100 m camera jump is held out for 30 s (`valid=False` from 4 s on), then adopted.

## Measured limits (not filed as bugs)

* **Trying to break the 0.6 m claim.** We removed 20, 30 and 60 s of ARCore lines from both real sessions at 4 start times each (20 windows where ARCore was tracking continuously, walker moving; ARCore's pose when it returns is the reference; state read just before the first returning frame). Result: `valid=False` in 20 of 20, claimed std 5-26 m, and the error of ours was below the claimed std in 20 of 20 (0.3-8.3 m; 11 of 20 above 1 m). The standard tracker was 16-26,000 m off. The error tracks the distance ARCore moved during the window (e.g. 4.1 m after 30 s with 4.05 m walked), i.e. the stand-still prior is what produces the small numbers: **0.6 m holds when the person stays roughly still; if they walk, the error equals the distance walked, and the tracker says so (`valid=False`) but is not accurate.** Four A windows (235/255/260 start, 340) end 7-8 m off with ARCore having moved under 1 m: a drift not explained by walking, still flagged invalid. A first version of this sweep scored the wrong frame; it was redone.
* **Frozen camera pose** (identical fixes, health 1.0, walker moving, TLIO val): `valid=True` for 56-58 s, 11-29 m off. ARCore re-sending identical frames is deduplicated by the server by timestamp, but a pose that really freezes is trusted.
* **Wrong camera, model off** (`camera_wrong_fix`, 12 val recordings x 3 seeds; the drift path comes from a fixed `default_rng(1234)`, so the 36 runs share one wrong path): 25% of samples in the window are `valid` with `e > 3 std`; 36 of 36 runs pass 1 m before the flag drops. With the learned model: 5%. README already lists this scenario as a tie.
* **`position_cov` is not floored.** Only `position_std_m` gets the outage floor. NEES computed from `position_cov` has mean 3.0 for 10 s blackouts without the model, but 16-198 with the model (tail driven); the consumer should read `position_std_m`.
* Honesty numbers over TLIO val (stream, shipped settings; `tests/stress/honesty.py`): see table. Blackouts are honest without the model (coverage 0.96 at 10 s, 0.83 at 30 s); `valid` rarely sits True while more than 1 m off (0.2% of samples).

| scenario (12 recordings x 3 seeds) | model | e>3 std | 95% cover | NEES mean | runs reaching 1 m | flag late |
|---|---|---:|---:|---:|---:|---:|
| blackout 10 s | off | 1.8% | 0.96 | 3.0 | 15 | 9 |
| blackout 10 s | on | 0.4% | 0.93 | 16.2 | 5 | 0 |
| blackout 30 s | off | 9.2% | 0.83 | 58.5 | 24 | 9 |
| blackout 30 s | on | 1.2% | 0.96 | 197.8 | 16 | 0 |
| blackout 10 s + biased IMU | off | 2.9% | 0.96 | 3.0 | 19 | 9 |
| wrong camera 10 s | off | 25.4% | 0.73 | 108.5 | 36 | 36 |
| wrong camera 10 s | on | 4.9% | 0.87 | 167.8 | 9 | 1 |
| dark camera 10 s | off | 0.1% | 0.99 | 6.6 | 20 | 0 |

* **Live server, reordering and skew:** lines swapped within +-0.4 s leave 2.4-4% of frames `valid` and more than 2 m from ARCore (baseline agreed); a 10 s gyro/accelerometer clock skew leaves 5-8%. Late lines are dropped by the 200 Hz grid. Timestamp shifts of the whole camera or IMU stream are harmless (fixes are applied on arrival). Before the first TRACKING frame the IMU buffer grows without bound (60,001 samples after 60 s; no O(n^2) because `_drain` returns early).
* The synthetic `head_walk` is faster than real heads (mean 1.4 m/s, p99 2.1-2.5 vs `v_max` 2; TLIO val mean 0.48, p99 1.6), so every "valid while wrong" claim above was re-measured on TLIO val, not on the synthetic walk.
* Baseline session A has 548 frames at 2-8 m from ARCore with `valid=True`; they sit right after ARCore itself jumped 2-4.5 m between consecutive frames (frames 5009, 5309, 8390, 15027). The tracker held those jumps out, so this is a result in its favour, not garbage.

## Not done

Phase 2 (campaign) was cancelled by the team in favour of stressing the Track 3 entry. Camera-only scenarios were not pursued further.
