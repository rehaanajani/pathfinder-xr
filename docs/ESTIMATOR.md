# Estimator contract, fault handling and evidence (issue #1)

**Historical implementation/evidence snapshot: `70830e5`.** The completion pass changes the default
model guard and validity policy and adds API/metric checks. See [ISSUE1_COMPLETION.md](ISSUE1_COMPLETION.md)
for the current contract and new evidence. Numerical tables below are retained as historical results,
not rerun measurements of the current defaults. In particular, model withholding is now enabled by
default, unvalidated learned covariance cannot certify an outage, and degraded-camera output is invalid.

Scope: `pathfinder/imu_health.py`, `parity.py`, `filter.py`, `estimator.py`, `pipeline.py`.
Baseline for every before/after number: `main` at `ff7ec44`. **All evidence below is synthetic**
(`synthetic.head_walk`, simulated camera fixes, simulated co-located redundant IMUs) unless a row says
otherwise. None of it is hardware evidence, and none of it re-runs the TLIO / TUM-VI benchmarks.

## 1. Runtime interface (reference-free)

`pathfinder.estimator.StreamingEstimator(cfg, init, predictor=None, expect_camera=True)`

| Input | Units / frame | Notes |
|---|---|---|
| `add_imu(t, gyro, acc, q_add_g, q_add_a, imu_state)` | s; rad/s and m/s² specific force, body frame | strictly increasing `t`; NaN samples allowed (held, flagged `IMU_LOST`) |
| `add_camera(t, fix, health)` | s; m, world/map frame; health in [0, 1] | applied once the IMU reaches `t`; events older than the state are counted `cam_late` and dropped |

Output per IMU sample (`OUTPUT_FIELDS`): `t`, `p` (m, world), `v` (m/s, world), `q` (xyzw, body→world),
`P_pos`, `P_vel` (3×3 filter covariances), `valid`, `track_state` (`initializing`, `tracking`, `degraded`,
`inertial_only`, `lost`), `camera_state`, `imu_state`, `reasons` (bit mask, `estimator.REASONS`), plus the
counters and `events` in `FilterLog`. `valid` is False while initialising, after a numerical failure, or when
the largest position standard deviation exceeds `max_valid_pos_std` (1 m).

World frame z-up, gravity (0, 0, −9.81). Error state `[δθ, δv, δp, δbg, δba]`, `R_true = R Exp(δθ)`.
`filter.run()` is now a thin offline replay of the same streaming code; `estimator.replay()` is the array form.

**Initialisation.** `ReferenceFreeInit` buffers 3 s, then solves position, velocity and full orientation by
Gauss–Newton from healthy camera fixes and IMU pre-integration (gravity fixes roll/pitch; yaw is only as
observable as the horizontal acceleration, and its uncertainty goes into `P0`). Without usable fixes it starts
IMU-only (yaw = 0, v = 0, origin at start) and says so (`HEADING_ARBITRARY`, `VELOCITY_ASSUMED_ZERO`,
`POSITION_FRAME_ARBITRARY`). Biases start at zero with the prior covariance. Buffered samples are reported
invalid. `TruthInit` (reference pose) is for benchmarks and diagnostics only and marks every output
`TRUTH_INIT_DIAGNOSTIC`. The benchmark (`pipeline.run_methods`, `init="truth"` by default) keeps truth init so
methods stay comparable; pass `init="reference_free"` for the runtime path.

**Not reference-free (fixtures only):** `pipeline.prepare` and `camera.visual_fixes` simulate fixes from the
reference trajectory; `tests/helpers.displacement_oracle` is a perfect-displacement oracle used only to isolate
filter behaviour. Neither is a runtime input. `data_tlio.load` rotates TLIO's world-frame IMU into the body frame
with the reference orientation: a valid inverse of the dataset's own conversion, but TLIO's IMU is also
**offline bias-compensated**, which a live headset does not provide. That stays disclosed.

## 2. IMU quality contract (finding 2)

Baseline: `pipeline.fuse_imus` returned `confidence = w.max(0)`, which with 0/1 parity weights is 1.0 whenever
one sensor is kept. Reproduced: 1.000 in every scenario including all-IMU noise and saturation
(`results/diagnostics/issue1/evidence/p1_baseline.txt`). The scalar is removed.

`ImuMonitor` reports per 0.1 s window a **state**, and per sample the **extra process-noise density**
`q_add_g`, `q_add_a` that the filter adds to its own:

| State | Meaning | Extra process noise |
|---|---|---|
| `warmup` | not calibrated yet; unverified | 0 |
| `nominal` | sensors agree (relative evidence only) | 0 |
| `single_unverified` | one sensor, nothing to compare | 0 |
| `isolated` | one or more sensors excluded by parity or a signal check | averaging lost by excluding sensors |
| `unisolated` | disagreement that cannot be attributed (2 sensors, multiple faults, never-calibrated drift) | size of the disagreement over `horizon_s` |
| `common_mode` | every usable sensor shows a signal anomaly (noise, pinned/saturated) | measured excess noise; motion envelope if pinned |
| `lost` | no usable sensor | motion envelope (`lost_sigma_*`) |

What is observable: relative agreement (≥2 sensors), attribution of one fault (≥3), signal-level anomalies per
sensor. What is **not**: any error identical on every sensor (common-mode bias, drift, scale) — verified by
`test_common_mode_bias_is_not_observable_and_is_not_claimed`, which expects `nominal`. That case needs another
modality (camera) and is the reason `nominal` is described as "no evidence of a problem", not "healthy".
No runtime path reads fault labels; the evaluation scripts use them only for scoring.

If every sensor is rejected the fused value is NaN and the state `lost`; the estimator holds the last sample
and propagates with envelope noise (no silent equal average). Samples, window state and timestamps stay
aligned, including incomplete final windows and rate changes (`test_rate_change_and_window_alignment`).
Output latency is one window (0.1 s): decisions use only data of the current and earlier windows
(`test_streaming_equals_batch_and_is_causal`).

## 3. Multi-IMU statistics and geometry (findings 3, 4)

**Calibration (finding 4).** Baseline: scalar MAD of the first ten window means of the *whole recording*;
clean false alarms 2.1 % (K=3) and 2.9 % (K=5) per window against a nominal 0.01 %. Now: full parity
covariance per block (gyro, accel), robust trimming, at least `3(K−1) + 20` clean windows and 2 s, a
two-half stationarity test that restarts calibration if the parity mean moves (a fault starting mid warm-up),
and a prior bound on calibrated inter-sensor offsets that excludes a sensor already faulty at power-on
(3+ sensors) or reports ambiguity (2 sensors). The detection threshold is the Hotelling prediction
threshold, which includes the uncertainty of the estimated mean and covariance (`parity.hotelling_crit`).

Evidence (`results/diagnostics/imu_fdi_evaluation.py --seeds 30`, JSON alongside):

| K | tested windows | false alarms | rate | 95 % Clopper–Pearson | p-value uniformity (KS) |
|---|---:|---:|---:|---|---|
| 3 | 17 250 | 3 | 1.7e-4 | [3.6e-5, 5.1e-4] | D = 0.009, p = 0.16 |
| 5 | 17 070 | 2 | 1.2e-4 | [1.4e-5, 4.2e-4] | D = 0.026, p = 3e-10 |

The nominal 1e-4 is **inside both intervals but not verified**: the upper bounds are 4–5× higher, and
verifying 1e-4 to ±50 % would need on the order of 10⁵–10⁶ independent windows. For K=5 the KS test rejects
uniform p-values, i.e. the Gaussian/independent-window model is measurably imperfect there even on synthetic
data (lag-1 autocorrelation of the statistic ≈ −0.01, so independence is not the main issue; the heavier
tail is not yet explained). Real sensors (non-Gaussian, temperature drift, vibration) will differ more; the
threshold must be re-checked on hardware recordings.

Detection / isolation (same JSON; 10 runs × 30 s per cell, fault from 10 s, faulty sensor rotated):
3 and 5 IMUs isolate bias ≥ (0.01 rad/s, 0.15 m/s²) on any single axis in ≥ 99.9 % of affected windows with
0.1 s delay and ≤ 0.05 % false isolation; a tiny bias (0.003 rad/s, 0.04 m/s²) is caught in 98 % (K=3) and
71 % (K=5) of windows. Two IMUs detect smooth biases (99.5 %) but never attribute them (by design); signal
checks attribute stuck, held/dropped packets, spikes and noise. Multiple simultaneous faults are only
handled by sequential re-testing and are not characterised.

**Geometry (finding 3).** Baseline: `lever_arm_compensate` was never called; ±2 cm offsets flagged 76 % of
clean windows. Now `ImuGeometry(R_bs, r_b)` rotates each sensor into the body frame and removes
`ω×(ω×r) + ω̇×r` (ω̇ from the buffered window). On exact synthetic rigid-body data with ±2/5/10 cm offsets and
90°/180° mounting rotations, non-nominal windows drop from 15–35 % (co-located assumption) to 0.4–0.5 %
(`results/diagnostics/issue1/evidence/p1_after.txt`, `test_lever_arms_and_mounting_rotations_need_geometry`). **Inter-sensor
time offsets are not estimated**, and `ImuGeometry.validated` stays False: real multi-IMU support is
unsupported until measured geometry and synchronised raw streams exist (issue #3 gate).

## 4. Camera supervision (finding 1)

**Root cause (baseline).** With `ours_cfg()` (`adaptive_q=True`), every *healthy* fix — including the ones the
gate rejected — fed the covariance-matching update, so a persistent wrong fix drove `q_scale` from 1 to ≈16
within 0.5 s (`results/diagnostics/issue1/evidence/rc1_baseline.txt`). The inflated process noise opened `P_pos`, the soft
gate's pull grew, and within ~2 s the wrong fix passed the gate as an ordinary measurement (NIS 3.0 at
k = 4400), so the "jump" hold-out never completed. The unit test used `FilterConfig()` (no adaptive Q) and
did not see it. Mean error while the camera was wrong: 0.80 m (shipped) vs 0.155 m (test config) vs 0.83 m
naive, perfect-displacement oracle, seeds 0–3.

**Mechanism now.**
1. Camera fixes pass a **hard** gate (no soft partial pull for vision); adaptive Q is fed only by fixes that
   passed the gate outside a disagreement episode, and is capped (`q_max`).
2. A healthy disagreement opens an **episode**. It is a `suspect_jump` if the state had been anchored by
   ≥ `anchor_min_s` of accepted healthy fixes (and is not inside the provisional period that follows any re-anchor which ended a jump
   episode, `provisional_factor` x the jump hold-out), otherwise `recovering` (the inertial estimate is the suspect), unless the offset exceeds the
   inertial drift envelope.
3. Re-anchoring requires the episode's innovations to be mutually consistent (offset + linear trend after a
   jump, + quadratic after an outage; χ² residual at `consistency_alpha`), the hold-out in **seconds**
   (`recover_after_jump_s` = 10, `recover_after_s` = 1), and, after a jump, that the state's own covariance can
   no longer rule out the fitted offset — or `max_jump_holdout_s` (30 s) has passed.
4. A jump episode only ends on agreement if a likelihood-ratio test prefers "camera unbiased again" over "the
   offset persists" (`episode_end_lr`). Passing the gate is not enough, because the inertial covariance grows
   while fixes are rejected (this is what broke the first rewrite, below).
5. A learned model that its own self-check has already measured as inconsistent
   (`model_scale > model_doubt_scale`, 10; for a consistent model the EWMA's 99.9 % quantile is ≈ 2) is not
   allowed to arbitrate: its updates are withheld during a jump episode, and its tight covariance is not used to
   classify a returning camera as a jump (`MODEL_DISTRUSTED` reason, `disp_withheld` counter).
6. Low-health frames never start, extend or end an episode.

States reported: `nominal`, `degraded`, `outage`, `suspect_jump`, `recovering`; simultaneous camera and IMU
degradation is reported as `BOTH_DEGRADED`.

**Failed attempts kept** (`results/diagnostics/issue1/evidence/`): attempt 1 closed episodes on gate agreement and let repeated
wrong fixes take over (max 2.4 m); later attempts traced and fixed classification after covariance growth,
step-back to the true position, the provisional state after re-anchoring onto a wrong fix, and the
saturation-plus-blackout divergence caused by vetoing the model in outage episodes (1276 m ATE).

**Results, shipped `ours_cfg()`, seeds 0–3** — see §6 table.

**Limits.** With no learned model and no second absolute position source, a rigid wrong fix that outlasts
the hold-out is indistinguishable from a genuine correction; the filter re-anchors to it and, after the camera
returns to the truth, re-anchors back (provisional period). Visual drift slow enough to stay inside the gate
is accepted. Large biases on low-health frames can be accepted once the covariance grows; tracking is then
reported `degraded`, never `tracking`.

## 5. Filter mathematics, uncertainty and the learned model (findings 5, 6, 7, 8)

Checked by tests (`tests/test_estimator.py`): propagation Jacobian (incl. the second-order position terms and
the exact `Exp(−ω dt)` rotation block) against finite differences; error-reset Jacobian `I − [δθ/2]×`;
clone measurement Jacobians (position and yaw-frame forms) against finite differences; clone removal; a 90 s
cloned yaw-frame run with repeated rejections and a re-anchor keeping P symmetric (≤1e-12) and PSD; variable
and irregular IMU timestamps; IMU gaps (sub-stepped with envelope noise); non-finite inputs. Updates use
Joseph form and linear solves.

**Uncertainty (finding 5).** `metrics.nees` / `metrics.coverage` evaluate the 3×3 covariance as a χ²(3)
region. The baseline only exported `sqrt(trace(P))`; its audit statistic was used for the before column.
After (synthetic, seeds 0–3, `results/diagnostics/issue1/evidence/coverage_*.txt`): clean and post-fault segments 100 % inside
the 99.7 % region; without the network, outages are conservative (100 %) and the estimate is withdrawn
(`valid` False) once its std passes 1 m. **With the models-v1 network the covariance is overconfident during
camera outages on synthetic motion**: only 29–41 % of outage samples fall inside the 99.7 % region, and the
network's claimed variance, even ×10 (a stand-in for overlap correlation) or with cloning, does not cover its
error on this out-of-domain motion. Mitigation, not a fix: while the camera does not anchor the state and the
self-check has measured the model as inconsistent, outputs carry `COVARIANCE_UNRELIABLE` and are not `valid`.
With that rule every *valid* outage sample is inside the 99.7 % region in the blackout, saturation, biased-IMU
and vibration scenarios, at the cost of declaring 96 % of those outage samples invalid. The dark-room case
(camera degraded but still accepted) stays at 70–76 % with and without the network: on this synthetic camera
model, low-health fixes are biased beyond the calibrated trust table. Calibrating the network's covariance
remains open (§7).

**Learned model (finding 7).** Synthetic before/after table (§6) and `results/diagnostics/issue1/evidence/model_formulation*.txt`:
on synthetic walks the network makes outages worse than inertial-only (blackout ATE 1.7 vs 0.11 m); its
self-check scale sits at ≈100 (i.e. errors ~10× its claimed std) before the outage. That is synthetic motion,
outside the TLIO training distribution, so it neither confirms nor refutes the TLIO result. An activation
policy `model_max_scale` (withhold the network without camera when its self-check exceeds a threshold) is
implemented but **off by default**: on synthetic data it removes the outage damage but also removes the
network's large benefit under all-IMU saturation (1.9 → 11.2 m), and choosing the threshold needs the TLIO
validation split, which is not available locally. Cloning / `disp_cov_scale = 1600` remain IMU-only results.

## 5b. Image-derived relative motion (issue #2)

`StreamingEstimator(camera_mode="relative")` accepts visual odometry from `pathfinder.stereo_vo`.
It has two event types:

* `add_camera_keyframe(t, kf)` clones (R, p) at `t`. Camera clones and learned-displacement clones are
  pruned independently (`vo_max_clones`), so one kind never evicts the other.
* `add_camera_relative(t, kf, dp, dR, cov6, health)` gives the IMU pose at `t` in keyframe `kf`'s
  IMU frame. The measurement model is `h_t = R_k^T (p - p_k)`, `h_R = R_k^T R`, and the residual is
  `[dp - h_t, Log(h_R^T dR)]`. The Jacobians use right perturbations of the current state and of the
  clone (`tests/test_relative_pose.py` checks them numerically). A 6-DoF χ² gate (`vo_gate`, 22.46)
  rejects inconsistent motion such as frozen frames. A missing clone gives `vo_no_anchor` and is never
  applied. Relative updates never re-anchor absolute position or heading.

The front end reads images only, so the IMU information is not counted twice. Its covariance is the
PnP information with the per-landmark stereo depth covariance, inflated by the residual scale and
multiplied by `vo_cov_scale`, which is calibrated on development data (`calibration/vo_room1_v1.json`).
`vo_covariance_validated=False` (default) marks every relative-mode output
`COVARIANCE_UNRELIABLE`/invalid until held-out coverage evidence exists. Initialisation in relative
mode is the gravity start: the frame is the start pose, with yaw prior 0.1° because odometry fixes the
gauge, and velocity unknown with 1.5 m/s std. Healthy VO updates count as camera anchoring, so the
issue #1 fallback policy holds: the learned model only self-checks while VO tracks.
`model_input_bias_corrected=True` feeds the model IMU samples minus the filter's bias estimate. This
removes a train/runtime mismatch on raw-IMU devices; TLIO training data are bias compensated. It is
off by default and evaluated as an ablation.

Front-end correction found on real data: DLT triangulation splits the residual epipolar error between
the two views. On TUM-VI room1 that error is systematically vertical, and it produced a constant
−0.019°/frame pitch bias against the gyro (about 0.4°/s). Landmarks are now anchored on their
left-image ray (`tests/test_vision_eval.py::test_landmarks_anchor_on_left_ray_so_zero_motion_has_no_bias`).

## 6. Before / after (synthetic)

Wrong-fix and recovery matrix (`results/diagnostics/issue1/wrongfix_matrix.py`; shipped config; mean / max position error in m
during the fault; `oracle` = perfect displacement, diagnostic only):

| Case | displacement | before mean / max | after mean / max | after: error 2–10 s later | after: re-anchors per seed |
|---|---|---:|---:|---:|---|
| wrong const 5s | oracle | 0.421 / 0.804 | **0.007 / 0.022** | 0.013 | [0, 0, 0, 0] |
| wrong const 5s | none | 0.677 / 0.974 | **0.080 / 0.284** | 0.024 | [0, 0, 0, 0] |
| wrong const 10s | oracle | 0.571 / 0.804 | **0.007 / 0.022** | 0.014 | [0, 0, 0, 0] |
| wrong const 10s | none | 0.699 / 0.974 | **0.216 / 1.016** | 0.028 | [1, 1, 1, 1] |
| wrong const 20s | oracle | 0.646 / 0.804 | **0.007 / 0.022** | 0.012 | [0, 0, 0, 0] |
| wrong const 20s | none | 0.710 / 0.974 | **0.468 / 1.016** | 0.026 | [2, 1, 2, 2] |
| wrong walk 10s (bench) | oracle | 0.725 / 1.122 | **0.007 / 0.022** | 0.014 | [0, 0, 0, 0] |
| wrong walk 10s (bench) | none | 0.790 / 1.155 | **0.216 / 1.016** | 0.028 | [0, 0, 0, 0] |
| gradual drift 10s | oracle | 0.082 / 0.153 | **0.080 / 0.148** | 0.015 | [0, 0, 0, 0] |
| gradual drift 10s | none | 0.104 / 0.194 | **0.104 / 0.194** | 0.028 | [0, 0, 0, 0] |
| isolated jump 0.15s | oracle | 0.009 / 0.016 | **0.009 / 0.016** | 0.014 | [0, 0, 0, 0] |
| isolated jump 0.15s | none | 0.015 / 0.024 | **0.015 / 0.019** | 0.026 | [0, 0, 0, 0] |
| repeated 3x2s | oracle | 0.125 / 0.957 | **0.013 / 0.037** | 0.014 | [0, 0, 0, 0] |
| repeated 3x2s | none | 0.320 / 1.065 | **0.047 / 0.262** | 0.027 | [0, 0, 0, 0] |
| blackout 10s | oracle | 0.008 / 0.027 | **0.007 / 0.022** | 0.014 | [0, 0, 0, 0] |
| blackout 10s | none | 0.213 / 0.999 | **0.216 / 1.016** | 0.026 | [0, 0, 0, 0] |
| blackout 10s + bias | oracle | 0.059 / 0.223 | **0.044 / 0.159** | nan | [0, 0, 0, 0] |
| blackout 10s + bias | none | 1.747 / 23.220 | **1.733 / 22.983** | nan | [0, 0, 0, 0] |
| dark 10s | oracle | 0.024 / 0.106 | **0.021 / 0.089** | 0.013 | [0, 0, 0, 0] |
| dark 10s | none | 0.531 / 1.362 | **0.532 / 1.358** | 0.027 | [0, 0, 0, 0] |

Before = `ff7ec44` with the pipeline's own confidence input; after = this branch. Without the network, the wrong-fix rows match the blackout row because a rejected camera leaves the inertial estimate alone; the 20 s no-model case re-anchors onto the wrong fix after the 10 s hold-out (see limits) and back after it ends. `blackout + bias` puts the same bias on every IMU, which no IMU comparison can see.

Scenario table, 2 synthetic sequences × 60 s, same setup as the audit's `synthetic_eval.log` (ATE m):

| Scenario | naive | ours_no_model before → after | ours (models-v1 net) before → after |
|---|---:|---:|---:|
| clean | 0.03 | 0.03 → 0.03 | 0.03 → 0.03 |
| camera_blackout | 0.18 | 0.10 → 0.11 | 1.43 → 1.69 |
| camera_dark | 0.25 | 0.25 → 0.25 | 0.25 → 0.25 |
| camera_blur | 0.23 | 0.23 → 0.26 | 0.22 → 0.25 |
| camera_wrong_fix | 0.31 | 0.32 → 0.11 | 0.31 → 0.15 |
| imu_noise_all | 0.03 | 0.03 → 0.03 | 0.03 → 0.03 |
| one_imu_bias | 0.03 | 0.03 → 0.03 | 0.03 → 0.03 |
| one_imu_stuck | 0.15 | 0.03 → 0.03 | 0.03 → 0.03 |
| one_imu_spikes | 0.04 | 0.03 → 0.03 | 0.03 → 0.03 |
| one_imu_saturation | 0.04 | 0.03 → 0.03 | 0.03 → 0.03 |
| blackout_plus_one_imu_bias | 1.54 | 0.08 → 0.10 | 1.43 → 1.69 |
| dark_plus_imu_drift | 0.25 | 0.30 → 0.25 | 0.30 → 0.25 |
| blackout_plus_vibration_all | 0.69 | 0.87 → 0.67 | 1.42 → 1.84 |
| blackout_plus_spikes_all | 2.47 | 0.95 → 0.26 | 1.57 → 1.71 |
| blackout_plus_saturation_all | 11.30 | 11.36 → 11.28 | 5.96 → 1.88 |
| blackout_plus_dropout_all | 3.68 | 3.84 → 3.83 | 2.87 → 2.15 |

Before = the audit's `synthetic_eval.log` (same seeds, same scenarios); after = `results/diagnostics/issue1/synthetic_eval_after.py` (writes nothing under `results/`). ATE over finite samples, truth-initialised benchmark mode. The network rows are dominated by its out-of-domain error on synthetic motion (§5).

## 7. Still open / blocked

| Item | Blocker |
|---|---|
| Re-run TLIO 36-sequence and TUM-VI benchmarks with the new estimator | datasets not on this machine (3.8 GB / TUM-VI rooms); issue #2/#3 |
| Learned-model activation/weighting policy and covariance calibration on validation | TLIO validation split not available locally |
| Network covariance overconfident in outages (synthetic) | needs validation-split calibration or retraining; not done |
| Real multi-IMU geometry, time offsets, hardware false-alarm rate | measured mounting geometry and synchronised raw streams (issue #3 gate) |
| Real visual measurements (wrong-fix behaviour against a real VIO front end) | issue #2 |
| 1e-4 false-alarm rate verification | needs ~10⁵–10⁶ independent windows, ideally real data |
| Multiple simultaneous IMU faults, inter-IMU time offset estimation | not implemented |
