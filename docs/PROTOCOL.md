# Real-vision evaluation protocol `pathfinder.vision-protocol/3`

**v3 registered before new evaluation, 2026-11-16.** This review-driven development revision
changes camera recovery (gate retained; bias gains frozen), distorted-pixel RANSAC/refinement,
frame-time motion checks, corruption composition, input validation, and scoring support rules.
Historical v1/v2 bundles remain evidence of the earlier implementations, not results for v3.
No untouched recording has become available.

For v3, calibration remains room1 0?60 s with the same scale and keyframe grids. Scale minimizes
`abs(log(median NIS / chi2_6(.5))) + abs(log(p95 NIS / chi2_6(.95)))`.
The keyframe interval minimizes **vision-only** calibration ATE; vision-only rotation RMSE breaks ties.
Raw-gyro SO(3) increments are diagnostics, not reference truth. These rules do not validate covariance.
The stored time offset must match evaluation; calibration transfer across rig/noise identities requires
an explicit recorded flag. The v3 calibration file will be versioned separately after the run.

The systems add `conventional_gated` and `pathfinder_no_reanchor`. `pathfinder_no_adaptive_q` remains
for historical continuity but is not an informative ablation when relative mode supplies no adaptive-Q
input. Health gating means a threshold, not continuous sensor weighting. Single-IMU plateau detections
are not hardware fault labels; this evaluation still reports IMU quality as unverified rather than
using the injected fault schedule to select a monitor. Real hardware range metadata and causal monitor
integration remain required before claiming saturation detection.

Fault baselines use the last supported sample before onset within 50 ms; end errors use the nearest
supported sample within 50 ms. Missing boundaries are unknown, never zero. Recovery needs continuous
sample support for its full hold. Reference gaps are not bridged in path length. VO-only tracking and
covariance-validity are separate fields. NEES remains a gauge-unadjusted marginal-covariance diagnostic.
Scenario resampling intervals are descriptive; unavailable pairs and fixed/random subsets are reported.

The sections below retain the v1/v2 historical definitions where the v3 rules above supersede them.

**Version history.** v1 was registered 2026-11-15 and evaluated once
(`results/vision/20261115T165121Z-vision-room1-aaac9f1/`, retained unchanged). That run exposed an
integration defect: with an IMU fault around an image outage, the 6-DoF gate rejected every later
healthy VO update and Path Finder diverged (1,159 m, 6,848 m, and three random schedules above
300 m). v2 changes the estimator and front end and nothing else: persistent-rejection
recovery (`vo_reanchor`, hold-out 2 s while anchored, 1 s after an outage) and a stale-frame check.
It then re-runs calibration (`calibration/vo_room1_v2.json`) and the identical scenario/system
matrix. The v2 choices were made after seeing v1 results on the **development** recording room1, so
v1 → v2 is development iteration, not confirmation. Neither version used room3–6 (which are exposed by other workstreams anyway; see §2).

Registered 2026-11-15 (v1 before its evaluation run; v2 as described above).
Calibration/selection runs that preceded it are listed in §6. Any change to a frozen item below needs a
new protocol version; analysis added after the evaluation run is labelled *exploratory*.

## 1. Question and scope

Does fusing **image-derived** stereo relative motion with the **real** IMU through Path Finder's
estimator track better than vision alone and than a conventional loosely coupled filter on the same
inputs, under clean and degraded inputs, and how do the safety features behave (outage reporting,
gating, covariance honesty)? Scope: offline replay of one recorded TUM-VI rig. No device, no live
camera, no multi-IMU hardware.

## 2. Data, exposure and splits

`data/exposure.json` is the inventory (all fetched branches were searched for every recording).

| Role | Data | Notes |
|---|---|---|
| Calibration | TUM-VI room1, 0–60 s | VO covariance scale and keyframe interval selection (§4) |
| Development evaluation | TUM-VI room1, full 141 s | Previously exposed (main health model, PR #5/#7/#8). Reported for the full run **and** the 60–141 s within-recording hold-out. Not independent. |
| Development (not run here) | TUM-VI room2, Monado MOO/MIO | Exposed by earlier PRs; room2 download incomplete |
| Exposed by other workstreams | TUM-VI room3–room6, all EuRoC sequences | **Correction (same day):** the first version of this protocol reserved room3–6 as untouched. A re-inventory of all remote branches found them used by the IMU-only Track 3 work (velocity network: room1–4 train, room5 validation, **room6 held-out test**; EuRoC MH/V for train/val/test and evo evaluation). No TUM-VI room or EuRoC recording is untouched. room6 must not be used for vision development because it is another workstream's test set. |
| Final confirmation | **none available** | Needs a stereo-inertial recording with reference that no branch has used (a new team recording or an external dataset), registered before download. Until then, every result here is development evidence. |
| Not usable for ATE | TUM-VI corridor/magistrale/outdoors/slides | Mocap only at start/end |

Conventions verified on development data by `tools/check_sync.py` (run `20261115T170934Z-check-sync-room1-797eb67`; the evaluation recorded the earlier run `…164518Z-check-sync-room1-db7015b`, whose camera offset is identical and whose mocap method was later corrected):
mocap poses are IMU-frame poses on the IMU clock (best offset 0.11 ms; increment rotation 0.05°), the
mocap frame is gravity aligned (mean specific force 0.42° from +z), and camera timestamps match the IMU
clock (best offset 0.17 ms). The camchain has **no** `timeshift_cam_imu`; that absence is recorded as an
assumption. Rule: a measured camera offset is applied only if it exceeds 1 ms (not the case).

## 3. Systems (identical inputs, calibration, initialisation and alignment)

| System | Definition |
|---|---|
| `vo_only` | Stereo VO relative poses chained keyframe to keyframe; holds the last pose across tracking breaks |
| `imu_only` | Same estimator, no camera (dead reckoning from the gravity start) |
| `conventional` | Loosely coupled ESKF with the same VO relative updates: claimed VO covariance (scale 1), no gate, no health weighting, no adaptive process noise |
| `pathfinder` | `pipeline.ours_cfg()` + relative VO updates, calibrated covariance scale, 6-DoF χ² gate |
| `pathfinder_model` | + learned displacement network `dispnet_faults` (TLIO-trained: cross-device), conservative fallback defaults |
| ablations | `pathfinder_no_gate`, `_uncal_cov`, `_no_adaptive_q`, `_no_health`, `pathfinder_model_no_selfcheck` (`model_max_scale=None`), `pathfinder_model_biascorr` (model sees bias-corrected IMU) |

All filters: IMU noise densities from the dataset's own `imu_config.yaml` (Allan values ×2 white, ×10
walk, as published); reference-free gravity initialisation (odometry frame = start pose); no reference
input. **Not run:** an established external VIO (OpenVINS/Basalt/ORB-SLAM3). A Linux toolchain exists
(WSL Ubuntu) but its dependencies could not be fetched at the available bandwidth; this remains an
open requirement, and no result here is "better than established VIO".

Front end: `pathfinder.stereo_vo` (KLT + stereo triangulation anchored on the left ray + RANSAC PnP +
covariance-weighted Gauss–Newton, calibrated equidistant model). Vision-only by construction
(no IMU input), so fusing it does not count IMU information twice.

## 4. Frozen parameters and how they were chosen

Selected by `tools/calibrate_vo.py` on room1 0–60 s and stored in `calibration/vo_room1_v2.json`
(v1: `vo_room1_v1.json`; front-end source hash checked by the evaluator):

1. for each `kf_every ∈ {1, 2, 4}`: covariance scale from {0.25 … 32} by **reference-free** rule
   `argmin |log(mean VO NIS / 6)|`;
2. `kf_every` by the smallest calibration-segment ATE (mocap of the calibration segment only).

Gate χ²₆(0.999) = 22.46; recovery bound/hold below; all other estimator parameters are the issue #1
defaults. The learned-model checkpoint is SHA-256 `611e97fd…eba3`.

## 5. Scenarios (corruption before use)

16 fixed scenarios plus 12 seeded random schedules (seeds 100–111): clean; 5 s blackout at 40 s
(vertical slice); 5 s blackout at start-up; 20 s blackout; darkness; blur; occlusion; low texture;
frame drops; timestamp jitter; frozen frames (wrong visual motion); stereo desynchronisation; blackout +
IMU bias step; blackout + IMU saturation (stress); IMU dropout; IMU noise burst (stress). Random
schedules draw kind, onset (10 % at start-up), log-uniform duration 0.5–10 s and severity; 30 % add an
IMU fault. IMU fault types noise/spikes/saturation/dropout also appear in the network's training
augmentation (torch seed 0, TLIO data); image faults and combinations do not.

## 6. Metrics, comparison and failure rules

* **Primary:** translation ATE RMSE over all estimated samples after start-SE(3) alignment at the first
  initialised estimate (drift fully visible; no re-alignment per outage).
* **Secondary:** 4-DoF full-trajectory ATE; RPE over 1 s and 10 s (translation, rotation); rotation
  error; final error as % of path; per-fault error growth, error at fault end, recovery time
  (first time after the fault from which error stays ≤ error-at-fault-start + 0.10 m for 2 s;
  never = reported), re-anchoring discontinuity (step beyond the reference's own step within 2 s);
  position NEES coverage at 50/90/99/99.7 % with the full 3×3 covariance in the aligned frame;
  valid, estimated and reference-supported fractions; VO used/rejected/no-anchor counts; runtime.
* **Comparison:** paired by scenario (identical corrupted inputs). Report median/mean deltas, wins and
  losses, worst case and a scenario bootstrap; with one recording these describe schedule variability,
  not generalisation. Regressions are reported as prominently as gains.
* **Failures:** every scenario × system is scored or listed with its harness error; nothing is dropped.
  Unsupported reference intervals (gaps > 50 ms) are excluded and counted, never interpolated.
* **Validity:** VO covariance is fitted on development data, so `vo_covariance_validated=False`; every
  relative-mode output is `valid=False` with `COVARIANCE_UNRELIABLE`. This is not changed to improve
  appearance. Validity can only be asserted after held-out coverage evidence on an unused recording.

## 7. Runs belonging to this protocol

* calibration: `calibration/vo_room1_v2.json` (v1: `vo_room1_v1.json`; source runs named inside)
* sync: `results/runs/20261115T170934Z-check-sync-room1-797eb67/sync.json` (evaluation input: `…164518Z-check-sync-room1-db7015b`)
* evaluation and report: see `results/vision/README.md`

Superseded attempts are retained in `results/runs/` and listed in `results/vision/README.md`
(including the calibration before the landmark-anchoring fix, whose ATE was 0.65 m on the calibration
segment).
