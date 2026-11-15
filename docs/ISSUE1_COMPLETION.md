# Issue #1 implementation and release boundary

This branch implements the estimator-correctness workstream. It does not certify a headset product,
replace the real visual front end in issue #2, or establish hardware readiness in issue #3.
The previous implementation/evidence is commit `70830e5` (based on `ff7ec44`); its before/after
tables remain in `ESTIMATOR.md`. They must not be relabelled as measurements of the latest code.

## Changes in the completion pass

* The model is a fallback by default: healthy visual tracking passively checks its predictions but
  does not fuse another correlated translation update. `model_during_visual_tracking=True` restores
  the unrestricted research behaviour. The six-recording validation comparison exposed both the
  clean-tracking cost and overconfident covariance that motivated this policy; the follow-up run
  keeps the same six recordings/scenarios for an explicitly labelled development comparison.
* `model_max_scale=10` is now the conservative default: a model measured inconsistent by the
  camera-anchored self-check is withheld when healthy visual anchoring disappears. An explicit
  `None` override remains available for research comparisons and must be identified in results.
  This guard can remove learned motion's benefit under severe IMU saturation. It is a safety
  trade-off, not a universal accuracy improvement.
* A learned model's covariance is **unvalidated by default**. With a predictor attached, an
  unanchored pose is marked `COVARIANCE_UNRELIABLE` and `valid=False` unless calibration has been
  explicitly established for that model/domain. A small self-check statistic alone is not such
  evidence. `model_covariance_validated=True` is an evidence assertion by the caller, not automatic
  calibration performed by the library. Keep it false for the current release checkpoint.
* Degraded visual fixes may still contribute to the best-effort pose, but their covariance has
  not been calibrated against real visual-odometry errors. They therefore cannot certify `valid`
  output in this release. The position/covariance and reason codes remain available to callers.
* Lost IMU data immediately invalidates the pose, instead of waiting for the position covariance
  to exceed its limit. Warm-up quality is explicitly unverified. If all sensor readings are
  rejected, the monitor emits NaNs with `lost`, never a median of rejected finite readings.
* Shapes, noise densities, confidence, camera input, geometry rotations, configuration and
  predictor timing parameters are checked at the API boundary. Buffered inputs and configs are
  owned copies. Invalid learned covariance is skipped before it can change filter state or its
  self-check. Rank-deficient calibration remains unverified rather than inventing a noise model.
* IMU p-value diagnostics preserve the original tested ensemble's statistic when isolating a
  sensor. Previously the p-value could be replaced by that of the selected surviving subset,
  which is not a valid null-calibration diagnostic for the initial test.
* Metrics retain missing samples on the original time axis. Relative-error windows cannot bridge
  an initialization/tracking gap; missing/finite/valid fractions are reported. A missing final
  pose has no final-error or drift score. Finite but indefinite covariance produces no NEES score.

## Supported assumptions and explicit restrictions

1. Raw sensor timestamps must be synchronized on a known clock. Time-offset estimation is not
   implemented. `ImuGeometry` accepts only finite proper rotations and mounting offsets; mounting
   calibration is caller-provided and real multi-IMU support remains unvalidated.
2. Parity isolation assumes a single independently faulty sensor with a healthy redundant
   ensemble. More complex simultaneous faults may be detected but can be misattributed, particularly
   if a faulty majority agrees. Identical common-mode bias is unobservable. Do not interpret
   `nominal` as proof of absolute correctness.
3. The Hotelling threshold assumes independent Gaussian windows and adequately estimated mean and
   covariance. Robust trimming, conditional subset selection, sensor correlation and geometry
   approximation can violate that model. `p_false_alarm` is a model target, not a verified hardware
   guarantee. Existing empirical confidence bounds do not establish a 1e-4 false-alarm rate.
4. Without a second trustworthy position source, a long rigid wrong fix cannot be distinguished
   perfectly from inertial drift. Recovery is bounded and explicitly reported; an arbitrary
   prolonged false visual trajectory cannot be promised rejection forever.
5. Full-system production initialization is reference-free, but `pipeline.prepare()` still
   simulates camera fixes from truth and benchmark `init="truth"` is diagnostic. Do not describe
   those experiments as end-to-end image-derived tracking. TLIO is already offline bias-compensated.

## Verification and evidence

**Final suite: 95 passed in 287.57 s.** The preceding pre-fallback-policy suite passed 94 tests.
Both logs are preserved under `results/diagnostics/issue1/completion_evidence/`; final runtime
source hashes were checked against the fallback validation manifest. No runtime edits followed
that validation run. `git diff --check` and Python compilation also passed.

Run `python -m pytest -q` from the repository root. The additional
`tests/test_runtime_safety.py` regressions use no external dataset or trained model. Existing
camera-state tests exercise `pipeline.ours_cfg()`; oracle inputs are diagnostic-only fixtures.

For a controlled validation comparison (not a fresh final test):

```text
python -m results.diagnostics.issue1.validation_policy --seqs 6 --workers 4 --out results/runs/issue1-policy.json
```

This compares no model, the conservative default, the unrestricted legacy policy, and cloned poses
with covariance x1600 on the **validation** split. It fingerprints source, checkpoint and data;
refuses overwrites; reports every per-run error, outage covariance coverage and valid fraction;
and keeps all outputs outside the published benchmark files. Model-policy selection is not a
claim of generalized performance on unseen devices. A wider network without a checkpoint and
comparable validation evidence is not promoted.

### Completion-pass results

Evidence is in `results/diagnostics/issue1/completion_evidence/`. The first six of 35 TLIO validation
recordings were extracted; no test recordings were evaluated. The archive download is ignored by git.
These validation recordings are development data, not an untouched final confirmation set.

`validation-policy.json` contains 96 trajectories (six recordings, four scenarios, four configurations)
before the fallback-only policy was introduced. `fallback-policy.json` contains the subsequent 24
trajectories under the final policy. Source/checkpoint/data hashes and every per-run result are included.
The checkpoint is SHA-256 `611e97fdad4ac7735a3086179e059edb772f6d93d13a40c3b88b8d4bfec3eba3`.

| Scenario | No network, mean ATE m | Earlier unrestricted updates, mean ATE m | Final fallback-only policy, mean ATE m |
| --- | ---: | ---: | ---: |
| Clean | 0.0280 | 0.0403 | 0.0280 |
| Camera blackout | 0.2981 | 0.1208 | 0.1449 |
| Wrong camera | 0.3089 | 0.2690 | 0.3755 |
| Blackout + all-IMU saturation | 3.1723 | 0.4903 | 0.4922 |

The fallback policy removes the clean-tracking penalty while retaining benefits in several outage
cases. **It does not improve every fault:** one wrong-camera recording makes its mean worse than
the no-network path. Its uncalibrated outage estimates are explicitly invalid, not certified poses.
The cloned/covariance-x1600 candidate performed better in this six-recording comparison, but is not
promoted on this small exploratory sample; broader independent validation/calibration remains in #2.
The recorded comparisons neither establish real-vision performance nor production readiness.

`fdi-clean.json` repeats 30 synthetic runs per sensor count: three alarms in 17,250 windows (K=3)
and two in 17,070 (K=5). The nominal 1e-4 rate is still not verified and K=5 still fails the null
p-value uniformity check. Preserving the pre-isolation statistic fixes diagnostic provenance, not
the underlying model mismatch. `unsupported-conditions.json` records 12 synthetic cases: two distinct
faults among five sensors, an agreeing faulty majority among three, and 2/20 ms clock offsets, each
over three seeds. The agreeing faulty majority excluded the healthy sensor in every affected window;
that counterexample is retained and the single-fault/synchronized-input restriction is explicit.

## Gates remaining after code review

| Gate | Owner / condition |
| --- | --- |
| Independent final-data confirmation and real camera measurements | Issue #2: frozen protocol, held-out recordings, real visual front end |
| Full-system learned-covariance calibration, including correlated updates and distribution shift | Issue #2: preserve unvalidated flags until supported by calibration and independent coverage evidence |
| Verified 1e-4 false-alarm operating envelope | Issues #2/#3: enough independent recordings/windows and physical sensor assumptions |
| Multiple simultaneous-fault guarantees and inter-IMU time-offset estimation | Issue #3 hardware extension, with a declared observable failure model |
| Real device latency, mounting calibration and production release | Issue #3 device-access and deployment gates |

These are not implemented by checking an issue box. Keep the parent issue's unverified acceptance
criteria open or explicitly carry them to the linked workstream with its owner and evidence gate.
