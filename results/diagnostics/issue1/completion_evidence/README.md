# Completion-pass evidence

This evidence extends the historical `70830e5` audit implementation. It does not replace or overwrite
the published TLIO/TUM-VI summaries. Dataset files and checkpoints are not committed here.

* `pytest-final.txt`: 94 passing tests before the fallback-only policy.
* `pytest-fallback-final.txt`: final code, 95 passing tests in 287.57 seconds.
* `validation-policy.json`: 96 development trajectories on the first six TLIO validation recordings;
  four scenarios by four configurations. This precedes the final fallback-only policy. The source,
  checkpoint and extracted data hashes, initialization and configuration are embedded.
* `fallback-policy.json`: 24 trajectories repeating the same six development recordings/scenarios
  under the final policy. Runtime source hashes match the final code. No test-split trajectories
  were evaluated. Camera fixes and redundant IMUs are simulated; startup uses the reference and
  is explicitly diagnostic. This is not real visual odometry or a fresh final benchmark.
* `fdi-clean.json`: 30 synthetic runs each for three/five IMUs. Confidence intervals do not verify
  the nominal 1e-4 false-alarm target. The five-IMU null-distribution check still rejects the model.
* `unsupported-conditions.json`: 12 synthetic characterizations of two distinct faults, colluding
  majorities, and 2/20 ms clock offsets. In the colluding-majority examples the healthy third sensor
  is excluded. These unsupported cases are deliberately retained, not counted as successful isolation.

Commands, from the repository root (BLAS/OMP/MKL threads limited to one for the final runs):

```text
python -m pytest -q
python -m results.diagnostics.issue1.validation_policy --seqs 6 --workers 4 --out results/runs/issue1-completion/validation-policy.json
python -m results.diagnostics.issue1.validation_policy --seqs 6 --workers 4 --modes safe_default --out results/runs/issue1-completion/fallback-policy.json
python -m results.diagnostics.issue1.unsupported_conditions --out results/runs/issue1-completion/unsupported-conditions.json
```

The first validation run used the pre-fallback source fingerprint stored in its manifest. The current
script's `legacy_unrestricted` mode explicitly restores learned updates during visual tracking.
The clean-sensor run used `clean(K, 30, 60.)` from `results.diagnostics.imu_fdi_evaluation` for K=3,5.

Execution history: intermediate full-suite runs were interrupted while the input/validity contract
was still being edited; they are not reported as passing. The initial validity assertions expected
all recovered frames to be valid, including low-health frames; the final tests instead require
healthy recovered poses to be valid and low-health poses to carry the explicit covariance warning,
while retaining the position-error checks. A new p-value regression initially expected detection
on every window after isolation; it was corrected to check the failed initial ensemble test because
hysteresis subsequently tests only retained sensors. The preceding passing and final passing suite
logs are both retained. Earlier implementation failures remain in the sibling `evidence/` directory.

The model is the `models-v1` fault-aware checkpoint; SHA-256 is embedded in both validation manifests.
Clean tracking improves with the fallback policy, but one wrong-camera recording regresses and learned
outage covariance remains uncalibrated. See `docs/ISSUE1_COMPLETION.md` for the supported contract and
explicit issue #2/#3 gates. No hardware or general-production-readiness claim is made.
