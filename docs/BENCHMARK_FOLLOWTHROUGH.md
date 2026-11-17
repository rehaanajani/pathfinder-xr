# Benchmark follow-through

Started 2026-11-17 from `c382494b414edf1924782d5892b7012274d3b481` on
`fix/benchmark-followthrough-20261117`. Implementation and verification are in progress.

The baseline evidence is `results/runs/20261116-main-c382494-benchmark/`.
Keep that bundle immutable. New results must identify the source, model,
protocol, input hashes, environment, and any uncommitted changes.

## Work ownership

- Entry: Python/C++ implementation and parity, runtime optimization, and blind-period diagnosis.
- Validation: independent demo performance, repository-check concurrency, versioned gap-aware scoring,
  and unused-recording registration/acquisition.
- Vision: fresh frontend calibration, real-image development evaluation, and an established external VIO baseline.
- Integration: review changes across components, run isolated final tests, freeze the implementation,
  and execute the registered unused-recording evaluation only after the freeze.

## Acceptance and execution order

1. Preserve existing correctness guarantees and meaningful regression assertions. Do not relax
   performance limits, change the tested implementation, or suppress failures merely to obtain a pass.
2. Profile the actual failing entry and demo paths. Report timings during parallel development as
   exploratory. Rerun resource checks and the failing timing tests alone after all workers stop.
3. Version changes to scoring. Preserve legacy results, explicitly mask unsupported reference
   intervals, report temporal coverage, and avoid reporting a blind period on a clean run.
4. Assess blind-prediction changes across the same two development rooms and all registered
   degradation variants, including standing-still and physics-only baselines. Record regressions.
   No ground truth or known injection schedule may enter inference.
5. Rerun the complete root, C++ stress, and separate demo suites; record skips and expected failures.
   Run synthetic vision smoke and repository checks. Verify serialized Python/C++ position parity
   to one micrometre with decimal-aware comparison at the output precision.
6. Repeat the registered 38-scenario development matrix with the versioned scorer. Resource tests
   use one warmup plus five measured runs on clean and degraded inputs from both rooms. Preserve
   wall, CPU, memory and individual measurements; existing targets are 20 ms/s and 8 MiB for C++.
7. Register unused data and a confirmatory protocol before model execution. Check team exposure
   and training provenance. Freeze source, weights, configuration, scoring, exclusions, and metrics
   before opening confirmatory results. A newly downloaded but previously used recording is not held out.
   If no suitable unused recording is available, explicitly leave this acceptance item incomplete.
8. Recalibrate the changed visual frontend using development data only; keep covariance validity
   disabled until independent evidence warrants it. Run real-image faults before tracking. Compare
   an established VIO system as a separate baseline, never as an IMU-containing input to the fusion filter.
9. Produce an updated report with per-recording results, worst cases, blind metrics, coverage,
   failures, environment differences and external blockers. Synthetic smoke or exposed-data gains
   are not proof of production readiness or headset performance.

Do not merge or publish incomplete numerical claims. Unavailable original organizer materials,
independent data, external dependencies, or hardware must remain visible limitations.
