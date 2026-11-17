# Track 3 scoring protocol `pathfinder.track3-score`

Registered 2026-11-17, before any new scored run. The benchmark bundle
`results/runs/20261116-main-c382494-benchmark/` stays as recorded. It was scored
with version 1.

## Version 1 — `legacy` (`pathfinder.track3-score/1`)

`python tools/score_track3.py ... --mode legacy` (also the default).

This is the scorer that wrote that bundle. It linearly interpolates motion-capture
position across every gap inside the first and last reference timestamps, including
the measured gaps of 1.058 s (room1) and 2.492 s (room2). With no `t_fail_s` it still
emits `blind_mean_m` and `blind_end_m` for the last sample. Those fields on a clean
run are not a blind period. Timestamp equality uses `rtol=0` and `atol=1e-6` s.
Metric values for a file that already matched timestamps are unchanged apart from
the added `schema` and `mode` labels.

## Version 2 — `support` (`pathfinder.track3-score/2`)

`python tools/score_track3.py ... --mode support`

An estimate sample is supported only when it lies inside the reference span and
either the bracketing reference gap is at most 50 ms (`--max-gap`) or its timestamp
matches a bracketing reference sample (`rtol=0`, `atol=1e-6` s). Samples before the
first or after the last reference sample are unsupported and are not extrapolated.
Unsupported samples are excluded from ATE, 1 s position drift, checkpoints, dropout
error and blind error. The result reports `support_coverage`, endpoint and gap
counts, and, when a failure or dropout list was declared, coverage of those windows.

`blind_mean_m` is present only when `t_fail_s` was declared and at least one
supported sample lies in the blind window. When `blind_s` is declared, the window
ends at `t_fail_s + blind_s`; otherwise it runs through the last estimate.
`blind_end_m` additionally requires support at the last estimated sample in that
window. `blind_last_supported_m` reports the last supported sample without
claiming it is the endpoint; `blind_end_supported` distinguishes these cases. A clean
run does not grow a one-sample blind metric. A failure window with no reference
support is `blind_status=unsupported` and has no numeric blind error.

SE(3) alignment for ATE uses supported samples only (at least three). The other
errors use a separate alignment fit on supported mission samples only, matching the
legacy split between full-trajectory ATE and mission-only alignment. Fewer than
three supported samples yields `status=insufficient_support` and no ATE.

Ground truth remains an evaluation input. It is not an inference input.
Non-finite poses/reference positions, duplicate or decreasing timestamps, and
non-finite/non-positive gap limits are rejected rather than scored.
