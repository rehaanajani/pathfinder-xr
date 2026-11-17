# Confirmatory recording protocol `pathfinder.confirmatory-protocol/1`

Registered 2026-11-17, before acquisition is evaluated and before any model run.
This is not permission to treat TUM-VI room2 as held-out. Room2 is already exposed.

## Inventory

Every remote branch present on 2026-11-17 was searched (`git grep` over
`git rev-parse --remotes`), plus the working tree. TUM-VI room1–room6 and every
EuRoC `MH_01`–`MH_05`, `V1_01`–`V1_03`, `V2_01`–`V2_03` sequence are referenced.
Monado sequences referenced by name are `MOO05`, `MOO07`, `MOO08`, `MOO09`,
`MOO11`, `MIO08`, `MIO09` and `MIO11` (with and without the descriptive suffix).
No branch names `MIO10` or `MIO10_short_2_panorama`. Literature mentions of
Blackbird, Hilti and KAIST are citations, not recordings this team used.

## Nominated sequence

| Field | Value |
|---|---|
| Id | `monado/MIO10_short_2_panorama` |
| Why this one | Valve Index stereo (2×960×960 @ 54 Hz) plus 1,000 Hz IMU and lighthouse reference, the same layout `demo_line/pathfinder/data_monado.py` already reads. Short unused sequence. Not a calibration take. |
| Source | `https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MI_valve_index/MIO_others/MIO10_short_2_panorama.zip` |
| Published LFS sha256 | `9ca151d8dc5cb84dafecdb8595a22116e0f95162e8a1794eb2ca779bedfe60c4` |
| Published size | 214,797,238 bytes |
| License | CC BY 4.0 (Collabora Monado SLAM dataset; Mateo de Mayo et al., arXiv 2508.00088) |
| Record | `data/confirmatory_mio10.json` |

The hash above is the Git LFS oid returned by the Hugging Face dataset API before download.

## Rules until the parent freezes and evaluates

- Do not open the archive, list poses, or read ground truth.
- Do not run a model, filter, calibrator or scorer on this sequence.
- Do not tune parameters from it.
- Room2 stays a development recording.
- Same headset family as the exposed Index sequences `MIO08`, `MIO09` and `MIO11`. Unused sequence, not a new site or a new device. That limit stays in the report.
- If the file is missing or the sha256 does not match, the confirmatory item is incomplete.

Scoring for a later parent run is `pathfinder.track3-score/2` for trajectories (`docs/TRACK3_SCORE.md`). Vision metrics stay on the vision protocol. This document does not authorise that run.

## Parent evaluation registration (before archive inspection)

The user's request authorizes evaluation after the implementation freeze. Record a
source-tree fingerprint, both model hashes, executable hash, scorer hash, exact
commands, runtime environment, and archive hash in a new immutable run manifest
before opening this archive. Do not reuse the development run directory.

The confirmatory entry experiment uses native IMU timestamps and sensor values
without ground-truth-derived initialization, resampling, or coordinate correction.
Read the reference only in scoring. Compare the frozen C++ default entry,
physics-only entry (`--model none --preset physics`), and a constant-position
baseline on identical timestamps. No method selection or tuning from these results.

Run the clean recording. If its IMU duration is at least 15 seconds, additionally
run the development fault matrix unchanged: seeds 1, 2, 3, terminal zero/NaN/frozen
failure for 5 seconds and missing/zero dropouts (18 variants). For a shorter
recording, report clean-only evidence and the untested fault cases; do not shorten
or tune the schedule after seeing scores. Record any generator incompatibility as
a failed scenario, not an omitted success.

Primary reporting: per-scenario supported ATE, paired differences against each
baseline, blind mean/end error where reference support exists, and support coverage.
Secondary reporting: checkpoint and dropout error, one-second position drift,
strict timestamp/row count, finite output and unit-quaternion checks. Report every
case, including divergence and unavailable metrics. ATE uses full-trajectory SE(3)
alignment; mission metrics use pre-failure alignment. Use a 50 ms reference-gap
limit fixed before evaluation. Do not interpret missing reference as a zero error.

There is no justified absolute accuracy threshold for this new recording. Report
the measured differences and all failures, not a manufactured certification pass.
One unused recording can test transfer to that recording, not establish population
generalization. If the schema cannot be handled without modifying runtime logic,
retain the failed attempt and treat any subsequent adapted run as development.
