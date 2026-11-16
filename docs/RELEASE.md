# Release, provenance, licences and rollback

Status: **research release candidate, not a production release.** No tag has been created by this
change, and creating one is an owner decision (issue #3 D).

## What a release must contain

| Item | Where | Verified by |
|---|---|---|
| Source at a tagged commit | git tag (owners) | CI on the tag |
| Model checkpoints + exports | GitHub release `models-v1` | `models/MODELS.json` digests, `tools/fetch_models.py --verify-only` |
| VO calibration | `calibration/vo_room1_v2.json` | front-end source hash (`evaluate_vision.py` refuses a mismatch) |
| Frozen protocol + exposure | `docs/PROTOCOL.md`, `data/exposure.json` | `tools/check_repo.py` schema check |
| Promoted results | `results/vision/<run>/bundle.json` | SHA-256 of every file and of the source run's rows |
| Environment | `requirements/core-dev-py313.lock`, `pyproject.toml` | hash-checked install |

The ONNX FP32/INT8 exports are in the release. Their numerical parity with the PyTorch model, their
accuracy on degraded input, and their end-to-end trajectory effect were **not** re-validated in this
change. Quantisation is not assumed to be faster.

## Compatibility and rollback

| Library change | Requires |
|---|---|
| `stereo_vo.py`, `camera_model.py`, `image_faults.py`, `vision_frontend.py` | new VO calibration (`calibration/vo_<seq>_v<N+1>.json`); the evaluator enforces this through the source hash |
| estimator/filter defaults | new protocol version; re-run the evaluation; issue #1 contract tests must pass |
| model checkpoint | new `MODELS.json` entry (never overwrite an existing asset), with its training data, seed and validation evidence recorded |

Known-good rollback points: model `models-v1/dispnet_faults.pt` (`611e97fd…eba3`), calibration
`vo_room1_v1.json`, protocol `pathfinder.vision-protocol/1`, and merged issue #1 estimator
`4c11d1f`. To roll back, check out the earlier commit and its calibration file. The models manifest
pins the checkpoint.

## Backup (issue #3 C)

Non-git essentials are the release assets (models), the TLIO and TUM-VI archives (both public,
re-downloadable; only the TUM-VI room1 archive SHA-256 is currently recorded in `data/exposure.json`), `results/runs/*` evaluation
run directories (gitignored, local to this machine), and `local_data/cache` (disposable).

**Open:** durable backup of `results/runs/` to an approved store, plus a restore test. No such store
was authorised in this session. The promoted bundle under `results/vision/` is committed, but the
full `rows.jsonl` of the source run (per-sample series) is local only. Its SHA-256 is recorded in
`bundle.json`, so a restored copy can be checked.

## Third-party licences (to be reviewed by the owners; not legal advice)

| Component | Licence (as stated by the source) | Implication |
|---|---|---|
| TLIO dataset (training data of every checkpoint) | CC BY-NC 4.0 | Non-commercial. Any commercial use of the trained models needs explicit permission or advice. |
| TUM-VI dataset | CC BY 4.0 (dataset website; the statement was not re-fetched in this session) | Attribution. Not redistributed here. |
| Monado SLAM datasets | not verified | Used only by earlier PRs |
| OpenCV | Apache-2.0 | |
| NumPy, SciPy, PyTorch, matplotlib | BSD-style / PSF-compatible | |

The project itself has **no licence file**. Choosing one is an owner decision and has not been made
here. `AI_DISCLOSURE.md` remains in force.

## Changelog (this change, unreleased)

* Real image-derived stereo VO fused through clone-based relative-pose updates. Evaluated on real
  TUM-VI room1 images and IMU under a frozen protocol (see `results/vision/`).
* Removed a constant VO pitch bias by anchoring stereo landmarks on the left ray.
* Relative-mode VO covariance is flagged unvalidated by default.
* Reproducibility: pinned hash-checked lock, CI, verified model download, smoke fixture,
  immutable run directories with manifests, safe dataset extraction, causal profiler.
* `reproduce.sh` no longer retrains silently, and it uses the Windows venv path when present.
