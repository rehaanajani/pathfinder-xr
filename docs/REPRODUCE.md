# Reproducing the results

## Supported environments

| Item | Status |
|---|---|
| Python 3.13, Windows 11 (x86-64) | Verified on the development machine. A clean venv was built from the hash-checked lock (see "Fresh-setup check" below). |
| Python 3.13, Ubuntu (GitHub Actions `ubuntu-latest`) and Windows (`windows-latest`) | Defined in `.github/workflows/ci.yml`. The first run happens on this PR, so check its status there. |
| Python 3.11–3.12 | Allowed by `pyproject.toml` but untested. The lock is for 3.13 only. |
| macOS / Apple GPU training | Not tested in this change. `train.py` is unchanged. |
| A second team member reproducing from a clean machine | **Not done.** This needs a person (issue #3 B). |

`requirements/core-dev-py313.lock` is compiled by uv with `--generate-hashes` from
`requirements/core-dev.in` and covers numpy, scipy, OpenCV (headless), PyYAML, matplotlib and
pytest. `opencv-python-headless 4.12.0.88` requires `numpy < 2.3`, so the lock pins numpy 2.2.6. The
development machine's system Python had numpy 2.3.4 with `opencv-python`, which is an inconsistent
pair and is not the supported setup. PyTorch is needed only for the learned model and is not in the
lock. Install the CPU wheel explicitly:
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`. That version is the one
used here. Its install is not hash-locked, which is a known gap.

```bash
# POSIX                                         # Windows (Git Bash / PowerShell)
uv venv --python 3.13 .venv                     uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \      uv pip install --python .venv/Scripts/python.exe \
    --require-hashes -r requirements/core-dev-py313.lock   --require-hashes -r requirements/core-dev-py313.lock
```

`reproduce.sh` now picks `.venv/Scripts/python.exe` or `.venv/bin/python`. It never retrains a model
silently: missing or mismatched checkpoints make it exit. Pass `--train` to retrain on purpose.

## Checks without data (what CI runs)

```bash
python tools/check_repo.py               # compile, JSON schemas, secret patterns, file sizes
python -m pytest -q                      # unit/regression tests + synthetic end-to-end smoke pipeline
python tools/smoke_eval.py --out ci-runs # gated synthetic smoke evaluation in its own run directory
```

The smoke fixture (`pathfinder/smoke.py`) is rendered deterministically into a temporary directory
in the TUM-VI layout. It is redistributable because it contains no third-party data. It is
plumbing evidence only.

## Models

```bash
python tools/fetch_models.py             # gh release download (or GH_TOKEN header), size + SHA-256 checked
python tools/fetch_models.py --verify-only
```

`models/MODELS.json` lists the release `models-v1` assets and their digests. A file is moved into
place only after verification. Tokens are never printed or written. Checkpoints load with
`torch.load(weights_only=True)`, so no arbitrary pickle code runs.

## Real-vision workflow (TUM-VI room1)

```bash
python tools/fetch.py https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar \
    local_data/tumvi/dataset-room1_512_16.tar \
    --sha256 20354392eab5cc82c9770ed29cf0130c5acce0db54a257571499008fb34c398f
python tools/extract_dataset.py local_data/tumvi/dataset-room1_512_16.tar local_data/tumvi
python tools/extract_dataset.py local_data/tumvi/dataset-room1_512_16.tar local_data/tumvi --verify
python tools/check_sync.py --seq room1                       # offsets / frames, development data only
python tools/calibrate_vo.py --seq room1 --segment 0 60 --write calibration/vo_room1_vN.json
python evaluate_vision.py --calibration calibration/vo_room1_v1.json \
    --sync results/runs/<check-sync-run>/sync.json --workers 14
python tools/profile_pipeline.py --seq room1 --seconds 60 --kf-every 4 --vo-cov-scale 4 \
    --model models/dispnet_faults/best.pt
python tools/vision_report.py results/runs/<vision-run> --profile results/runs/<profile-run>
```

The archive SHA-256 was recorded at download (trust on first use). TUM-VI does not publish a
checksum that this session could verify. The verified extraction has 5,658 regular files,
byte-identical to the archive. Two `dso/cam*/images` symlinks were skipped.

Approximate durations on the i7-11800H (16 threads): sync check 3 min, calibration 6 min, full
evaluation (28 scenarios × 11 systems, 14 workers) about 1 h, report under 1 min. The front end
caches results under `local_data/cache/vo`. The cache key covers the code hash, configuration,
inputs, schedule and seed, so any change in these recomputes.

## Result directories

Every run writes `results/runs/<utc>-<name>-<sha>[-dirty]/` with `manifest.json` (commit, dirty
flag, diff hash, packages, hardware, command, input hashes) and `status.json` (`running` →
`complete`, `failed` with traceback, or `interrupted`). Run directories are never overwritten.
`results/runs/` is git-ignored. A run becomes a committed artifact only through
`tools/vision_report.py`, which writes `results/vision/<run>/bundle.json` with the source run id and
the hash of every file. Failed and superseded runs are kept, and `results/vision/README.md` lists them.

## Fresh-setup check (this session, 2026-11-15)

Run on the development machine (i7-11800H, Windows 11, Python 3.13.5). This is the same person and
machine, **not** an independent second-person reproduction.

1. The link to PyPI ran at about 50 KB/s, and `uv pip install --require-hashes` failed twice on network
   timeouts (opencv). The wheels were therefore fetched resumably with `tools/fetch.py`, with the
   SHA-256 checked against PyPI, into a wheelhouse.
2. `uv venv --python 3.13` and `uv pip install --offline --no-index --find-links wheelhouse --require-hashes -r requirements/core-dev-py313.lock`
   installed 19 packages in 19 s. The venv has numpy 2.2.6, scipy 1.18.0, OpenCV 4.12.0 and
   matplotlib 3.10.7, and no torch.
3. In a fresh `git clone` of commit `51192ba`, with BLAS threads = 1: `tools/check_repo.py` found 0
   problems; `pytest` gave **145 passed, 1 skipped** (the symlink test is skipped on Windows without
   symlink privilege) in 462 s; `tools/smoke_eval.py` passed every gate. Total time was 495 s.
4. Not covered: the GitHub Actions run (see the PR checks), Linux, the torch install, and the dataset
   and model-dependent workflow from a clean machine.
