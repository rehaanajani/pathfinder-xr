# What we have: repository, Mac and how it runs (14 Nov 2026)

## On GitHub: `rehaanajani/pathfinder-xr` (private)
32+ commits, all by Rehaan. Collaborators invited with write access: Nikhil (solnikhil), Debadrita (halderdebadrita), Mayank (mayank-ydvv).

| Folder / file | What it is |
|---|---|
| `pathfinder/` (13 files) | The system: `faults.py` (16 failure scenarios), `camera.py` (camera health + simulated camera fixes), `imu_vote.py` (multi-IMU voting), `filter.py` (Kalman filter, soft gate, re-anchoring, model self-check), `model.py` + `predictor.py` (network), `pipeline.py` (runs all 6 methods), `data_tlio.py` (dataset loader), `synthetic.py`, `integrator.py`, `metrics.py`, `sequence.py` |
| `train.py` | Trains the network (fault-aware or TLIO-style) |
| `evaluate.py` | The benchmark: 36 test recordings x 16 scenarios x 6 methods + blackout sweep + IMU isolation, on all CPU cores |
| `make_readme.py`, `make_demo.py`, `make_video.py`, `deck/build_deck.py` | Generate the README results, demo page, video and slides from `results/summary.json` |
| `tests/` | 10 tests (all passing) |
| `results/` | `summary.json` (all numbers), `report.html`, charts, `demo.mp4`, training and benchmark logs |
| `demo/index.html` | Interactive demo page |
| `deck/` | Samsung template + filled deck |
| `docs/industry_research.md` | Maths, competitors, benchmarks, reading list |
| `docs/research_raw/` | Grok, Meta AI and Perplexity answers |
| `reproduce.sh` | One command: set up, download data and models, test, benchmark |
| `README.md`, `AI_DISCLOSURE.md`, `requirements.txt` | Docs and dependencies |

**Release `models-v1`:** `dispnet_faults.pt` (ours) and `dispnet_tlio.pt` (comparison), 1.5 MB each, plus training histories.

## On the Mac (not in git)
All inside the session folder `pathfinder/`, about 9.5 GB in total. **This folder belongs to the Claude session; deleting the session deletes it.**

| Path | Size | What |
|---|---|---|
| `local_data/tlio_golden/` | 8.5 GB | Meta's TLIO headset dataset (356 recordings). Re-downloadable with `reproduce.sh` |
| `models/dispnet_faults/`, `models/dispnet_tlio/` | 3 MB | Our trained models (also on the GitHub release) |
| `models/TLIO_so2/` | 102 MB | EqNIO's pretrained checkpoint. Downloaded but **not used** (no licence) |
| `.venv/` | 881 MB | Python 3.11 with PyTorch, NumPy, SciPy, matplotlib, imageio-ffmpeg |
| Session scratch space | 632 MB | Clones of 16 reference repos (TLIO, RoNIN, EqNIO, AirIO...) used for reading only |

## How it runs
1. `bash reproduce.sh` creates `.venv`, installs `requirements.txt`, runs the tests.
2. Downloads the TLIO dataset (3.8 GB zip) if missing, and fetches both models from the release (or retrains: about 12 min each on an M-series GPU).
3. `python evaluate.py --seqs 36` runs every scenario on 9 CPU cores (about 10 min) and writes `results/`.
4. `python make_demo.py`, `make_readme.py`, `make_video.py`, `deck/build_deck.py` rebuild the demo, README, video and slides from those results.

Inside one run: the dataset recording is loaded, IMU copies get their own noise, faults are injected, the camera health score and visual fixes are computed, then each method (IMU only, naive fusion, learned IMU only, ours without model, ours with TLIO model, ours) runs the Kalman filter over 60 s and is scored against the true path.
