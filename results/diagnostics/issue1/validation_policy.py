"""Reproducible issue #1 model-policy comparison on TLIO VALIDATION recordings only.

    python -m results.diagnostics.issue1.validation_policy --seqs 6 --workers 4 --out results/runs/issue1-policy.json

This is development/validation evidence, not an untouched final test or real-vision benchmark.
Camera fixes and redundant sensors are simulated; initial state is reference-based diagnostic mode.
Never overwrites an existing result. The checkpoint, source, data and protocol are fingerprinted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from pathfinder import data_tlio as D, metrics, pipeline
from pathfinder.faults import scenarios
from pathfinder.filter import run

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/dispnet_faults/best.pt"
NAMES = ("clean", "camera_blackout", "camera_wrong_fix", "blackout_plus_saturation_all")
MODES = ("no_model", "safe_default", "legacy_unrestricted", "cloning_x1600")
PREDICTOR = None


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def init():
    import torch
    from pathfinder.predictor import Predictor
    torch.set_num_threads(1)
    global PREDICTOR
    PREDICTOR = Predictor(MODEL)


def job(task):
    sid, seed, name = task
    s = D.load(sid)
    clip = 60.
    start = int(min(30., max(0., s.t[-1] - clip - 1)) * 200)
    s = D.redundant_imus(s.slice(start, start + int(clip * 200) + 1), 3, seed=seed)
    bad, cam = pipeline.prepare(s, scenarios(10.)[name], seed=seed)
    fg, fa, info = pipeline.fuse_imus(bad)
    cfg = pipeline.ours_cfg()
    rows = []
    for mode in MODES:
        config = cfg
        if mode == "legacy_unrestricted":
            config = replace(cfg, model_max_scale=None)
        elif mode == "cloning_x1600":
            config = replace(cfg, cloning=True, disp_cov_scale=1600.)
        p, log, out = run(s, fg, fa, None, cam, None if mode == "no_model" else PREDICTOR,
                          config, imu_quality=info["quality"], return_outputs=True)
        err = np.linalg.norm(p - s.p, axis=1)
        fault = (s.t - s.t[0] >= 20.) & (s.t - s.t[0] < 30.)
        nees = metrics.nees(p[fault], s.p[fault], out["P_pos"][fault])
        rows.append({"sequence": sid, "scenario": name, "mode": mode,
                     "ate_m": metrics.ate(p, s.p), "fault_max_error_m": float(err[fault].max()),
                     "fault_997_coverage": metrics.coverage(nees)[.997],
                     "fault_valid_fraction": float(out["valid"][fault].mean()),
                     "model_used": log.disp_used, "model_withheld": log.disp_withheld,
                     "camera_recoveries": log.cam_recovered})
    print(json.dumps({"sequence": sid, "scenario": name,
                      "ate_m": {r["mode"]: round(r["ate_m"], 4) for r in rows}}), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite existing evidence")
    ids = D.split("val")[:args.seqs]
    if not ids or args.seqs <= 0 or args.workers <= 0:
        raise SystemExit("need validation recordings and positive seq/worker counts")
    sources = {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted((ROOT / "pathfinder").glob("*.py"))}
    manifest = {"data": "TLIO validation split; simulated position fixes and redundant IMUs",
                "split": "val", "initialization": "truth_init_diagnostic", "python": platform.python_version(),
                "sequences": ids, "seed_policy": "validation-list index", "clip_s": 60.,
                "source_sha256": sources, "checkpoint_sha256": fingerprint(MODEL),
                "dataset_sha256": {sid: fingerprint(D.ROOT / sid / "imu0_resampled.npy") for sid in ids},
                "config": asdict(pipeline.ours_cfg()),
                "scope": "development comparison; thresholds not selected on test; no final generalization claim"}
    tasks = [(sid, i, sc) for i, sid in enumerate(ids) for sc in NAMES]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init) as ex:
        for result in ex.map(job, tasks):
            rows.extend(result)
    summary = {sc: {mode: {"mean_ate_m": float(np.mean([r["ate_m"] for r in rows if r["scenario"] == sc and r["mode"] == mode])),
                          "median_ate_m": float(np.median([r["ate_m"] for r in rows if r["scenario"] == sc and r["mode"] == mode]))}
                    for mode in MODES} for sc in NAMES}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def encode(v):
        return v.tolist() if isinstance(v, np.ndarray) else float(v)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "summary": summary, "per_run": rows}, f, indent=2, default=encode)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
