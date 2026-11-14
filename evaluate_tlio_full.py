"""IMU-only tracking on the FULL TLIO test split, to compare with published TLIO / EqNIO numbers.

    python evaluate_tlio_full.py

Every test sequence from start to end (3-7 min), one real IMU, no camera, initialised from the
ground-truth pose at t=0 (as in TLIO). Reports ATE (RMSE, metres) both without alignment and
after yaw + translation alignment. Published reference values (EqNIO, arXiv 2408.06321, Table 1;
TLIO, arXiv 2007.01867): TLIO 1.722 m, TLIO + rotation augmentation 1.812 m, EqNIO O(2) 1.433 m.
Their exact alignment and filter settings differ from ours, so treat this as an indicative, not
like-for-like, comparison.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pathfinder import data_tlio as D
from pathfinder import metrics
from pathfinder.filter import FilterConfig, run

_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    _W["ours"] = Predictor("models/dispnet_faults/best.pt")
    _W["tlio"] = Predictor("models/dispnet_tlio/best.pt")


def _job(sid):
    s = D.load(sid)
    out = {"seq": sid, "seconds": float(s.t[-1]),
           "length_m": float(np.sum(np.linalg.norm(np.diff(s.p, axis=0), axis=1)))}
    for name in ("ours", "tlio"):
        est, _ = run(s, s.gyro[0], s.acc[0], None, None, _W[name], FilterConfig())
        out[name] = {"ate_raw_m": metrics.ate(est, s.p, align=False), "ate_yaw_aligned_m": metrics.ate(est, s.p),
                     "drift_pct": metrics.drift_pct(est, s.p)}
    print(sid, {k: round(out[k]["ate_raw_m"], 3) for k in ("ours", "tlio")}, flush=True)
    return out


def main():
    ids = D.split("test")
    with ProcessPoolExecutor(min(8, os.cpu_count() or 1), initializer=_init) as ex:
        rows = list(ex.map(_job, ids))
    summ = {}
    for name in ("ours", "tlio"):
        summ[name] = {k: float(np.mean([r[name][k] for r in rows])) for k in rows[0][name]}
        summ[name]["ate_raw_median_m"] = float(np.median([r[name]["ate_raw_m"] for r in rows]))
    res = {"n_sequences": len(rows), "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
           "summary": summ, "published": {"TLIO (paper)": 1.722, "TLIO + rot. aug. (EqNIO table)": 1.812,
                                           "EqNIO O(2)": 1.433},
           "per_sequence": rows}
    Path("results/tlio_full_split.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_sequences", "mean_seconds", "summary", "published")}, indent=1))


if __name__ == "__main__":
    main()
