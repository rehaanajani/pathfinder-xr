"""Same table as the audit's synthetic_eval.log (evaluate.py --synthetic --seqs 2, 60 s) WITHOUT writing results/.
Synthetic head walks, simulated camera fixes and redundant IMUs; ATE after yaw alignment, valid samples only."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, '.')
from pathfinder.faults import scenarios
from pathfinder.synthetic import head_walk

M = ["imu_only", "naive_fusion", "tlio_style", "ours_no_model", "ours_tlio_model", "ours"]
_W = {}


def init():
    import torch
    torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    _W["o"] = Predictor("models/dispnet_faults/best.pt")
    _W["t"] = Predictor("models/dispnet_tlio/best.pt")


def job(a):
    name, i = a
    from pathfinder.pipeline import run_methods
    s = head_walk(60, n_imus=3, seed=i)
    r = run_methods(s, scenarios(10.0)[name], seed=i, ours_model=_W["o"], tlio_model=_W["t"], methods=M)
    out = {m: r[m]["ate_m"] for m in M}
    for m in M:
        if "valid" in r[m]:
            out[m + "_valid"] = float(np.mean(r[m]["valid"]))
    return name, i, out


if __name__ == "__main__":
    jobs = [(n, i) for n in scenarios(10.0) for i in range(2)]
    with ProcessPoolExecutor(6, initializer=init) as ex:
        res = list(ex.map(job, jobs))
    tab = {}
    for n, i, o in res:
        tab.setdefault(n, []).append(o)
    for n, rows in tab.items():
        print(f"{n:28s} " + "  ".join(f"{m}={np.mean([r[m] for r in rows]):.2f}" for m in M) +
              "   valid ours=" + f"{np.mean([r.get('ours_valid', 1) for r in rows]):.2f}")
    json.dump(tab, open(sys.argv[1], "w"), indent=1)
