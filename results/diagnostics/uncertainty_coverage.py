"""Is the filter's claimed position uncertainty honest? (issue #1 D)

    python results/diagnostics/uncertainty_coverage.py [n_seqs]

For every sample, NEES = e^T P^-1 e with e = estimate - truth and P the filter's 3x3 position
covariance. If P is honest, NEES ~ chi2(3): 95 % of samples inside chi2_3(0.95) = 7.81, mean NEES 3.
Coverage well below 95 % = overconfident; well above = too cautious. Validation split only.
Note: runs start from the true pose (benchmark mode), so the first seconds are trivially covered.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, '.')
from pathfinder import data_tlio as D, pipeline
from pathfinder.faults import scenarios
from pathfinder.filter import run

SCEN = ["clean", "camera_blackout", "camera_dark", "dark_plus_imu_drift", "blackout_plus_one_imu_bias",
        "blackout_plus_saturation_all", "camera_wrong_fix"]
T95 = chi2.ppf(0.95, 3)
W = {}


def init():
    import torch; torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    W['p'] = Predictor('models/dispnet_faults/best.pt')


def job(a):
    i, sid = a
    s = D.load(sid)
    st = int(min(30, max(0, s.t[-1] - 61)) * 200)
    s = D.redundant_imus(s.slice(st, st + 12001), 3, seed=i)
    out = {}
    for sc in SCEN:
        bad, cam = pipeline.prepare(s, scenarios(10.0)[sc], seed=i)
        fg, fa, info = pipeline.fuse_imus(bad)
        conf = np.repeat(info["confidence"], 20)[: len(s.t)]
        est, log = run(s, fg, fa, conf, cam, W['p'], pipeline.ours_cfg())
        e = est - s.p
        P = np.array(log.p_cov)
        nees = np.einsum("ni,nij,nj->n", e, np.linalg.inv(P + np.eye(3) * 1e-12), e)
        sl = slice(int(20 * 200), int(40 * 200))           # fault window and 10 s after it
        out[sc] = {"coverage95_all": float(np.mean(nees <= T95)), "coverage95_fault": float(np.mean(nees[sl] <= T95)),
                   "mean_nees_all": float(np.mean(nees)), "median_nees_all": float(np.median(nees))}
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with ProcessPoolExecutor(9, initializer=init) as ex:
        rows = list(ex.map(job, enumerate(D.split('val')[:n])))
    res = {sc: {k: float(np.mean([r[sc][k] for r in rows])) for k in rows[0][sc]} for sc in SCEN}
    print(f"{'scenario':30s} {'cover95 all':>12s} {'cover95 fault':>14s} {'median NEES':>12s}   (honest: 0.95, 0.95, 2.37)")
    for sc, v in res.items():
        print(f"{sc:30s} {v['coverage95_all']:12.2f} {v['coverage95_fault']:14.2f} {v['median_nees_all']:12.2f}")
    Path("results/uncertainty_coverage.json").write_text(json.dumps({"n_seqs": n, "split": "val", "scenarios": res}, indent=1))
