"""Does cloning + calibrated network uncertainty help the full system (cameras + IMUs)?

    python results/diagnostics/headset_cloning_check.py [n_seqs]

Validation split only (the test split is kept for the final benchmark). 60 s clips, faults at 20 s.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
sys.path.insert(0, '.')
from pathfinder import data_tlio as D, pipeline
from pathfinder.faults import scenarios

SCEN = ["clean", "camera_blackout", "camera_dark", "dark_plus_imu_drift", "blackout_plus_one_imu_bias",
        "blackout_plus_spikes_all", "blackout_plus_saturation_all"]
CFGS = {"current": {}, "clone_x100": dict(cloning=True, disp_cov_scale=100.0),
        "clone_x1600": dict(cloning=True, disp_cov_scale=1600.0)}
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
    base = pipeline.ours_cfg
    out = {}
    for name, kw in CFGS.items():
        pipeline.ours_cfg = (lambda kw=kw: replace(base(), **kw))
        for sc in SCEN:
            r = pipeline.run_methods(s, scenarios(10.0)[sc], seed=i, ours_model=W['p'], methods=["ours"])
            out[(name, sc)] = r["ours"]["ate_m"]
        pipeline.ours_cfg = base
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    with ProcessPoolExecutor(8, initializer=init) as ex:
        rows = list(ex.map(job, enumerate(D.split('val')[:n])))
    print(f"{'scenario':30s}" + "".join(f"{c:>13s}" for c in CFGS))
    for sc in SCEN:
        print(f"{sc:30s}" + "".join(f"{np.mean([r[(c, sc)] for r in rows]):13.3f}" for c in CFGS))
