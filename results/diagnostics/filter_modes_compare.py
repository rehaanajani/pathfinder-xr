"""Filter variants (position update, cloning, yaw frame, network-uncertainty calibration), IMU only, full TLIO test sequences.

    python results/diagnostics/filter_modes_compare.py [n_seqs] [model_path] [workers]   (env: SPLIT=val|test, MODES=old,cloning_x1600,...)
"""
import os
import sys
import numpy as np
sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor
from pathfinder import data_tlio as D, metrics
from pathfinder.filter import FilterConfig, run

W = {}
MODES = {"old": FilterConfig(), "cloning": FilterConfig(cloning=True),
         "yaw_frame": FilterConfig(cloning=True, yaw_frame=True)}
import os
for sc in (2.0, 4.0, 9.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 10000.0):
    MODES[f"old_x{sc:g}"] = FilterConfig(disp_cov_scale=sc)
    MODES[f"cloning_x{sc:g}"] = FilterConfig(cloning=True, disp_cov_scale=sc)
if os.environ.get("MODES"):
    MODES = {k: MODES[k] for k in os.environ["MODES"].split(",")}


def init(path):
    import torch; torch.set_num_threads(1)
    from pathfinder.predictor import Predictor
    W['p'] = Predictor(path)


def job(sid):
    s = D.load(sid)
    out = {}
    for name, cfg in MODES.items():
        e, _ = run(s, s.gyro[0], s.acc[0], None, None, W['p'], cfg)
        out[name] = metrics.ate(e, s.p, align=False)
    print(sid, {k: round(v, 2) for k, v in out.items()}, flush=True)
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    path = sys.argv[2] if len(sys.argv) > 2 else 'models/dispnet_faults/best.pt'
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 8, initializer=init, initargs=(path,)) as ex:
        rows = list(ex.map(job, D.split(os.environ.get('SPLIT', 'test'))[:n]))
    for k in MODES:
        v = [r[k] for r in rows]
        print(f'{k:10s} ATE raw mean {np.mean(v):.2f} m  median {np.median(v):.2f} m  (n={len(v)})')
