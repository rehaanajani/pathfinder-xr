"""Compare learned-displacement formulations with the real network on synthetic walks (NOT validation data;
the network is trained on TLIO, so synthetic motion is out of domain). Reports error and chi2(3) coverage."""
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, '.')
import torch

from pathfinder import metrics, pipeline
from pathfinder.faults import scenarios
from pathfinder.filter import run
from pathfinder.predictor import Predictor
from pathfinder.synthetic import head_walk

torch.set_num_threads(2)
model = Predictor("models/dispnet_faults/best.pt")
base = pipeline.ours_cfg()
CFG = {"shipped (non-clone)": base,
       "clone, cov x1": replace(base, cloning=True),
       "clone, cov x10 (overlap)": replace(base, cloning=True, disp_cov_scale=10.0),
       "withhold if self-check > 10": replace(base, model_max_scale=10.0),
       "no model": None}
for name in sys.argv[1:] or ["clean", "camera_blackout", "camera_wrong_fix"]:
    for lab, cfg in CFG.items():
        rows = []
        for seed in range(3):
            s = head_walk(60, n_imus=3, seed=seed)
            bad, cam = pipeline.prepare(s, scenarios(10.0)[name], seed)
            fg, fa, info = pipeline.fuse_imus(bad)
            est, log, out = run(s, fg, fa, None, cam, None if cfg is None else model, cfg or base,
                                imu_quality=info["quality"], return_outputs=True)
            e = np.linalg.norm(est - s.p, axis=1)
            nv = metrics.nees(est, s.p, out["P_pos"])
            f = slice(4000, 6000)
            rows.append((metrics.ate(est, s.p), e[f].max(), np.mean(nv[f] <= 14.16), np.mean(nv[:4000] <= 14.16)))
        r = np.array(rows)
        print(f"{name:18s} {lab:26s} ATE {r[:,0].mean():.3f} | max err 20-30 s {r[:,1].mean():6.2f} | "
              f"in 99.7% region: 20-30 s {100*r[:,2].mean():5.1f}%  0-20 s {100*r[:,3].mean():5.1f}%")
