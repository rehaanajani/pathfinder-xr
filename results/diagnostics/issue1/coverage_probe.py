"""Finding 5: does the reported position covariance cover the actual error? (scoring-only probe)

Baseline (ff7ec44): only a scalar p_std = sqrt(trace(P_pos)) is exported; we report the audit's
scalar statistic. New code: full 3x3 P_pos -> NEES / chi2(3) coverage.
Synthetic head_walk, 3 IMUs, seeds 0-3, shipped ours_cfg(), model = models/dispnet_faults/best.pt
(sha256 611E97FD... = release models-v1 dispnet_faults.pt) or none.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
import torch

from pathfinder import metrics, pipeline
from pathfinder.faults import scenarios
from pathfinder.predictor import Predictor
from pathfinder.synthetic import head_walk

torch.set_num_threads(2)
NEW = hasattr(metrics, "nees")
model = Predictor("models/dispnet_faults/best.pt")
SEG = {"before 5-20s": slice(1000, 4000), "fault 20-30s": slice(4000, 6000), "after 32-60s": slice(6400, 12000)}
for name in ["clean", "camera_blackout", "camera_dark", "camera_wrong_fix", "blackout_plus_saturation_all",
             "blackout_plus_one_imu_bias", "blackout_plus_vibration_all"]:
    for mname, m in (("model", model), ("no_model", None)):
        acc = {k: [] for k in SEG}
        for seed in range(4):
            s = head_walk(60, n_imus=3, seed=seed)
            meth = "ours" if m is not None else "ours_no_model"
            r = pipeline.run_methods(s, scenarios(10.0)[name], seed=seed, ours_model=m, methods=[meth])[meth]
            err = np.linalg.norm(r["p"] - s.p, axis=1)
            for lab, sl in SEG.items():
                if NEW:
                    nv = metrics.nees(r["p"][sl], s.p[sl], r["P_pos"][sl])
                    valid = r["valid"][sl]
                    vin = np.mean(nv[valid] <= 14.16) if valid.any() else np.nan
                    acc[lab].append((np.mean(nv <= 14.16), vin, err[sl].max(), np.mean(valid)))
                else:
                    sd = np.maximum(r["p_std"][sl], 1e-9)
                    acc[lab].append((np.mean(err[sl] <= 3 * sd), np.nan, err[sl].max(), 1.0))
        for lab in SEG:
            x = np.array(acc[lab])
            if NEW:
                print(f"{name:30s} {mname:8s} {lab:13s} inside chi2(3) 99.7% region {100*x[:,0].mean():5.1f}% "
                      f"| valid samples inside 99.7% {100*np.nanmean(x[:,1]):5.1f}% | max err {x[:,2].max():6.2f} m | valid {100*x[:,3].mean():5.1f}%")
            else:
                print(f"{name:30s} {mname:8s} {lab:13s} |e|<=3*sqrt(trP) {100*x[:,0].mean():5.1f}% "
                      f"| max err {x[:,2].max():6.2f} m")
