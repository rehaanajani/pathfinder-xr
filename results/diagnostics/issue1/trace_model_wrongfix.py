"""Trace camera_wrong_fix (benchmark params) with the REAL network (models-v1 dispnet_faults)."""
import sys

import numpy as np

sys.path.insert(0, '.')
import torch

from pathfinder import pipeline
from pathfinder.faults import scenarios
from pathfinder.filter import ESKF
from pathfinder.predictor import Predictor
from pathfinder.synthetic import head_walk

torch.set_num_threads(2)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
name = sys.argv[2] if len(sys.argv) > 2 else "camera_wrong_fix"
rec = []
orig = ESKF.camera_update


def cu(self, fix, h, log, t=0.0):
    sd = float(np.sqrt(np.trace(self.P[6:9, 6:9])))
    r = orig(self, fix, h, log, t)
    rec.append((round(t, 2), r, round(sd, 3), round(self.last_nis, 1), round(h, 2),
                None if self.episode is None else self.episode.origin, round(self.model_scale, 2)))
    return r


ESKF.camera_update = cu
model = Predictor("models/dispnet_faults/best.pt")
s = head_walk(60, n_imus=3, seed=seed)
r = pipeline.run_methods(s, scenarios(10.0)[name], seed=seed, ours_model=model, methods=["ours"])["ours"]
e = np.linalg.norm(r["p"] - s.p, axis=1)
print("seed", seed, name, "disp used/rej/skip", r["disp_used"], r["disp_rejected"], r["disp_skipped"])
for tt in (19.9, 20.5, 22, 25, 28, 29.9, 30.1, 31, 33, 36, 40, 45, 50, 59):
    i = int(tt * 200)
    row = [x for x in rec if abs(x[0] - round(tt, 1)) < 0.051]
    print(tt, "err", round(float(e[i]), 3), row[:1])
