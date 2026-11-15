"""Trace camera supervision decisions for one case. Oracle displacement = diagnostic only.
usage: python trace_episode.py <case> <seed> <oracle|none>"""
import sys

import numpy as np

sys.path.insert(0, '.')
from pathfinder import pipeline
from pathfinder.faults import Fault
from pathfinder.filter import ESKF, run
from pathfinder.synthetic import head_walk

CASES = {
    "c20": [Fault("camera_wrong_fix", 20, 20, params={"offset": [0.6, -0.4, 0], "step": 0.0})],
    "c10": [Fault("camera_wrong_fix", 20, 10, params={"offset": [0.6, -0.4, 0], "step": 0.0})],
    "walk": [Fault("camera_wrong_fix", 20, 10)],
}


def oracle(s, std=0.05):
    def predictor(win):
        k = predictor.k
        predictor.k += 20
        return s.p[k] - s.p[k - 200], np.eye(3) * std ** 2
    predictor.k, predictor.window, predictor.every = 200, 200, 20
    return predictor


case, seed, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rec = []
orig = ESKF.camera_update


def cu(self, fix, h, log, t=0.0):
    sd = float(np.sqrt(np.trace(self.P[6:9, 6:9])))
    r = orig(self, fix, h, log, t)
    rec.append((round(t, 2), r, round(sd, 3), round(self.last_nis, 1), round(h, 2),
                None if self.episode is None else self.episode.origin))
    return r


ESKF.camera_update = cu
s = head_walk(60, n_imus=3, seed=seed)
bad, cam = pipeline.prepare(s, CASES[case], seed)
fg, fa, info = pipeline.fuse_imus(bad)
est, log = run(s, fg, fa, None, cam, oracle(s) if mode == "oracle" else None, pipeline.ours_cfg(),
               imu_quality=info["quality"])
e = np.linalg.norm(est - s.p, axis=1)
print([(round(t, 2), k, o) for t, k, o in log.events][:20])
show = {float(x) for x in sys.argv[4:]} if len(sys.argv) > 4 else set()
for x in rec:
    if x[1] == "reanchored" or x[0] in show:
        print("  ", x, "err", round(float(e[min(int(x[0] * 200), len(e) - 1)]), 3))
