"""Root-cause instrumentation for finding 1 (baseline code). Oracle displacement = diagnostic only."""
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, '.')
from pathfinder import pipeline
from pathfinder.faults import scenarios
from pathfinder.filter import ESKF, FilterConfig, run
from pathfinder.synthetic import head_walk

orig_cam = ESKF.camera_update
trace = []


def traced(self, fix, health, log, k=0):
    before = self.p.copy()
    y = fix - self.p
    Pp = self.P[6:9, 6:9].copy()
    orig_cam(self, fix, health, log, k)
    trace.append((k, float(np.linalg.norm(y)), float(np.sqrt(np.trace(Pp))), getattr(self, "q_scale", 1.0),
                  getattr(self, "last_nis", np.nan), float(np.linalg.norm(self.p - before)), log.cam_rejected,
                  log.cam_recovered))


ESKF.camera_update = traced


def oracle(s):
    def predictor(win):
        k = predictor.k
        predictor.k += 20
        return s.p[k] - s.p[k - 200], np.eye(3) * 0.05 ** 2
    predictor.k, predictor.window, predictor.every = 200, 200, 20
    return predictor


s = head_walk(60, n_imus=3, seed=0)
bad, cam = pipeline.prepare(s, scenarios(10.0)["camera_wrong_fix"], 0)
fg, fa, _ = pipeline.fuse_imus(bad)
for lab, cfg in [("FilterConfig()", FilterConfig()), ("ours_cfg()", pipeline.ours_cfg()),
                 ("ours_cfg adaptive_q=False", replace(pipeline.ours_cfg(), adaptive_q=False)),
                 ("ours_cfg bias_q_gain=1", replace(pipeline.ours_cfg(), bias_q_gain=1.0)),
                 ("FilterConfig(trust=TRUST)", FilterConfig(trust=pipeline.TRUST))]:
    trace.clear()
    est, log = run(s, fg, fa, None, cam, oracle(s), cfg)
    err = np.linalg.norm(est - s.p, axis=1)
    print(f"== {lab}: mean err 21-30 s {err[4200:6000].mean():.3f} m, recovered={log.cam_recovered}")
    for row in trace:
        if row[0] in (3800, 3980, 4000, 4020, 4100, 4200, 4400, 4600, 5000, 5400, 5980, 6000, 6200):
            print("  k=%5d |y|=%.3f sqrt(trP)=%.3f q=%.1f nis=%.1f move=%.4f rej=%d rec=%d" % row)
