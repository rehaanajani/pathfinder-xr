"""Wrong-fix / recovery matrix with the shipped configuration. Oracle displacement is DIAGNOSTIC ONLY."""
import sys

import numpy as np

sys.path.insert(0, '.')
from pathfinder import pipeline
from pathfinder.faults import Fault
from pathfinder.filter import run
from pathfinder.synthetic import head_walk


def oracle(s, std=0.05):
    def predictor(win):
        k = predictor.k
        predictor.k += 20
        return s.p[k] - s.p[k - 200], np.eye(3) * std ** 2
    predictor.k, predictor.window, predictor.every = 200, 200, 20
    return predictor


CASES = {
    "wrong const 5s": [Fault("camera_wrong_fix", 20, 5, params={"offset": [0.6, -0.4, 0], "step": 0.0})],
    "wrong const 10s": [Fault("camera_wrong_fix", 20, 10, params={"offset": [0.6, -0.4, 0], "step": 0.0})],
    "wrong const 20s": [Fault("camera_wrong_fix", 20, 20, params={"offset": [0.6, -0.4, 0], "step": 0.0})],
    "wrong walk 10s (bench)": [Fault("camera_wrong_fix", 20, 10)],
    "gradual drift 10s": [Fault("camera_wrong_fix", 20, 10, params={"offset": [0, 0, 0], "step": 0.01})],
    "isolated jump 0.15s": [Fault("camera_wrong_fix", 20, 0.15, params={"offset": [1.0, 0, 0], "step": 0.0})],
    "repeated 3x2s": [Fault("camera_wrong_fix", a, 2, params={"offset": [0.8, 0, 0], "step": 0.0}) for a in (15, 22, 29)],
    "blackout 10s": [Fault("camera_blackout", 20, 10)],
    "blackout 10s + bias": [Fault("camera_blackout", 20, 10), Fault("imu_bias", 15, 1e9, None, {"gyro": [0.01, 0, 0], "acc": [0.2, 0.1, 0]})],
    "dark 10s": [Fault("camera_dark", 20, 10)],
}
for label, faults in CASES.items():
    for model in ("oracle", "none"):
        rows = []
        for seed in range(4):
            s = head_walk(60, n_imus=3, seed=seed)
            bad, cam = pipeline.prepare(s, faults, seed)
            fg, fa, info = pipeline.fuse_imus(bad)
            pr = oracle(s) if model == "oracle" else None
            if "quality" in info:     # new code
                est, log = run(s, fg, fa, None, cam, pr, pipeline.ours_cfg(), imu_quality=info["quality"])
            else:                     # baseline ff7ec44: the shipped pipeline passed confidence
                conf = np.repeat(info["confidence"], 20)[: len(s.t)]
                est, log = run(s, fg, fa, conf, cam, pr, pipeline.ours_cfg())
            e = np.linalg.norm(est - s.p, axis=1)
            fs = int(faults[0].start * 200)
            fe = int((faults[-1].start + faults[-1].duration) * 200)
            rows.append((e[fs:fe].mean(), e[fs:fe].max(), e[fe + 400:fe + 2000].mean() if fe + 2000 < len(e) else np.nan,
                         e[-2000:].mean(), log.cam_recovered))
        r = np.array(rows)
        print(f"{label:24s} {model:6s} fault mean {r[:,0].mean():.3f} max {r[:,1].max():.3f} | 2-10 s after {np.nanmean(r[:,2]):.3f} | last 10 s {r[:,3].mean():.3f} | reanchors {r[:,4].astype(int).tolist()}")
