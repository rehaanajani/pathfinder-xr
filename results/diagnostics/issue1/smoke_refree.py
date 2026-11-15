import sys
import time

import numpy as np

sys.path.insert(0, '.')
from pathfinder import pipeline
from pathfinder.estimator import TRACK_NAMES, reason_names
from pathfinder.faults import Fault
from pathfinder.filter import run
from pathfinder.synthetic import head_walk

for seed in range(3):
    s = head_walk(40, n_imus=3, seed=seed)
    bad, cam = pipeline.prepare(s, [Fault("camera_blackout", 20, 5)], seed)
    fg, fa, info = pipeline.fuse_imus(bad)
    t0 = time.time()
    for init in ("truth", "reference_free"):
        est, log, out = run(s, fg, fa, None, cam, None, pipeline.ours_cfg(), imu_quality=info["quality"],
                            init=init, return_outputs=True)
        ok = out["valid"]
        e = np.linalg.norm(est - s.p, axis=1)
        first = int(np.argmax(ok))
        print(seed, init, "first valid t", s.t[first], "mean err valid", np.nanmean(e[ok]).round(3),
              "max err 20-25", np.nanmax(e[4000:5000]).round(3), "events", log.events[:2],
              "reasons@10s", reason_names(out["reasons"][2000]), TRACK_NAMES[out["track_state"][2000]],
              "secs", round(time.time() - t0, 1))
# IMU-only start (no camera at all)
s = head_walk(20, n_imus=1, seed=0)
est, log, out = run(s, s.gyro[0], s.acc[0], None, None, None, pipeline.ours_cfg(), init="reference_free",
                    return_outputs=True)
print("imu-only", log.events, reason_names(out["reasons"][-1]), TRACK_NAMES[out["track_state"][-1]],
      "valid frac", out["valid"].mean().round(3))
