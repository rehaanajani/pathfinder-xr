"""Builds demo/index.html: a self-contained page judges can open with no install.

Pick a failure (camera off for 0-20 s, dark room, a faulty IMU) and watch the true path,
naive fusion and ours on a real held-out headset recording, with a live readout of how much
the system trusts each sensor.
"""

import json
from pathlib import Path

import numpy as np

from pathfinder import data_tlio as D
from pathfinder.faults import Fault
from pathfinder.pipeline import run_methods
from pathfinder.predictor import Predictor

OUT = Path("demo")
STEP = 20            # 10 Hz in the page


def main():
    OUT.mkdir(exist_ok=True)
    ours = Predictor("models/dispnet_faults/best.pt")
    sid = D.split("test")[1]
    base = D.load(sid)
    start = 30 * 200
    seq = D.redundant_imus(base.slice(start, start + 70 * 200 + 1), 3, seed=0)
    cases = {}
    menu = {
        "Camera off 2 s": [Fault("camera_blackout", 20, 2)],
        "Camera off 5 s": [Fault("camera_blackout", 20, 5)],
        "Camera off 10 s": [Fault("camera_blackout", 20, 10)],
        "Camera off 20 s": [Fault("camera_blackout", 20, 20)],
        "Dark room 15 s": [Fault("camera_dark", 20, 15)],
        "Camera tracks the wrong thing 10 s": [Fault("camera_wrong_fix", 20, 10)],
        "One IMU dies + camera off 10 s": [Fault("imu_stuck", 15, 1e9, 0), Fault("camera_blackout", 20, 10)],
        "One IMU biased + camera off 10 s": [Fault("imu_bias", 15, 1e9, 0), Fault("camera_blackout", 20, 10)],
    }
    for label, faults in menu.items():
        r = run_methods(seq, faults, seed=1, ours_model=ours, methods=("naive_fusion", "ours"))
        cam = r["_cam"]
        o = r["ours"]
        w = o["imu_weights"]
        cases[label] = {
            "naive": np.round(r["naive_fusion"]["p"][::STEP, :2], 3).tolist(),
            "ours": np.round(o["p"][::STEP, :2], 3).tolist(),
            "naive_err": np.round(np.linalg.norm(r["naive_fusion"]["p"][::STEP] - seq.p[::STEP], axis=1), 3).tolist(),
            "ours_err": np.round(np.linalg.norm(o["p"][::STEP] - seq.p[::STEP], axis=1), 3).tolist(),
            "cam_health": np.round(np.where(np.isnan(cam["fix"][:, 0]), 0, cam["health"]), 2).tolist(),
            "imu_w": np.round(w, 2).tolist(),          # one value per 0.1 s, same clock as the page
            "ate": {"naive": round(r["naive_fusion"]["ate_m"], 2), "ours": round(o["ate_m"], 2)},
        }
    data = {"truth": np.round(seq.p[::STEP, :2], 3).tolist(), "t": np.round(seq.t[::STEP] - seq.t[0], 1).tolist(),
            "cases": cases, "seq": sid}
    html = (Path(__file__).parent / "demo_template.html").read_text().replace("__DATA__", json.dumps(data))
    (OUT / "index.html").write_text(html)
    print("wrote demo/index.html")


if __name__ == "__main__":
    main()
