"""Pretends to be the phone app (for testing the live demo without a phone): streams a synthetic head
walk in the app's exact line format, with ARCore losing tracking for 10 s ("camera covered").

    python tools/live_demo/fake_phone.py [--speed 4]
"""
import argparse, json, socket, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pathfinder.synthetic import head_walk  # noqa: E402

Z_UP_TO_Y_UP = R.from_matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

ap = argparse.ArgumentParser(); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--seconds", type=float, default=40)
args = ap.parse_args()
s = head_walk(args.seconds, n_imus=1, seed=3)
rng = np.random.default_rng(0)
t_imu = np.arange(0, s.t[-1], 1 / 400.0)
g = np.stack([np.interp(t_imu, s.t, s.gyro[0][:, i]) for i in range(3)], 1)
a = np.stack([np.interp(t_imu, s.t, s.acc[0][:, i]) for i in range(3)], 1)
t_ar = np.arange(0, s.t[-1], 1 / 30.0)
sock = socket.create_connection(("127.0.0.1", 7777))
ev = sorted([(t, "g", i) for i, t in enumerate(t_imu)] + [(t, "a", i) for i, t in enumerate(t_imu)] +
            [(t, "ar", i) for i, t in enumerate(t_ar)])
w0, base = time.time(), 10**12
for t, k, i in ev:
    d = t / args.speed - (time.time() - w0)
    if d > 0: time.sleep(d)
    ns = base + int(t * 1e9)
    if k in ("g", "a"):
        v = (g if k == "g" else a)[i]
        msg = {"k": k, "t": ns, "v": v.tolist()}
    else:
        j = min(int(round(t * 200)), len(s.t) - 1)
        covered = 15 <= t < 25
        p = Z_UP_TO_Y_UP.apply(s.p[j] + rng.normal(0, 0.005, 3))
        q = (Z_UP_TO_Y_UP * R.from_quat(s.q[j])).as_quat()
        msg = {"k": "ar", "t": ns, "s": "PAUSED" if covered else "TRACKING",
               "r": "INSUFFICIENT_LIGHT" if covered else "NONE", "p": p.tolist(), "q": q.tolist(), "n": 0 if covered else 120}
    sock.sendall((json.dumps(msg) + "\n").encode())
sock.close()
