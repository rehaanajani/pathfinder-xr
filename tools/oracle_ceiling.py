"""Ceiling check: replace the velocity network by ground truth (body-frame velocity over the last 0.25 s) and score.

    python tools/oracle_ceiling.py --data euroc:MH_04_difficult euroc:V1_03_difficult --seeds 1 2 3 [-- main.py args]

Tells how much better the filter could do with a perfect network. Optional --noise adds Gaussian error (m/s) to
the oracle velocity, to see how good a network has to be before it moves the score.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402
import main as entry  # noqa: E402
import velnet  # noqa: E402

class Oracle:
    window, dec, rate, version, n_in = 200, 1, 200.0, 3, 10
    sd = 0.3

    def __init__(self, path):
        from scipy.spatial.transform import Rotation as R, Slerp
        gt = np.loadtxt(Oracle.gt, delimiter=",", comments="#")
        tg = gt[:, 0] * 1e-9
        ok = np.r_[True, np.diff(tg) > 0]
        self.gt, self.tg = gt[ok], tg[ok]
        self.rot = Slerp(self.tg, R.from_quat(self.gt[:, [5, 6, 7, 4]]))
        self.rng = np.random.default_rng(0)

    def pos(self, t):
        t = np.clip(t, self.tg[0], self.tg[-1])
        return np.array([np.interp(t, self.tg, self.gt[:, 1 + i]) for i in range(3)])

    def predict(self, ring, start):
        t = self.pf.t_prev * 1e-9
        r = self.rot(np.clip(t, self.tg[0], self.tg[-1]))
        v = r.inv().apply((self.pos(t) - self.pos(t - 0.25)) / 0.25)
        v = v + self.rng.normal(size=3) * Oracle.noise
        s = [max(Oracle.sd, Oracle.noise)] * 3
        return [0.0] * 3, [10.0] * 3, v.tolist(), s


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args(argv)
    Oracle.noise = a.noise
    velnet.VelNet = Oracle
    orig_init = entry.PathFinder.__init__

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        if self.net is not None:
            self.net.pf = self
    entry.PathFinder.__init__ = init
    rows = []
    work = ROOT / "runs" / "track3" / "oracle"
    for spec in a.data:
        src, gt = source(spec)
        Oracle.gt = gt
        for seed in a.seeds:
            d = work / f"{spec.replace(':', '_')}_s{seed}"
            subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d), "--seed", str(seed),
                            "--death", "zeros", "--dropout-mode", "missing"], check=True, capture_output=True)
            est = d / "estimated_trajectory.txt"
            entry.run(str(d), str(est), entry.args([str(d), "--model", "oracle", *extra]))
            res = score(est, gt, json.load(open(d / "truth.json")), d / "imu0" / "data.csv")
            res.update(data=spec, seed=seed)
            rows.append(res)
            print(spec, seed, {k: round(res[k], 2) for k in ("ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m")},
                  flush=True)
    keys = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m", "rpe_1s_m"]
    print("MEAN", {k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 2) for k in keys})


if __name__ == "__main__":
    main()
