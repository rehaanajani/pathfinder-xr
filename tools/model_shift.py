"""Learned-model calibration, window correlation and distribution shift (issue #2 E).

    python tools/model_shift.py [--model models/dispnet_faults/best.pt] [--tlio-records 6] [--seq room1]

Offline ANALYSIS (never runtime): windows of 1 s every 0.1 s, as the filter uses them.

* in-domain: TLIO validation records available locally (world-frame, bias-compensated IMU, as in
  training); target = reference displacement over the window;
* cross-device: TUM-VI ``--seq`` raw BMI160 IMU rotated into the gravity-aligned mocap frame with the
  *reference* orientation (diagnostic only: at runtime the filter's own orientation is used, which adds
  orientation error on top of what is measured here); not bias compensated (TLIO data are).

For each domain: displacement RMSE, NEES (3 dof) of the network's own diagonal covariance, coverage at
50/90/99 %, the variance scale that would make mean NEES = 3, and the autocorrelation of window errors
at lags 0.1-1 s. Overlapping windows share 90 % of their samples, so their errors are strongly
correlated; the filter's independent-update assumption then overstates the information by roughly the
correlation time / update interval, one reason large covariance multipliers (x1600 in issue #1) can
look better. Nothing here changes a default: `model_covariance_validated` stays False.
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from scipy.stats import chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.provenance import run_bundle  # noqa: E402

W, EVERY = 200, 20


def predict_all(pred, X):
    import torch
    out_d, out_s = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.from_numpy(X[i:i + 512].transpose(0, 2, 1).astype(np.float32))
            d, ls = pred.net(x)
            out_d.append(d.numpy())
            out_s.append(np.maximum(np.exp(ls.numpy()), pred.min_std))
    return np.concatenate(out_d), np.concatenate(out_s)


def analyse(pred, imu_w, pos, segments):
    """imu_w (N, 6) world-frame, pos (N, 3); segments: list of index ranges (contiguous, 200 Hz)."""
    Xs, Ys, seg_id = [], [], []
    for k, (a, b) in enumerate(segments):
        for s in range(a, b - W - 1, EVERY):
            Xs.append(imu_w[s:s + W])
            Ys.append(pos[s + W] - pos[s])
            seg_id.append(k)
    X, Y, seg_id = np.array(Xs), np.array(Ys), np.array(seg_id)
    d, sd = predict_all(pred, X)
    e = d - Y
    nees = np.sum((e / sd) ** 2, 1)
    lags = {}
    for L in range(1, 11):
        num, den = [], []
        for k in np.unique(seg_id):
            ek = e[seg_id == k]
            if len(ek) > L + 10:
                ek = ek - ek.mean(0)
                num.append(np.sum(ek[:-L] * ek[L:]))
                den.append(np.sqrt(np.sum(ek[:-L] ** 2) * np.sum(ek[L:] ** 2)))
        lags[f"{L * EVERY / 200:.1f}s"] = float(np.sum(num) / np.sum(den)) if den else None
    rho = np.array([v for v in lags.values() if v is not None])
    tau = 1 + 2 * float(np.sum(rho[rho > 0]))           # integrated autocorrelation (in update intervals)
    return {"windows": int(len(e)), "rmse_m": float(np.sqrt(np.mean(np.sum(e ** 2, 1)))),
            "rmse_per_axis_m": np.sqrt(np.mean(e ** 2, 0)).tolist(),
            "mean_claimed_std_m": float(np.mean(np.sqrt(np.sum(sd ** 2, 1)))),
            "nees_mean": float(nees.mean()), "nees_median": float(np.median(nees)),
            "coverage": {str(c): float(np.mean(nees <= chi2.ppf(c, 3))) for c in (0.5, 0.9, 0.99)},
            "variance_scale_for_mean_nees_3": float(nees.mean() / 3.0),
            "variance_scale_for_median": float(np.median(nees) / chi2.ppf(0.5, 3)),
            "error_autocorrelation": lags, "integrated_autocorrelation_updates": tau,
            "note": "windows every 0.1 s overlap by 90 %; tau ~ how many consecutive updates carry one window's worth of independent information"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="models/dispnet_faults/best.pt")
    ap.add_argument("--tlio-records", type=int, default=6)
    ap.add_argument("--seq", default="room1")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from pathfinder import data_tlio as T
    from pathfinder import data_tumvi_raw as D
    from pathfinder.predictor import Predictor
    pred = Predictor(a.model)
    val = [s for s in T.split("val") if (T.ROOT / s / "imu0_resampled.npy").is_file()][: a.tlio_records]
    inputs = {"model": Path(a.model), "tumvi_imu": D.seq_dir(a.seq) / "mav0/imu0/data.csv",
              "tumvi_mocap_analysis_only": D.seq_dir(a.seq) / "mav0/mocap0/data.csv"}
    with run_bundle("model-shift", vars(a), inputs, protocol="analysis") as run:
        res = {"model": a.model, "tlio_val_records": val, "label": "offline analysis; reference orientation used to "
                                                                     "rotate TUM-VI IMU (diagnostic only)"}
        imus, poss, segs, off = [], [], [], 0
        for sid in val:
            dd = np.load(T.ROOT / sid / "imu0_resampled.npy")
            imus.append(np.c_[dd[:, 1:4], dd[:, 4:7]])
            poss.append(dd[:, 11:14])
            segs.append((off, off + len(dd)))
            off += len(dd)
        res["tlio_val_in_domain"] = analyse(pred, np.concatenate(imus), np.concatenate(poss), segs)
        s = D.load_streams(a.seq)
        ref = D.load_reference(a.seq)
        t = s.imu_t
        ok = (t > ref.t[0]) & (t < ref.t[-1])
        j = np.clip(np.searchsorted(ref.t, t), 1, len(ref.t) - 1)
        gap_ok = ok & ((ref.t[j] - ref.t[j - 1]) < 0.02)
        from scipy.spatial.transform import Slerp
        Rw = Slerp(ref.t, Rot.from_quat(ref.q))(np.clip(t, ref.t[0], ref.t[-1]))
        p = np.stack([np.interp(t, ref.t, ref.p[:, i]) for i in range(3)], 1)
        imu_w = np.c_[Rw.apply(s.gyro), Rw.apply(s.acc)]
        # contiguous runs with reference support
        segs2, start = [], None
        for i, g in enumerate(gap_ok):
            if g and start is None:
                start = i
            if (not g or i == len(gap_ok) - 1) and start is not None:
                if i - start > W + 50:
                    segs2.append((start, i))
                start = None
        res[f"tumvi_{a.seq}_cross_device"] = analyse(pred, imu_w, p, segs2)
        dom = res["tlio_val_in_domain"], res[f"tumvi_{a.seq}_cross_device"]
        res["shift_summary"] = {"rmse_ratio_cross_over_in": dom[1]["rmse_m"] / dom[0]["rmse_m"],
                                "nees_mean_in": dom[0]["nees_mean"], "nees_mean_cross": dom[1]["nees_mean"]}
        run.write_json("model_shift.json", res)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("windows", "rmse_m", "mean_claimed_std_m", "nees_mean",
                                                                        "coverage", "variance_scale_for_mean_nees_3",
                                                                        "integrated_autocorrelation_updates")}
                          for k, v in res.items() if isinstance(v, dict) and "windows" in v}, indent=1))
        print(res["shift_summary"])
        print("run:", run.path)


if __name__ == "__main__":
    main()
