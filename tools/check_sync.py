"""Measure time offsets and frame conventions on a DEVELOPMENT recording (issue #2 B).

    python tools/check_sync.py --seq room1 [--max-offset-ms 50] [--out results/runs]

Nothing measured here is assumed by the runtime. The output records:

1. mocap <-> IMU: rotation increments of the mocap orientation over 50 ms windows against the gyro
   integrated over the same windows, for a grid of time offsets (0.25 ms steps). Reports the best
   offset, the residual and the rotation that best maps mocap increments onto gyro increments
   (identity = the mocap pose is the IMU pose, as the column names ``q_RS`` claim).
2. gravity: the mean specific force rotated into the mocap frame over the whole recording. Its
   angle from +z says whether the reference frame is gravity aligned (needed for 4-DoF alignment).
3. camera <-> IMU: rotation increments measured by the stereo visual front end (images only) against
   the gyro integrated over the same interval, for a grid of offsets. The camchain's missing
   ``timeshift_cam_imu`` is otherwise only an assumption.

These are measurements on development data, not fits on any final-test reference. The evaluation
protocol reads the stored result; it never re-fits on test recordings.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.provenance import run_bundle  # noqa: E402


def _interp_gyro(t_imu, gyro, t):
    return np.stack([np.interp(t, t_imu, gyro[:, i]) for i in range(3)], 1)


def mocap_rates(ref, max_dt=0.02):
    """Body angular velocity (rad/s) at mocap mid-times; skips mocap gaps."""
    R = Rot.from_quat(ref.q)
    dt = np.diff(ref.t)
    w = (R[:-1].inv() * R[1:]).as_rotvec() / dt[:, None]
    tm = 0.5 * (ref.t[1:] + ref.t[:-1])
    ok = (dt > 0) & (dt < max_dt) & np.isfinite(w).all(1)
    return tm[ok], w[ok]


def kabsch(a, b):
    """Rotation C minimising sum |C a_i - b_i|^2."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def offset_scan(cost, grid):
    c = np.array([cost(d) for d in grid])
    k = int(np.argmin(c))
    # parabolic refinement around the minimum
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = c[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        step = grid[1] - grid[0]
        best = grid[k] + (0.5 * (y0 - y2) / den * step if den > 0 else 0.0)
    else:
        best = grid[k]
    return float(best), float(c[k]), c


def _gyro_increments(s, t0, t1, d):
    """Integrated gyro (rad) over [t0, t1] with the IMU clock shifted by d. Integrating (instead of
    interpolating single samples) avoids a bias toward half-sample offsets: interpolating between two
    noisy samples halves their variance, which a residual minimum would otherwise prefer."""
    dt = np.diff(s.imu_t)
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (s.gyro[1:] + s.gyro[:-1]) * dt[:, None], 0)])
    tt = s.imu_t + d
    return np.stack([np.interp(t1, tt, cum[:, i]) - np.interp(t0, tt, cum[:, i]) for i in range(3)], 1)


def mocap_imu(s, ref, grid, span=0.05):
    """Rotation increments of the reference over ``span`` s windows against the integrated gyro."""
    R = Rot.from_quat(ref.q)
    j = np.searchsorted(ref.t, ref.t + span)
    ok = (j < len(ref.t))
    i = np.where(ok)[0]
    j = j[ok]
    seg_ok = np.array([np.all(np.diff(ref.t[a:b + 1]) < 0.02) for a, b in zip(i, j)])
    i, j = i[seg_ok], j[seg_ok]
    inner = (ref.t[i] > s.imu_t[0] + 0.1) & (ref.t[j] < s.imu_t[-1] - 0.1)
    i, j = i[inner][::2], j[inner][::2]
    t0, t1 = ref.t[i], ref.t[j]
    dR = (R[i].inv() * R[j]).as_rotvec()

    def cost(d):            # gyro sample time t_imu corresponds to reference time t_imu + d
        return float(np.mean(np.sum((_gyro_increments(s, t0, t1, d) - dR) ** 2, 1)))

    best, c, curve = offset_scan(cost, grid)
    g = _gyro_increments(s, t0, t1, best)
    C = kabsch(dR, g)
    resid = (g - dR) / (t1 - t0)[:, None]
    return {"offset_s": best, "convention": "reference time = imu time + offset_s",
            "window_s": span, "rms_increment_residual_deg": float(np.rad2deg(np.sqrt(c))),
            "rms_increment_at_zero_deg": float(np.rad2deg(np.sqrt(cost(0.0)))),
            "mocap_to_imu_rate_rotation_deg": float(np.rad2deg(np.linalg.norm(Rot.from_matrix(C).as_rotvec()))),
            "gyro_bias_estimate_rad_s": resid.mean(0).tolist(),
            "n_increments": int(len(t0)), "scan_ms": [float(grid[0] * 1e3), float(grid[-1] * 1e3)]}


def gravity(s, ref, offset):
    R = Rot.from_quat(ref.q)
    t = s.imu_t + offset
    ok = (t > ref.t[0]) & (t < ref.t[-1])
    j = np.clip(np.searchsorted(ref.t, t[ok]), 1, len(ref.t) - 1)
    gap = ref.t[j] - ref.t[j - 1]
    good = gap < 0.02
    Rw = R[j[good]]
    f_w = Rw.apply(s.acc[ok][good])
    m = f_w.mean(0)
    ang = float(np.rad2deg(np.arccos(np.clip(m[2] / np.linalg.norm(m), -1, 1))))
    return {"mean_specific_force_world": m.tolist(), "norm": float(np.linalg.norm(m)),
            "angle_from_plus_z_deg": ang, "n": int(good.sum()),
            "note": "mean world specific force over the recording = gravity reaction if mean acceleration ~ 0"}


def camera_imu(s, rig, events, grid):
    kf_t = {e["kf"]: e["t"] for e in events if e["kind"] == "kf"}
    rel = [e for e in events if e["kind"] == "rel" and e["kf"] in kf_t]
    ta = np.array([kf_t[e["kf"]] for e in rel])
    tb = np.array([e["t"] for e in rel])
    vis = np.array([Rot.from_matrix(e["dR"]).as_rotvec() for e in rel])
    # cumulative gyro integral (first order is enough for ~50 ms increments)
    dt = np.diff(s.imu_t)
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (s.gyro[1:] + s.gyro[:-1]) * dt[:, None], 0)])

    def integ(t0, t1):
        return np.stack([np.interp(t1, s.imu_t, cum[:, i]) - np.interp(t0, s.imu_t, cum[:, i]) for i in range(3)], 1)

    inner = (ta > s.imu_t[0] + 0.1) & (tb < s.imu_t[-1] - 0.1)
    ta, tb, vis = ta[inner], tb[inner], vis[inner]
    # Front-end times are t_cam + shift(file). Candidate d: true IMU time of the image = t + d.

    def cost(d):
        return float(np.mean(np.sum((integ(ta + d, tb + d) - vis) ** 2, 1)))

    best, c, _ = offset_scan(cost, grid)
    return {"offset_s": best, "convention": "imu time of image = front-end time + offset_s",
            "file_time_shift_s": rig.time_shift, "file_time_shift_source": rig.meta.get("time_shift_source"),
            "rms_increment_residual_deg": float(np.rad2deg(np.sqrt(c))),
            "rms_increment_at_zero_deg": float(np.rad2deg(np.sqrt(cost(0.0)))), "n_increments": int(len(ta))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", default="room1")
    ap.add_argument("--max-offset-ms", type=float, default=50.0)
    ap.add_argument("--step-ms", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="run-bundle root (default results/runs)")
    a = ap.parse_args(argv)
    grid = np.arange(-a.max_offset_ms, a.max_offset_ms + 1e-9, a.step_ms) * 1e-3
    s = D.load_streams(a.seq)
    ref = D.load_reference(a.seq)
    rig = RigCalibration.from_kalibr(s.calib_path)
    from pathfinder.vision_frontend import run_frontend
    kw = {"root": Path(a.out)} if a.out else {}
    inputs = {"imu": s.dir / "mav0/imu0/data.csv", "mocap": s.dir / "mav0/mocap0/data.csv",
              "cam_index": s.dir / "mav0/cam0/data.csv", "camchain": s.calib_path}
    with run_bundle(f"check-sync-{a.seq}", vars(a), inputs, protocol="dev-measurement", **kw) as run:
        out = {"sequence": a.seq, "split_role": "development (conventions/offset measurement only)",
               "conventions": D.check_conventions(a.seq)}
        out["mocap_imu"] = mocap_imu(s, ref, grid)
        out["gravity"] = gravity(s, ref, out["mocap_imu"]["offset_s"])
        fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], index_sha=D.index_sha256(a.seq))
        out["camera_imu"] = camera_imu(s, rig, fe["events"], grid)
        out["frontend_key"] = fe["key"]
        run.write_json("sync.json", out)
        print(json.dumps({k: out[k] for k in ("mocap_imu", "gravity", "camera_imu")}, indent=1))
        print("run:", run.path)


if __name__ == "__main__":
    main()
