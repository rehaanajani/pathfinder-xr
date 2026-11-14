"""Issue #2 B / #1 D: the visual-odometry relative-pose update is mathematically right."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.filter import ESKF, FilterConfig, FilterLog, run, skew
from pathfinder.synthetic import head_walk


def vo_from_truth(s, every=10, std_t=0.003, std_r=0.002, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(0, len(s.t), every)
    Rg = R.from_quat(s.q)
    t, Rr = np.full((len(idx), 3), np.nan), np.tile(np.eye(3), (len(idx), 1, 1))
    for j in range(1, len(idx)):
        a, b = idx[j - 1], idx[j]
        t[j] = Rg[a].as_matrix().T @ (s.p[b] - s.p[a]) + rng.normal(0, std_t, 3)
        Rr[j] = (Rg[a].inv() * Rg[b] * R.from_rotvec(rng.normal(0, std_r, 3))).as_matrix()
    n = len(idx)
    return {"frame_idx": idx, "t": t, "R": Rr, "cov_t": np.tile(np.eye(3) * std_t ** 2, (n, 1, 1)),
            "cov_r": np.tile(np.eye(3) * std_r ** 2, (n, 1, 1))}


def test_vo_keeps_a_biased_imu_on_track():
    s = head_walk(60, n_imus=1, seed=1)
    bad = apply_imu_faults(s, [Fault("imu_bias", 0, 1e9, 0, {"gyro": [0.0, 0.0, 0.01], "acc": [0.1, 0, 0]})])
    imu_only, _ = run(s, bad.gyro[0], bad.acc[0], None, None, None, FilterConfig())
    est, log = run(s, bad.gyro[0], bad.acc[0], None, None, None, FilterConfig(), vo=vo_from_truth(s))
    e_imu = np.linalg.norm(imu_only[-1] - s.p[-1])
    e_vo = np.linalg.norm(est[-1] - s.p[-1])
    assert e_imu > 5 and e_vo < 0.5, (e_imu, e_vo)
    assert log.vo_used > 0.95 * (log.vo_used + log.vo_rejected)


def test_vo_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    f = ESKF(FilterConfig(), np.zeros(3), np.zeros(3), R.from_rotvec([0.1, -0.2, 0.3]).as_quat())
    f.add_clone(("vo", 0), 2)
    f.p = np.array([0.3, -0.1, 0.05])
    f.R = f.R * R.from_rotvec([0.05, 0.02, -0.04])
    i = 0
    def h(dx):
        Ri = (f.clone_R[i] * R.from_rotvec(dx[15:18])).as_matrix()
        pi = f.clone_p[i] + dx[18:21]
        Rc = (f.R * R.from_rotvec(dx[0:3])).as_matrix()
        p = f.p + dx[6:9]
        return np.r_[Ri.T @ (p - pi), R.from_matrix(Ri.T @ Rc).as_rotvec()]
    # Analytic H from the update code (residual y = z - h, so dh/dx = H).
    Ri, Rc = f.clone_R[i].as_matrix(), f.R.as_matrix()
    h_t, M = Ri.T @ (f.p - f.clone_p[i]), Ri.T @ Rc
    H = np.zeros((6, 21))
    H[0:3, 6:9], H[0:3, 18:21], H[0:3, 15:18] = Ri.T, -Ri.T, skew(h_t)
    H[3:6, 0:3], H[3:6, 15:18] = np.eye(3), -M.T
    num = np.zeros((6, 21))
    h0 = h(np.zeros(21))
    for j in list(range(0, 3)) + list(range(6, 9)) + list(range(15, 21)):
        d = np.zeros(21); d[j] = 1e-6
        diff = h(d) - h0
        diff[3:] = (R.from_rotvec(h0[3:]).inv() * R.from_rotvec(h(d)[3:])).as_rotvec()   # on-manifold
        num[:, j] = diff / 1e-6
    assert np.allclose(H, num, atol=1e-4), np.abs(H - num).max()
