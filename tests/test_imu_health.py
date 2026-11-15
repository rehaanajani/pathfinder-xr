"""Issue #1 sections B, C, E: IMU quality contract, causal calibration, geometry, invalid inputs.

Baseline ff7ec44 (baseline_evidence/p1_baseline.txt): pipeline confidence = w.max(0) was 1.000 in
every scenario including all-IMU noise and saturation; parity false alarms on clean data 2.1 %
(K=3) / 2.9 % (K=5) per window against a nominal 0.01 %; +-2 cm lever arms flagged 76 % of clean
windows as faulty.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from pathfinder import pipeline
from pathfinder.faults import Fault, apply_imu_faults, scenarios
from pathfinder.imu_health import (COMMON_MODE, ISOLATED, LOST, NOMINAL, SINGLE, UNISOLATED, WARMUP,
                                   ImuGeometry, ImuMonitor)
from pathfinder.synthetic import head_walk

W = 20   # samples per 0.1 s window


def windows(t0, t1):
    return slice(int(t0 * 10), int(t1 * 10))


def test_no_scalar_confidence_and_quality_is_informative():
    s = head_walk(30, n_imus=3, seed=1)
    out = {}
    for name in ("clean", "one_imu_bias", "imu_noise_all", "blackout_plus_saturation_all"):
        f = [Fault(x.kind, 12, 10, x.imu, x.params) for x in scenarios(10.0)[name] if x.kind.startswith("imu")]
        bad = apply_imu_faults(s, f, 0)
        fg, fa, info = pipeline.fuse_imus(bad)
        assert "confidence" not in info
        ws = info["window_state"][windows(13, 21)]
        out[name] = (ws, info["q_add_a"][13 * 200:21 * 200])
    assert np.all(out["clean"][0] == NOMINAL) and out["clean"][1].max() == 0
    assert np.all(out["one_imu_bias"][0] == ISOLATED) and out["one_imu_bias"][1].min() > 0
    assert np.mean(out["imu_noise_all"][0] != NOMINAL) > 0.9 and out["imu_noise_all"][1].mean() > 1e-3
    sat = out["blackout_plus_saturation_all"]
    assert np.mean(sat[0] == COMMON_MODE) > 0.5 and sat[1].max() >= 4.0


def test_common_mode_bias_is_not_observable_and_is_not_claimed():
    # Identical bias on every IMU: parity cannot see it. The monitor must stay NOMINAL (it has no
    # evidence) and must not report anything stronger than relative agreement.
    s = head_walk(30, n_imus=3, seed=2)
    bad = apply_imu_faults(s, [Fault("imu_bias", 12, 1e9, None, {"gyro": [0.02, 0, 0], "acc": [0.3, 0, 0]})])
    _, _, info = pipeline.fuse_imus(bad)
    assert np.all(info["window_state"][windows(13, 29)] == NOMINAL)


@pytest.mark.parametrize("K", [3, 5])
def test_clean_false_alarms_with_causal_calibration(K):
    # 8 seeds x 60 s = ~4600 tested windows per K. Nominal p_fa = 1e-4 per window gives an expected
    # 0.46 alarms; P(X >= 6 | Poisson(0.46)) < 1e-5. This bounds a gross miscalibration only: it
    # cannot verify 1e-4 itself (see docs/evidence/imu_false_alarm.json for the Clopper-Pearson bound).
    alarms = tested = 0
    for seed in range(8):
        s = head_walk(60, n_imus=K, seed=seed)
        _, _, info = ImuMonitor(K).run_batch(s.t, s.gyro, s.acc)
        c = info["calibrated"]
        alarms += int(info["detected"][c].sum())
        tested += int(c.sum())
    assert tested > 4000
    assert alarms <= 5, (alarms, tested)


def test_two_imus_detect_but_do_not_attribute():
    s = head_walk(20, n_imus=2, seed=0)
    bad = apply_imu_faults(s, [Fault("imu_bias", 8, 1e9, 0, {"gyro": [0.1, 0, 0], "acc": [1.0, 0, 0]})])
    _, _, info = pipeline.fuse_imus(bad)
    ws = info["window_state"][windows(9, 19)]
    assert np.all(ws == UNISOLATED)
    assert np.allclose(info["weights"][:, windows(9, 19)], 0.5)
    assert info["q_add_a"][9 * 200:].min() > 0.1        # the unattributed 1 m/s^2 disagreement


def test_fault_present_from_power_on_is_not_learned_as_healthy():
    # IMU 1 carries a bias beyond the datasheet prior from t = 0: it is excluded at calibration.
    s = head_walk(20, n_imus=3, seed=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 0, 1e9, 1, {"gyro": [0.05, 0, 0], "acc": [0.8, 0, 0]})])
    fg, _, info = pipeline.fuse_imus(bad)
    assert info["permanent_exclusions"] == {1: "calibration_offset_exceeds_prior"}
    assert np.abs(fg[1000:] - s.gyro[0, 1000:]).mean() < 0.01


def test_fault_starting_inside_warmup_restarts_calibration():
    s = head_walk(20, n_imus=3, seed=4)
    bad = apply_imu_faults(s, [Fault("imu_bias", 1.2, 1e9, 2, {"gyro": [0.01, 0, 0], "acc": [0.15, 0, 0]})])
    _, _, info = pipeline.fuse_imus(bad)
    # Either the shift is detected in warm-up (restart) or it is caught right after calibration;
    # what must not happen is a calibration that averages healthy and faulty data and then calls
    # the faulty sensor nominal for the rest of the run.
    late = info["window_state"][windows(8, 19)]
    assert info["restarts"] >= 1 or np.all(late == ISOLATED)
    assert np.all(info["weights"][2, windows(8, 19)] == 0) or info["restarts"] >= 1


def test_warmup_and_single_imu_are_reported_unverified():
    s = head_walk(5, n_imus=3, seed=0)
    _, _, info = ImuMonitor(3).run_batch(s.t, s.gyro, s.acc)
    assert info["window_state"][0] == WARMUP and not info["calibrated"][0]
    s1 = head_walk(5, n_imus=1, seed=0)
    _, _, info1 = ImuMonitor(1).run_batch(s1.t, s1.gyro, s1.acc)
    assert set(np.unique(info1["window_state"])) <= {SINGLE}


def _rigid_offsets(s, arms, rots=None):
    """Exact rigid-body readings of IMUs at offsets r_k with mounting rotations R_bs[k] (synthetic)."""
    out = s.copy()
    rot = R.from_quat(s.q)
    wb = np.zeros_like(s.p)
    wb[:-1] = (rot[:-1].inv() * rot[1:]).as_rotvec() / s.dt
    wb[-1] = wb[-2]
    wd = np.gradient(wb, s.dt, axis=0)
    for k, r in enumerate(arms):
        r = np.asarray(r, float)
        a = s.acc[k] + np.cross(wb, np.cross(wb, r)) + np.cross(wd, r)
        g = s.gyro[k]
        if rots is not None:
            Rsb = rots[k].T                                   # body -> sensor
            a, g = a @ Rsb.T, g @ Rsb.T
        out.acc[k], out.gyro[k] = a, g
    return out


@pytest.mark.parametrize("arm", [0.02, 0.05])
def test_lever_arms_and_mounting_rotations_need_geometry(arm):
    arms = [[arm, 0, 0], [-arm, 0, 0], [0, arm, 0]]
    rots = [np.eye(3), R.from_euler("z", 90, degrees=True).as_matrix(), R.from_euler("x", 180, degrees=True).as_matrix()]
    alarms_geo, alarms_naive, tested = 0, 0, 0
    for seed in range(3):
        s = _rigid_offsets(head_walk(40, n_imus=3, seed=seed), arms, rots)
        geo = ImuGeometry(np.array(rots), np.array(arms, float))
        _, _, ig = ImuMonitor(3, geometry=geo).run_batch(s.t, s.gyro, s.acc)
        _, _, ib = ImuMonitor(3).run_batch(s.t, s.gyro, s.acc)
        c = ig["calibrated"]
        alarms_geo += int(ig["detected"][c].sum())
        tested += int(c.sum())
        alarms_naive += int((ib["window_state"] != NOMINAL).sum())
        assert not geo.validated
    assert alarms_geo / tested < 0.01, alarms_geo / tested
    assert alarms_naive > 0.3 * tested                         # co-located assumption breaks


def test_lever_arm_geometry_still_isolates_a_fault():
    arms = [[0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0]]
    s = _rigid_offsets(head_walk(20, n_imus=3, seed=1), arms)
    bad = apply_imu_faults(s, [Fault("imu_bias", 8, 1e9, 0, {"gyro": [0.02, 0, 0], "acc": [0.3, 0, 0]})])
    _, _, info = ImuMonitor(3, geometry=ImuGeometry(np.repeat(np.eye(3)[None], 3, 0), np.array(arms))).run_batch(
        bad.t, bad.gyro, bad.acc)
    assert np.mean(info["weights"][0, windows(9, 19)] == 0) > 0.95


def test_streaming_equals_batch_and_is_causal():
    s = head_walk(12, n_imus=3, seed=5)
    N = len(s.t) - 7                                           # incomplete final window
    m = ImuMonitor(3)
    rows = []
    for i in range(N):
        rows += m.push(s.t[i], s.gyro[:, i], s.acc[:, i])
    rows += m.flush()
    fg, fa, info = ImuMonitor(3).run_batch(s.t[:N], s.gyro[:, :N], s.acc[:, :N])
    assert len(rows) == N and np.allclose([r["gyro"] for r in rows], fg)
    assert info["window_n"][-1] == N % W
    # Corrupting the future must not change anything already emitted.
    g2 = s.gyro[:, :N].copy()
    g2[:, 1500:] += 5.0
    fg2, _, info2 = ImuMonitor(3).run_batch(s.t[:N], g2, s.acc[:, :N])
    assert np.array_equal(fg2[:1500], fg[:1500])
    assert np.array_equal(info2["state"][:1500], info["state"][:1500])


def test_rate_change_and_window_alignment():
    s = head_walk(12, n_imus=3, seed=6)
    keep = np.r_[np.arange(0, 1200), np.arange(1200, len(s.t), 2)]     # 200 Hz then 100 Hz
    t, g, a = s.t[keep], s.gyro[:, keep], s.acc[:, keep]
    fg, _, info = ImuMonitor(3).run_batch(t, g, a)
    assert np.array_equal(info["t"], t)
    w0 = info["window_t0"]
    assert np.allclose(np.diff(w0), 0.1, atol=1e-6)
    assert set(info["window_n"][:60]) == {20} and set(info["window_n"][60:-1]) == {10}
    assert np.all(info["window_state"][40:] == NOMINAL)


def test_invalid_inputs():
    m = ImuMonitor(3)
    with pytest.raises(ValueError):
        m.run_batch(np.arange(5) / 200, np.zeros((2, 5, 3)), np.zeros((2, 5, 3)))
    with pytest.raises(ValueError):
        ImuMonitor(3).run_batch(np.array([0, 0.01, 0.01]), np.zeros((3, 3, 3)), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        ImuMonitor(0)
    fg, fa, info = ImuMonitor(3).run_batch(np.zeros(0), np.zeros((3, 0, 3)), np.zeros((3, 0, 3)))
    assert fg.shape == (0, 3) and len(info["window_state"]) == 0


def test_nonfinite_stale_and_all_lost():
    s = head_walk(12, n_imus=3, seed=7)
    g, a = s.gyro.copy(), s.acc.copy()
    g[0, 1000:1100] = np.nan                                  # one sensor non-finite
    g[1, 1400:1500], a[1, 1400:1500] = g[1, 1399], a[1, 1399]  # one sensor repeats its last packet
    g[:, 1800:1900] = np.nan                                  # everything lost
    fg, fa, info = ImuMonitor(3).run_batch(s.t, g, a)
    ws = info["window_state"]
    assert np.all(ws[50:55] == ISOLATED) and np.all(np.isfinite(fg[1000:1100]))
    assert np.all(ws[70:75] == ISOLATED)
    assert np.all(ws[90:95] == LOST) and np.all(np.isnan(fg[1800:1900]))
    assert info["q_add_a"][1800:1900].min() >= 4.0
    assert np.all(ws[100:] == NOMINAL)                        # recovery
