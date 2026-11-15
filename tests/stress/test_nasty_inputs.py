"""Nasty inputs: non-finite values, degenerate IMU data, extreme sizes and rates, hostile camera fixes.

Expected behaviour (what a tracking service must do): never raise, never report garbage as valid,
and tell the caller when it cannot be trusted.
"""
import numpy as np
import pytest

from harness import check_invariants, make_case, run_pair
from pathfinder.faults import Fault
from pathfinder.pipeline import ours_cfg
from pathfinder.stream import StreamingTracker
from pathfinder.synthetic import head_walk


def _stream(s, g, a, fixes=(), conf=None, cfg=None):
    trk = StreamingTracker(cfg or ours_cfg(), s.p[0], s.q[0], s.v[0], rate=1 / s.dt)
    fx = {int(k): (f, h) for k, f, h in fixes}
    outs = []
    for k in range(len(g)):
        if k in fx:
            trk.push_camera_fix(*fx[k])
        outs.append(trk.push_imu(g[k], a[k], 1.0 if conf is None else conf[k]))
    return trk, outs


@pytest.fixture(scope="module")
def walk():
    s = head_walk(20, n_imus=1, seed=1)
    fixes = [(k, s.p[k], 1.0) for k in range(0, len(s.t), 20)]
    return s, s.gyro[0].copy(), s.acc[0].copy(), fixes


# ---- non-finite values -------------------------------------------------------------------------
@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["gyro", "acc"])
@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#13")
def test_nonfinite_imu_sample_does_not_crash_stream(walk, val, where):
    s, g, a, fixes = walk
    g, a = g.copy(), a.copy()
    (g if where == "gyro" else a)[1000:1010] = val
    _, outs = _stream(s, g, a, fixes)
    assert np.isfinite(outs[-1]["position"]).all()


@pytest.mark.parametrize("n_imus", [1, 2])
@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#13")
def test_nan_burst_in_one_or_two_imus_through_pipeline(n_imus):
    def mut(b):
        b.gyro[0, 1000:1100] = np.nan
        b.acc[0, 1000:1100] = np.nan
        return b
    r = run_pair(make_case(2, 20, 200, n_imus), mutate=mut)
    assert not check_invariants(r)


@pytest.mark.parametrize("n_imus", [3, 5])
def test_nan_burst_with_three_or_more_imus_is_handled(n_imus):
    """Documented strength: parity masks unusable IMUs. Must keep working."""
    def mut(b):
        b.gyro[0, 1000:1200] = np.nan
        b.acc[0, 1000:1200] = np.inf
        return b
    r = run_pair(make_case(2, 20, 200, n_imus), mutate=mut)
    assert not check_invariants(r)
    err = np.linalg.norm(r["live"] - r["seq"].p, axis=1)
    assert err.max() < 0.5


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#13")
def test_huge_imu_values_do_not_crash(walk):
    s, g, a, fixes = walk
    g, a = g.copy(), a.copy()
    g[1000:1020] = 1e30
    _, outs = _stream(s, g, a, fixes)
    assert np.isfinite(outs[-1]["position"]).all()


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#14")
def test_nan_camera_health_does_not_crash(walk):
    s, g, a, _ = walk
    fixes = [(k, s.p[k], np.nan) for k in range(0, len(s.t), 20)]
    _, outs = _stream(s, g, a, fixes)
    assert all(np.isfinite(o["position"]).all() for o in outs)


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#14")
def test_inf_camera_fix_does_not_crash(walk):
    s, g, a, _ = walk
    fixes = [(k, np.array([np.inf, 0, 0]), 1.0) for k in range(100, len(s.t), 20)]
    _, outs = _stream(s, g, a, fixes)
    assert all(np.isfinite(o["position"]).all() for o in outs)


@pytest.mark.parametrize("health", [0.0, -1.0, 5.0])
def test_out_of_range_health_is_survivable(walk, health):
    s, g, a, _ = walk
    fixes = [(k, s.p[k], health) for k in range(0, len(s.t), 20)]
    _, outs = _stream(s, g, a, fixes)
    assert all(np.isfinite(o["position"]).all() for o in outs)


# ---- degenerate IMU data -----------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#15")
def test_all_zero_imu_is_not_reported_valid_when_wrong(walk):
    """A dead IMU (all zeros) with no camera: the walker moves ~1.2 m/s, the tracker thinks gravity
    is gone. It must not claim a trustworthy position."""
    s, g, a, _ = walk
    _, outs = _stream(s, np.zeros_like(g), np.zeros_like(a), fixes=())
    err = np.linalg.norm(np.array([o["position"] for o in outs]) - s.p, axis=1)
    bad = [(k, e) for k, (e, o) in enumerate(zip(err, outs)) if e > 1.0 and o["valid"]]
    assert not bad, f"{len(bad)} samples valid with error > 1 m, first at {bad[0][0] / 200:.1f}s ({bad[0][1]:.1f} m)"


def test_constant_imu_three_imus_pipeline():
    def mut(b):
        b.gyro[:] = 0.1
        b.acc[:] = 9.81
        return b
    r = run_pair(make_case(2, 20, 200, 3), mutate=mut)
    assert not check_invariants(r)


# ---- sizes and rates -----------------------------------------------------------------------------
def test_zero_length_input_is_rejected_cleanly():
    with pytest.raises((ValueError, IndexError)):
        run_pair(make_case(3, 0.0, 200, 3))


def test_two_sample_run():
    r = run_pair(make_case(3, 0.005, 200, 3))
    assert not check_invariants(r)


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#19")
def test_single_sample_sequence_through_prepare():
    from pathfinder.pipeline import prepare
    s = head_walk(1.0, n_imus=1, seed=1).slice(0, 1)
    prepare(s, [])


def test_single_sample_stream():
    s = head_walk(1.0, n_imus=1, seed=1)
    trk = StreamingTracker(ours_cfg(), s.p[0], s.q[0], s.v[0])
    out = trk.push_imu(s.gyro[0, 0], s.acc[0, 0])
    assert np.isfinite(out["position"]).all() and out["t"] == 0.0


@pytest.mark.parametrize("rate", [100, 400, 1000])
def test_other_imu_rates(rate):
    c = make_case(3, 12, rate, 3, [Fault("camera_blackout", 5, 3)])
    r = run_pair(c, check_every=25)
    assert not check_invariants(r), (rate, check_invariants(r))
    assert r["rt_factor"] > 1.0


@pytest.mark.slow
def test_ten_minute_run():
    c = make_case(7, 600, 200, 3, [Fault("camera_blackout", 100, 60), Fault("imu_bias", 200, 1e9, 0)])
    r = run_pair(c, check_every=200)
    assert not check_invariants(r), check_invariants(r)
    err = np.linalg.norm(r["live"] - r["seq"].p, axis=1)
    assert np.isfinite(err).all()


# ---- hostile camera --------------------------------------------------------------------------------
def test_camera_jump_100m_mid_run_is_held_out(walk):
    """A healthy-looking fix suddenly 100 m away is held out (suspect state) and the flag says so:
    once the estimate is > 3 m from the truth, valid must be False (after a 4 s grace)."""
    s, g, a, _ = walk
    fixes = [(k, s.p[k] + (np.array([100.0, 0, 0]) if k >= 2000 else 0), 1.0) for k in range(0, len(s.t), 20)]
    _, outs = _stream(s, g, a, fixes)
    pos = np.array([o["position"] for o in outs])
    err = np.linalg.norm(pos - s.p, axis=1)
    held = slice(2000 + 800, 2000 + 5000)                 # while the hold-out is active (30 s in the code)
    assert err[held].max() < 40.0, "estimate followed the bad camera during the hold-out"
    assert not any(o["valid"] for o, e in zip(outs[held], err[held]) if e > 3.0)


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#17")
def test_stream_equals_run_at_nominal_rate():
    """Fuzz case 10: stream built with the nominal 1/rate instead of the offline median dt."""
    from fuzz import random_case
    r = run_pair(random_case(10), exact_dt=False)
    assert r["max_diff"] <= 1e-9


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#16")
def test_saturation_mask_is_causal_in_first_two_seconds():
    """mask(x)[i] must not depend on samples after i (offline run vs streaming)."""
    from pathfinder.filter import saturation_mask
    x = np.random.default_rng(0).normal(0, 1, (1000, 6))
    x[100:110, 3] = 5.0
    x[200, 3] = 9.0
    full = saturation_mask(x[:, :3], x[:, 3:], 200.0)
    prefix = np.array([saturation_mask(x[:k + 1, :3], x[:k + 1, 3:], 200.0)[-1] for k in range(1000)])
    assert (full == prefix).all()


@pytest.mark.xfail(strict=False, reason="measured limit, STRESS_REPORT: frozen camera pose trusted (val: valid for up to 57 s, 11-29 m off)")
def test_identical_camera_fixes_are_not_trusted_forever(walk):
    """Camera frozen on one position while the walker moves: tracker must lose validity."""
    s, g, a, _ = walk
    fixes = [(k, s.p[0], 1.0) for k in range(0, len(s.t), 20)]
    _, outs = _stream(s, g, a, fixes)
    err = np.linalg.norm(np.array([o["position"] for o in outs]) - s.p, axis=1)
    assert not any(e > 3.0 and o["valid"] for e, o in zip(err, outs))


# ---- never-had-vision ------------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#15")
def test_no_camera_ever_still_widens_uncertainty(walk):
    """No fix has ever arrived: the honest-uncertainty floor (0.15 m/s/axis) must still apply."""
    s, g, a, _ = walk
    trk, outs = _stream(s, g, a, fixes=())
    t_blind = 15.0
    std = outs[int(t_blind * 200)]["position_std_m"]
    assert std >= np.sqrt(3) * 0.15 * (t_blind - 1) * 0.9, f"std {std:.2f} m after {t_blind}s with no camera ever"
