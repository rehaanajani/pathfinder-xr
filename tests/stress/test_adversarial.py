"""Adversarial combinations, documented limits, and determinism."""
from dataclasses import replace

import numpy as np
import pytest

from harness import check_invariants, make_case, run_pair
from pathfinder.faults import Fault
from pathfinder.pipeline import fuse_imus, ours_cfg, prepare
from pathfinder.stream import StreamingTracker
from pathfinder.synthetic import head_walk


def _errors(r):
    return np.linalg.norm(r["live"] - r["seq"].p, axis=1)


def _valid_wrong(r, thr, grace_s=1.0):
    """Samples where the tracker says valid although it is more than thr metres wrong, ignoring the
    first grace_s seconds of each wrong stretch (the flag cannot react instantly)."""
    e = _errors(r)
    valid = np.array([o["valid"] for o in r["outs"]])
    wrong = (e > thr) & valid
    g = int(grace_s * r["case"]["rate"])
    run, n = 0, 0
    for w in wrong:
        run = run + 1 if w else 0
        n += run > g
    return n


# 1 -- slow wrong camera + IMU bias ------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_slow_drifting_wrong_camera_plus_imu_bias(seed):
    c = make_case(seed, 60, 200, 3, [Fault("camera_wrong_fix", 15, 40, None, {"step": 0.02, "offset": [0, 0, 0]}),
                                     Fault("imu_bias", 10, 1e9, 0)])
    r = run_pair(c, check_every=100)
    assert not check_invariants(r), check_invariants(r)
    # Documented limit: a slow wrong camera cannot be told from the truth. Never crash; the number of
    # samples that claim validity while >2 m wrong is reported by the STRESS_REPORT.
    assert _valid_wrong(r, 2.0) < 0.05 * len(r["outs"]), \
        f"valid=True for {_valid_wrong(r, 2.0) / 200:.1f} s while more than 2 m wrong"


# 2 -- camera flapping every 0.2 s ---------------------------------------------------------------------
def test_camera_good_bad_every_200ms():
    faults = [Fault("camera_blackout", 5 + 0.4 * i, 0.2) for i in range(35)]
    c = make_case(3, 25, 200, 3, faults)
    r = run_pair(c, check_every=50)
    assert not check_invariants(r), check_invariants(r)
    assert _errors(r).max() < 1.5, _errors(r).max()
    flags = np.array([o["tracked"] for o in r["outs"]])
    assert (np.diff(flags.astype(int)) != 0).sum() < 200, "tracked flag chatters"


def test_camera_alternating_good_and_wrong_every_200ms():
    faults = [Fault("camera_wrong_fix", 5 + 0.4 * i, 0.2, None, {"step": 0.0, "offset": [1.0, -1.0, 0]})
              for i in range(35)]
    c = make_case(3, 25, 200, 3, faults)
    r = run_pair(c, check_every=50)
    assert not check_invariants(r), check_invariants(r)
    assert _errors(r).max() < 2.0, _errors(r).max()


# 3 -- one-axis saturation ------------------------------------------------------------------------------
@pytest.mark.parametrize("all_imus", [False, True])
def test_one_axis_saturation(all_imus):
    def mut(b):
        idx = range(b.n_imus) if all_imus else [0]
        for k in idx:
            sl = slice(2000, 2800)
            b.acc[k, sl, 2] = np.clip(b.acc[k, sl, 2], -10.0, 10.0)
        return b
    c = make_case(5, 20, 200, 3, [Fault("camera_blackout", 9, 8)])
    r = run_pair(c, mutate=mut, check_every=50)
    assert not check_invariants(r), check_invariants(r)
    assert _valid_wrong(r, 3.0) == 0


# 4 -- two of three IMUs with the same fault (common-mode on the majority) --------------------------------
def test_two_of_three_imus_share_a_bias():
    bias = {"gyro": [0.03, -0.02, 0.01], "acc": [0.4, -0.3, 0.2]}
    f = [Fault("imu_bias", 5, 1e9, 0, bias), Fault("imu_bias", 5, 1e9, 1, bias),
         Fault("camera_blackout", 9, 10)]
    c = make_case(6, 25, 200, 3, f)
    r = run_pair(c, check_every=50)
    assert not check_invariants(r), check_invariants(r)
    # Two faulty copies outvote the healthy IMU (known limit). The flag must still react quickly:
    # measured 0.4 s beyond the 1 s grace on synthetic head_walk; allow 2 s.
    assert _valid_wrong(r, 2.0) < 400, "valid while >2 m wrong after a majority fault"


# 5 -- determinism -------------------------------------------------------------------------------------------
def _go(seed=11):
    c = make_case(seed, 15, 200, 3, [Fault("camera_blackout", 5, 4), Fault("imu_spikes", 6, 3, 0, {"rate": 0.05})])
    return run_pair(c, check_every=100)["live"]


def test_same_seed_same_output():
    assert np.array_equal(_go(), _go())


def test_run_a_crash_b_run_a():
    """A crash in one run must not change the next run (shared config objects, module-level state)."""
    a1 = _go()
    s = head_walk(5, n_imus=1, seed=2)
    trk = StreamingTracker(ours_cfg(), s.p[0], s.q[0], s.v[0])
    try:
        for k in range(300):
            trk.push_camera_fix(np.array([np.inf, np.inf, np.inf]), 1.0) if k % 20 == 0 else None
            trk.push_imu(s.gyro[0, k] * (np.nan if k == 250 else 1), s.acc[0, k])
    except Exception:                                    # noqa: BLE001
        pass
    assert np.array_equal(a1, _go())


def test_config_object_not_left_modified_after_exception():
    """NOT DEMONSTRATED as a bug (re-anchor path has no try/finally but we could not make it raise). The re-anchor and speed-check paths flip cfg.use_gate temporarily; an exception must not leave it off."""
    cfg = ours_cfg()
    assert cfg.use_gate
    s = head_walk(8, n_imus=1, seed=2)
    for fix in (np.array([np.inf, 0, 0]), np.array([1e300, 1e300, 1e300])):
        trk = StreamingTracker(cfg, s.p[0], s.q[0], s.v[0])
        try:
            for k in range(1200):
                if k % 10 == 0:
                    trk.push_camera_fix(fix if k > 400 else s.p[k], 1.0)
                trk.push_imu(s.gyro[0, k], s.acc[0, k])
        except Exception:                                # noqa: BLE001
            pass
        assert cfg.use_gate, "cfg.use_gate left False after an exception"
