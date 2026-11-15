"""Issue #1 section A: camera fault rejection and recovery, on the SHIPPED configuration.

Baseline ff7ec44 (reproduced, baseline_evidence/p4_baseline.txt and wrongfix_matrix_baseline.txt):
benchmark wrong fix with pipeline.ours_cfg() + oracle: mean error 0.80 m while the camera is wrong
(naive fusion 0.83 m); the old unit test passed only because it used FilterConfig().
"""

import numpy as np
import pytest

from pathfinder import pipeline
from pathfinder.faults import Fault
from pathfinder.filter import FilterConfig, run
from pathfinder.synthetic import head_walk

from .helpers import displacement_oracle, err, run_shipped

FS = 200


def wrong(start, dur, offset=(0.6, -0.4, 0.0), step=0.0):
    return Fault("camera_wrong_fix", start, dur, params={"offset": list(offset), "step": step})


def test_shipped_config_is_what_the_pipeline_uses():
    c = pipeline.ours_cfg()
    assert c.adaptive_q and c.bias_q_gain == 10.0 and c.trust is not None
    assert c.recover_after_jump_s == 10.0 and c.recover_after_s == 1.0


@pytest.mark.parametrize("dur", [5.0, 10.0, 20.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_persistent_wrong_fix_rejected_with_shipped_config(dur, seed):
    # The 20 s case is twice the 10 s jump hold-out: rejection must not rely on the hold-out alone.
    s = head_walk(25 + dur, n_imus=3, seed=seed)
    est, log, out, _ = run_shipped(s, [wrong(12, dur)], seed, displacement_oracle(s))
    e = err(est, s)
    a, b = 13 * FS, int((12 + dur) * FS)
    assert e[a:b].mean() < 0.05, e[a:b].mean()
    assert log.cam_recovered == 0
    assert np.any(out["camera_state"][a:b] == 4)          # suspect_jump was reported


def test_benchmark_wrong_fix_random_walk_rejected():
    s = head_walk(40, n_imus=3, seed=2)
    est, log, _, _ = run_shipped(s, [Fault("camera_wrong_fix", 15, 10)], 2, displacement_oracle(s))
    assert err(est, s)[16 * FS:25 * FS].mean() < 0.05


def test_adaptive_q_is_not_raised_by_rejected_fixes():
    s = head_walk(35, n_imus=3, seed=3)
    est, log, _, _ = run_shipped(s, [wrong(12, 15, offset=(2.0, 0, 0))], 3)
    q = np.array(log.q_scale)
    assert q[12 * FS:27 * FS].max() <= q[12 * FS - 1] + 1e-9   # only decays while fixes are rejected


@pytest.mark.parametrize("dur", [5.0, 10.0, 20.0])
def test_genuine_recovery_after_inertial_drift_without_model(dur):
    # Camera blackout longer than, equal to and shorter than the jump hold-out; no learned model, so
    # the inertial estimate drifts and the returning camera carries a genuine correction.
    s = head_walk(30 + dur, n_imus=3, seed=4)
    est, log, out, _ = run_shipped(s, [Fault("camera_blackout", 12, dur)], 4)
    e = err(est, s)
    back = int((12 + dur) * FS)
    assert e[back + 3 * FS:].max() < 0.15
    assert out["valid"][back + 3 * FS:].all()


def test_genuine_correction_with_bias_on_every_imu():
    # Common-mode accelerometer bias (unobservable to the IMU monitor) drives large drift during the
    # blackout; the returning camera must still be taken within seconds.
    s = head_walk(40, n_imus=3, seed=5)
    f = [Fault("camera_blackout", 12, 10), Fault("imu_bias", 8, 1e9, None, {"gyro": [0.01, 0, 0], "acc": [0.2, 0.1, 0]})]
    est, log, _, _ = run_shipped(s, f, 5)
    e = err(est, s)
    assert e[21 * FS:22 * FS].max() > 0.5                   # the drift is real
    assert e[26 * FS:].max() < 0.15


def test_repeated_wrong_fixes_no_model_do_not_take_over():
    # Regression for the first rewrite (baseline_evidence/wrongfix_matrix_attempt1_regression.txt):
    # covariance growth let the wrong fix in and the true camera was then held out as a "jump".
    s = head_walk(45, n_imus=3, seed=0)
    f = [wrong(a, 2, offset=(0.8, 0, 0)) for a in (12, 19, 26)]
    est, log, _, _ = run_shipped(s, f, 0)
    e = err(est, s)
    assert e[10 * FS:].max() < 0.5
    assert e[32 * FS:].max() < 0.1


def test_isolated_jump_and_gradual_drift():
    s = head_walk(35, n_imus=3, seed=1)
    est, _, _, _ = run_shipped(s, [wrong(12, 0.15, offset=(1.0, 0, 0))], 1)
    assert err(est, s)[11 * FS:].max() < 0.1
    est, _, _, _ = run_shipped(s, [wrong(12, 10, offset=(0, 0, 0), step=0.01)], 1, displacement_oracle(s))
    assert err(est, s)[12 * FS:22 * FS].max() < 0.25


def test_low_health_frames_never_reanchor():
    s = head_walk(35, n_imus=3, seed=2)

    def degrade(cam):
        cam = dict(cam)
        h = cam["health"].copy()
        t = cam["idx"] / FS
        m = (t >= 12) & (t < 27)
        h[m] = 0.3                                         # below recover_health
        fix = cam["fix"].copy()
        fix[m] += np.array([1.5, 0, 0])
        cam["health"], cam["fix"] = h, fix
        return cam

    est, log, out, _ = run_shipped(s, [], 2, cam_transform=degrade)
    assert log.cam_recovered == 0
    # Known limit (docs/ESTIMATOR.md): a bias far beyond the calibrated low-health error is accepted
    # once the state covariance grows; it is reported as degraded tracking, never as nominal.
    from pathfinder.estimator import TRACK_TRACKING
    assert not np.any(out["track_state"][13 * FS:27 * FS] == TRACK_TRACKING)
    assert err(est, s)[30 * FS:].max() < 0.1


@pytest.mark.parametrize("keep_every,jitter", [(1, 0.0), (2, 0.004), (3, 0.0)])
def test_holdout_is_in_seconds_under_irregular_and_dropped_frames(keep_every, jitter):
    # 10, 5 and 3.3 Hz cameras with jittered timestamps: the outage recovery takes ~recover_after_s
    # of camera time regardless of how many frames that is.
    s = head_walk(35, n_imus=3, seed=3)
    rng = np.random.default_rng(0)

    def thin(cam):
        sel = np.arange(len(cam["idx"]))[::keep_every]
        idx = cam["idx"][sel] + np.round(rng.uniform(-jitter, jitter, len(sel)) * FS).astype(int)
        idx = np.clip(idx, 0, len(s.t) - 1)
        return {"idx": idx, "fix": cam["fix"][sel], "health": cam["health"][sel]}

    est, log, out, _ = run_shipped(s, [Fault("camera_blackout", 12, 8)], 3, cam_transform=thin)
    ev = [e for e in log.events if e[1] == "reanchored"]
    starts = [e for e in log.events if e[1] == "episode_start"]
    if ev:
        assert ev[0][0] - starts[0][0] == pytest.approx(1.0, abs=0.35)
    assert err(est, s)[23 * FS:].max() < 0.15


def test_naive_fusion_still_fails_for_reference():
    s = head_walk(30, n_imus=1, seed=5)
    bad, cam = pipeline.prepare(s, [wrong(12, 10, offset=(3.0, 0, 0))])
    naive, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig(use_health=False, use_gate=False))
    assert np.linalg.norm(naive[18 * FS] - s.p[18 * FS]) > 2.0


def _biased_oracle(seq, bias=(1.0, 0.0, 0.0), std=0.05):
    """Diagnostic: a 'network' that is confidently wrong (error 20x its claimed std)."""
    p = displacement_oracle(seq, std)
    inner = p

    def predictor(win):
        d, c = inner(win)
        return d + np.asarray(bias), c
    predictor.window, predictor.every = p.window, p.every
    return predictor


def test_distrusted_model_cannot_veto_the_camera():
    # Reproduced with the models-v1 network on synthetic walks (ATE 4.8 m on camera_wrong_fix): a model that
    # the self-check already measured as inconsistent held the state covariance tight, and its drifting
    # estimate then rejected the true camera. During a jump episode such a model is now withheld.
    from pathfinder.estimator import REASONS
    s = head_walk(45, n_imus=3, seed=1)
    est, log, out, _ = run_shipped(s, [wrong(15, 10)], 1, _biased_oracle(s))
    assert log.disp_withheld > 0
    assert np.any(out["reasons"] & REASONS["MODEL_DISTRUSTED"])
    assert err(est, s)[30 * FS:].max() < 0.15


def test_distrusted_model_outage_is_not_declared_valid():
    from pathfinder.estimator import REASONS
    s = head_walk(35, n_imus=3, seed=2)
    est, log, out, _ = run_shipped(s, [Fault("camera_blackout", 15, 10)], 2, _biased_oracle(s))
    sl = slice(16 * FS, 25 * FS)
    assert not out["valid"][sl].any()
    assert np.all(out["reasons"][sl] & REASONS["COVARIANCE_UNRELIABLE"])
    assert out["valid"][28 * FS:].all()
