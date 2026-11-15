"""End-to-end smoke test on the generated synthetic fixture: loader -> image corruption -> stereo front
end -> streaming estimator -> scorer, the same code path as a real TUM-VI recording (issue #3 D).
Plumbing and contract checks only; synthetic results are not accuracy evidence."""

import numpy as np
import pytest

from pathfinder import data_tumvi_raw as D
from pathfinder import scoring
from pathfinder.camera_model import RigCalibration
from pathfinder.estimator import REASONS
from pathfinder.filter import CAM_NOMINAL, CAM_OUTAGE
from pathfinder.smoke import make_fixture
from pathfinder.vision_eval import Frozen, imu_noise_from_config, run_system, system_configs
from pathfinder.vision_frontend import run_frontend


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    make_fixture(root, duration=6.0)
    return root


def _run(root, schedule=(), system="pathfinder"):
    s = D.load_streams("smoke", root)
    ref = D.load_reference("smoke", root)
    rig = RigCalibration.from_kalibr(s.calib_path)
    fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], schedule=schedule, cache=False)
    fz = Frozen(imu_noise=imu_noise_from_config(s.dir / "dso/imu_config.yaml"))
    cfg, cam, mdl = (None, True, False) if system == "vo_only" else system_configs(fz)[system]
    out, diag = run_system(system, cfg, cam, mdl, s.imu_t, s.gyro, s.acc, fe["events"])
    res, series = scoring.score(out, ref, [{"start": f["start"], "end": f["end"], "kind": f["kind"]}
                                           for f in fe["schedule"]])
    return s, fe, out, diag, res


def test_conventions_and_loader(fixture_dir):
    info = D.check_conventions("smoke", fixture_dir)
    assert info["imu"]["non_increasing"] == 0 and info["frames"]["n"] == 60
    rig = RigCalibration.from_kalibr(D.load_streams("smoke", fixture_dir).calib_path)
    assert rig.meta["time_shift_source"] == "absent_assumed_zero"


def test_clean_end_to_end_is_finite_and_tracks(fixture_dir):
    s, fe, out, diag, res = _run(fixture_dir)
    assert fe["stats"]["rel"] >= 50 and diag["vo_used"] >= 45
    started = out["track_state"] != 0
    assert np.isfinite(out["p"][started]).all() and np.isfinite(out["P_pos"][started]).all()
    assert res["start_se3"]["ate_all"]["rmse"] < 0.05
    assert res["rpe_1s"]["rot_deg"]["rmse"] < 1.0
    # Uncertified VO covariance: best-effort output, explicitly not valid.
    assert res["valid"] == 0
    assert np.all(out["reasons"][started] & REASONS["COVARIANCE_UNRELIABLE"])
    assert np.mean(out["camera_state"][started] == CAM_NOMINAL) > 0.9


def test_image_blackout_before_tracker_is_an_outage_with_recovery(fixture_dir):
    blk = [{"kind": "blackout", "start": 3.5, "end": 4.5, "severity": 1.0}]
    s, fe, out, diag, res = _run(fixture_dir, blk)
    rel = out["t"] - out["t"][0]
    inside = (rel > 4.1) & (rel < 4.5)
    assert np.all(out["camera_state"][inside] == CAM_OUTAGE)
    assert np.mean(out["camera_state"][rel > 5.2] == CAM_NOMINAL) > 0.9
    f = res["faults"][0]
    assert f["err_at_end"] < 0.2 and np.isfinite(f["max_step_dev_after_m"])


def test_vo_only_and_imu_only_run_on_the_same_inputs(fixture_dir):
    for system in ("vo_only", "imu_only", "conventional"):
        _, _, out, _, res = _run(fixture_dir, system=system)
        assert "start_se3" in res, system
