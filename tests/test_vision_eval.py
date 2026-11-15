"""Real-vision evaluation plumbing: front-end geometry fixes, VO-only chaining, IMU corruption, scoring
definitions, data-loader safety and runtime isolation (issue #2). No dataset or model needed."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pathfinder import scoring
from pathfinder.camera_model import CameraModel, RigCalibration
from pathfinder.render import render
from pathfinder.stereo_vo import StereoVO, VOConfig
from pathfinder.vision_eval import corrupt_imu, vo_only_trajectory

ROOT = Path(__file__).resolve().parent.parent
EQUI = CameraModel("equidistant", 190.978, 190.973, 254.932, 256.897,
                   (0.00348, 0.000716, -0.00205, 0.000203), 512, 512)
T_C0_I = np.eye(4)
T_C0_I[:3, :3] = Rot.from_euler("x", -90, degrees=True).as_matrix()


def _rig(T_c1_c0=None):
    if T_c1_c0 is None:
        T_c1_c0 = np.eye(4)
        T_c1_c0[:3, 3] = [-0.10, 0, 0]
    return RigCalibration(EQUI, EQUI, T_C0_I, T_c1_c0)


def _pair(rig_true, T_w_i, seed=0):
    T_w_c0 = T_w_i @ np.linalg.inv(rig_true.T_c0_i)
    return (render(rig_true.cam0, T_w_c0, seed=seed),
            render(rig_true.cam1, T_w_c0 @ np.linalg.inv(rig_true.T_c1_c0), seed=seed))


def _pose(k):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("zyx", [2 * k, 0.8 * k, -0.4 * k], degrees=True).as_matrix() @ \
        Rot.from_euler("x", 90, degrees=True).as_matrix()
    T[:3, 3] = [0.03 * k, -0.015 * k, 0.01 * k]
    return T


def test_landmarks_anchor_on_left_ray_so_zero_motion_has_no_bias():
    # The images come from a rig whose right camera is pitched 0.3 deg relative to the calibration
    # (a vertical epipolar error, as found on TUM-VI room1). Without anchoring, DLT moved every
    # landmark off its left ray and a static camera measured a rotation.
    true = np.eye(4)
    true[:3, :3] = Rot.from_euler("x", 0.3, degrees=True).as_matrix()
    true[:3, 3] = [-0.10, 0, 0]
    rig_true, rig_cal = _rig(true), _rig()
    vo = StereoVO(rig_cal)
    a, b = _pair(rig_true, _pose(0))
    ev = vo.step(0.0, a, b) + vo.step(0.05, a, b)
    rel = [e for e in ev if e["kind"] == "rel"]
    assert len(rel) == 1
    assert np.rad2deg(np.linalg.norm(Rot.from_matrix(rel[0]["dR"]).as_rotvec())) < 0.005
    assert np.linalg.norm(rel[0]["dp"]) < 1e-4


def test_keyframe_interval_solves_against_the_keyframe():
    rig = _rig()
    vo = StereoVO(rig, VOConfig(kf_every=3))
    kinds, rels = [], []
    for k in range(7):
        a, b = _pair(rig, _pose(k))
        ev = vo.step(0.05 * k, a, b)
        kinds.append([e["kind"] for e in ev])
        rels += [(k, e) for e in ev if e["kind"] == "rel"]
    assert kinds == [["kf"], [], [], ["rel", "kf"], [], [], ["rel", "kf"]]
    for k, e in rels:
        T = np.linalg.inv(_pose(k - 3)) @ _pose(k)
        assert e["frames"] == 3
        assert np.linalg.norm(e["dp"] - T[:3, 3]) < 0.01
        assert np.linalg.norm(Rot.from_matrix(T[:3, :3].T @ e["dR"]).as_rotvec()) < np.deg2rad(0.3)


def test_vo_only_chains_and_holds_across_a_break():
    R1 = Rot.from_euler("z", 10, degrees=True).as_matrix()
    ev = [{"t": 0.0, "kind": "kf", "kf": 1},
          {"t": 0.1, "kind": "rel", "kf": 1, "dp": np.array([1.0, 0, 0]), "dR": R1},
          {"t": 0.1, "kind": "kf", "kf": 2},
          {"t": 0.2, "kind": "rel", "kf": 2, "dp": np.array([1.0, 0, 0]), "dR": np.eye(3)},
          {"t": 0.2, "kind": "kf", "kf": 3},
          {"t": 0.5, "kind": "kf", "kf": 4},          # restart after a break: no relative solve reached it
          {"t": 0.6, "kind": "rel", "kf": 4, "dp": np.array([0, 1.0, 0]), "dR": np.eye(3)},
          {"t": 0.6, "kind": "kf", "kf": 5}]
    t = np.array([-0.05, 0.0, 0.15, 0.25, 0.55, 0.65])
    out = vo_only_trajectory(t, ev)
    assert np.isnan(out["p"][0]).all() and not out["valid"][0]
    np.testing.assert_allclose(out["p"][2], [1, 0, 0], atol=1e-12)
    x2 = np.array([1, 0, 0]) + R1 @ np.array([1.0, 0, 0])
    np.testing.assert_allclose(out["p"][3], x2, atol=1e-12)
    np.testing.assert_allclose(out["p"][4], x2, atol=1e-12)      # held through the break
    assert not out["valid"][4]
    np.testing.assert_allclose(out["p"][5], x2 + R1 @ np.array([0, 1.0, 0]), atol=1e-12)
    assert out["valid"][5]


def test_imu_corruption_is_confined_to_its_interval():
    t = np.arange(0, 10, 0.005)
    g = np.zeros((len(t), 3))
    a = np.tile([0, 0, 9.81], (len(t), 1))
    g2, a2 = corrupt_imu(t, g, a, [{"kind": "dropout", "start": 2.0, "end": 3.0},
                                   {"kind": "saturation", "start": 5.0, "end": 6.0, "acc_max": 5.0},
                                   {"kind": "bias", "start": 8.0, "end": 9.0}])
    rel = t - t[0]
    assert np.isnan(g2[(rel >= 2) & (rel < 3)]).all() and np.isfinite(g2[rel < 2]).all()
    assert np.all(a2[(rel >= 5) & (rel < 6), 2] == 5.0) and np.all(a2[(rel >= 6) & (rel < 8), 2] == 9.81)
    assert np.all(g2[(rel >= 8) & (rel < 9), 0] == 0.02) and np.all(g2[rel >= 9] == 0)
    assert np.all(g == 0)                                            # inputs untouched
    with pytest.raises(ValueError):
        corrupt_imu(t, g, a, [{"kind": "teleport", "start": 0, "end": 1}])


def test_recovery_requires_a_sustained_return_and_discontinuity_subtracts_true_motion():
    t = np.arange(0, 20, 0.005)
    err = np.full(len(t), 0.5)
    err[(t >= 11) & (t < 11.5)] = 0.0            # brief crossing: not a recovery
    err[t >= 13] = 0.05
    assert scoring.recovery_time(t, err, 10.0, 0.1, hold=2.0) == pytest.approx(3.0, abs=0.01)
    assert scoring.recovery_time(t, np.full(len(t), 0.5), 10.0, 0.1) is None
    # A trajectory moving at 1 m/s with a 0.2 m jump after the fault: only the jump counts.
    ref = SimpleNamespace(t=t, p=np.c_[t, np.zeros((len(t), 2))], q=np.tile([0, 0, 0, 1.0], (len(t), 1)))
    p = ref.p.copy()
    p[t >= 10.5, 1] += 0.2
    out = {"t": t, "p": p, "q": ref.q.copy(), "valid": np.ones(len(t), bool), "P_pos": np.tile(np.eye(3), (len(t), 1, 1))}
    res, _ = scoring.score(out, ref, [{"start": 9.0, "end": 10.0, "kind": "x"}])
    assert res["faults"][0]["max_step_dev_after_m"] == pytest.approx(0.2, abs=1e-9)
    assert res["path_length_m"] == pytest.approx(t[-1], rel=1e-6)


def test_unsupported_reference_intervals_are_excluded_not_interpolated():
    rt = np.r_[np.arange(0, 5, 0.01), np.arange(6, 10, 0.01)]        # 1 s mocap gap
    gp, gq, ok = scoring.reference_at(rt, np.zeros((len(rt), 3)), np.tile([0, 0, 0, 1.0], (len(rt), 1)),
                                      np.array([1.0, 5.5, 7.0, 11.0]))
    assert ok.tolist() == [True, False, True, False]
    assert np.isnan(gp[1]).all()


def test_kalibr_loader_records_absent_time_shift_as_assumption(tmp_path):
    yaml_text = """
cam0:
  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: equidistant
  intrinsics: [190, 190, 256, 256]
  resolution: [512, 512]
cam1:
  T_cam_imu: [[1,0,0,-0.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: equidistant
  intrinsics: [190, 190, 256, 256]
  resolution: [512, 512]
"""
    p = tmp_path / "camchain.yaml"
    p.write_text(yaml_text)
    rig = RigCalibration.from_kalibr(p)
    assert rig.time_shift == 0.0 and rig.meta["time_shift_source"] == "absent_assumed_zero"
    p.write_text(yaml_text.replace("cam0:\n", "cam0:\n  timeshift_cam_imu: 0.002\n")
                 .replace("cam1:\n", "cam1:\n  timeshift_cam_imu: 0.002\n"))
    rig = RigCalibration.from_kalibr(p)
    assert rig.time_shift == 0.002 and rig.meta["time_shift_source"] == "file"


def test_loader_never_follows_symlinked_images(tmp_path):
    from pathfinder import data_tumvi_raw as D
    d = tmp_path / "dataset-x_512_16" / "mav0"
    for c in ("cam0", "cam1"):
        (d / c / "data").mkdir(parents=True)
    (d / "imu0").mkdir()
    (d / "imu0" / "data.csv").write_text("#t,wx,wy,wz,ax,ay,az\n1000,0,0,0,0,0,9.8\n2000,0,0,0,0,0,9.8\n")
    (d / "cam0" / "data.csv").write_text("#t,f\n1000,a.png\n2000,b.png\n")
    for c in ("cam0", "cam1"):
        (d / c / "data" / "a.png").write_bytes(b"x")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"y")
    try:
        os.symlink(outside, d / "cam0" / "data" / "b.png")
        os.symlink(outside, d / "cam1" / "data" / "b.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform/account")
    with pytest.warns(UserWarning):
        s = D.load_streams("x", root=tmp_path)
    assert [p[0].name for p in s.frame_paths] == ["a.png"]


RUNTIME = ("estimator.py", "filter.py", "stereo_vo.py", "camera_model.py", "image_faults.py",
           "vision_frontend.py", "imu_health.py", "predictor.py")


@pytest.mark.parametrize("name", RUNTIME)
def test_runtime_modules_cannot_reach_reference_or_scoring(name):
    tree = ast.parse((ROOT / "pathfinder" / name).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {node.module or ""} | {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not imported & {"scoring", "pathfinder.scoring", "metrics", "data_tumvi_raw", "data_tumvi", "data_tlio"}
    src = (ROOT / "pathfinder" / name).read_text()
    assert "load_reference" not in src and "mocap0" not in src
