import cv2
import numpy as np
import pytest

from pathfinder.camera_model import CameraModel, RigCalibration

# TUM-VI 512x512 cam0 (equidistant) and a Monado Odyssey+-like rational 8-coefficient camera.
EQUI = CameraModel("equidistant", 190.978, 190.973, 254.932, 256.897,
                   (0.00348, 0.000716, -0.00205, 0.000203), 512, 512)
RAT8 = CameraModel("radtan", 278.6, 278.4, 319.5, 239.2,
                   (0.31, 0.08, 1e-4, -2e-4, 0.002, 0.68, 0.17, 0.011), 640, 480)


def _points(n=300, seed=0, max_angle=1.1):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, max_angle, n)
    az = rng.uniform(-np.pi, np.pi, n)
    z = rng.uniform(0.5, 6.0, n)
    r = np.tan(ang) * z
    return np.stack([r * np.cos(az), r * np.sin(az), z], 1)


def test_equidistant_matches_opencv_fisheye_and_roundtrips():
    X = _points()
    ours = EQUI.project(X)
    cv, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), EQUI.K, np.array(EQUI.dist))
    assert np.max(np.abs(ours - cv[:, 0])) < 1e-6
    xy = EQUI.unproject(ours)
    assert np.max(np.abs(xy - X[:, :2] / X[:, 2:])) < 1e-6


def test_rational8_matches_opencv_and_roundtrips():
    X = _points(max_angle=0.8)
    ours = RAT8.project(X)
    cv, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), RAT8.K, np.array(RAT8.dist))
    assert np.max(np.abs(ours - cv[:, 0])) < 1e-6
    xy = RAT8.unproject(ours)
    assert np.max(np.abs(xy - X[:, :2] / X[:, 2:])) < 1e-6


def test_wrong_model_is_measurably_wrong():
    # Applying the fisheye model to a rational-model camera (the PR #7 assumption) misplaces rays.
    X = _points(max_angle=0.8)
    uv = RAT8.project(X)
    wrong = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), RAT8.K, np.array(RAT8.dist[:4]))[:, 0]
    err_px = np.linalg.norm(wrong - X[:, :2] / X[:, 2:], axis=1) * RAT8.focal
    assert np.median(err_px) > 5.0


def test_validation_rejects_bad_calibration():
    with pytest.raises(ValueError):
        CameraModel("equidistant", 190, 190, 256, 256, (0.1, 0.2, 0.3, 0.4, 0.5), 512, 512)
    with pytest.raises(ValueError):
        CameraModel("kb8", 190, 190, 256, 256, (0, 0, 0, 0), 512, 512)
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError):
        RigCalibration(EQUI, EQUI, bad, np.eye(4))
    with pytest.raises(ValueError):
        RigCalibration(EQUI, EQUI, np.eye(4), np.eye(4), time_shift=2.0)


def test_kalibr_loader(tmp_path):
    text = """
cam0:
  camera_model: pinhole
  distortion_model: equidistant
  distortion_coeffs: [0.00348, 0.000716, -0.00205, 0.000203]
  intrinsics: [190.978, 190.973, 254.932, 256.897]
  resolution: [512, 512]
  T_cam_imu:
  - [-0.99952504, 0.00750192, -0.02989013, 0.04557484]
  - [0.02961534, -0.03439736, -0.99896935, -0.07116180]
  - [-0.00852233, -0.99938008, 0.03415885, -0.04468125]
  - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: 0.0
cam1:
  camera_model: pinhole
  distortion_model: radtan8
  distortion_coeffs: [0.31, 0.08, 0.0001, -0.0002, 0.002, 0.68, 0.17, 0.011]
  intrinsics: [278.6, 278.4, 319.5, 239.2]
  resolution: [640, 480]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, -0.101]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
"""
    p = tmp_path / "camchain.yaml"
    p.write_text(text)
    rig = RigCalibration.from_kalibr(p)
    assert rig.cam0.model == "equidistant" and rig.cam1.model == "radtan" and len(rig.cam1.dist) == 8
    assert abs(rig.baseline - 0.101) < 1e-9 and len(rig.meta["sha256"]) == 64
