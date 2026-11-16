"""The visual front end against exactly known motion, rendered through the real camera models."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pathfinder.camera_model import CameraModel, RigCalibration
from pathfinder.render import render
from pathfinder.stereo_vo import StereoVO

EQUI = CameraModel("equidistant", 190.978, 190.973, 254.932, 256.897,
                   (0.00348, 0.000716, -0.00205, 0.000203), 512, 512)
RAT8 = CameraModel("radtan", 300.0, 300.0, 319.5, 239.5,
                   (0.31, 0.08, 1e-4, -2e-4, 0.002, 0.68, 0.17, 0.011), 640, 480)
# TUM-VI-like IMU -> cam0 (camera looks along the IMU's -y axis) and a 10 cm baseline.
T_C0_I = np.array([[-0.9995, 0.0075, -0.0299, 0.0456], [0.0296, -0.0344, -0.9990, -0.0712],
                   [-0.0085, -0.9994, 0.0342, -0.0447], [0, 0, 0, 1.0]])
T_C0_I[:3, :3] = Rot.from_matrix(T_C0_I[:3, :3]).as_matrix()


def _rig(cam, angle_deg=0.0):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("y", angle_deg, degrees=True).as_matrix()
    T[:3, 3] = [-0.10, 0, 0]
    return RigCalibration(cam, cam, T_C0_I, T)


def _frames(rig, T_w_i_list, seed=0):
    T_i_c0 = np.linalg.inv(rig.T_c0_i)
    T_c0_c1 = np.linalg.inv(rig.T_c1_c0)
    for T_w_i in T_w_i_list:
        T_w_c0 = T_w_i @ T_i_c0
        yield render(rig.cam0, T_w_c0, seed=seed), render(rig.cam1, T_w_c0 @ T_c0_c1, seed=seed)


def _trajectory(n=6):
    out = []
    for k in range(n):
        T = np.eye(4)
        T[:3, :3] = Rot.from_euler("zyx", [3 * k, 1.0 * k, -0.5 * k], degrees=True).as_matrix() @ \
            Rot.from_euler("x", 90, degrees=True).as_matrix()
        T[:3, 3] = [0.04 * k, -0.02 * k, 0.01 * k]
        out.append(T)
    return out


@pytest.mark.parametrize("cam,angle", [(EQUI, 0.0), (RAT8, 0.0), (RAT8, 20.0)])
def test_relative_pose_matches_rendered_motion_within_covariance(cam, angle):
    rig = _rig(cam, angle)
    vo = StereoVO(rig)
    traj = _trajectory()
    rels, kfs = [], 0
    for k, (a, b) in enumerate(_frames(rig, traj)):
        for e in vo.step(0.1 * k, a, b):
            if e["kind"] == "rel":
                rels.append((k, e))
            else:
                kfs += 1
    assert kfs == len(traj) and len(rels) == len(traj) - 1
    for k, e in rels:
        T_rel = np.linalg.inv(traj[k - 1]) @ traj[k]
        err_t = e["dp"] - T_rel[:3, 3]
        err_r = Rot.from_matrix(T_rel[:3, :3].T @ e["dR"]).as_rotvec()
        assert np.linalg.norm(err_t) < 0.01 and np.linalg.norm(err_r) < np.deg2rad(0.3)
        x = np.r_[err_t, err_r]
        nees = x @ np.linalg.solve(e["cov"], x)
        assert nees < 60.0          # renderer aliasing is not in the noise model; gross misfit only
        assert e["inliers"] >= 50 and e["health"] > 0.5


def test_wrong_model_degrades_metric_motion():
    # Feeding a rational-model camera through an equidistant model with its first 4 coefficients
    # (the PR #7 assumption) must not silently produce the same motion.
    rig = _rig(RAT8)
    wrong_cam = CameraModel("equidistant", RAT8.fx, RAT8.fy, RAT8.cx, RAT8.cy, RAT8.dist[:4], 640, 480)
    wrong = StereoVO(RigCalibration(wrong_cam, wrong_cam, rig.T_c0_i, rig.T_c1_c0))
    traj = _trajectory(3)
    errs = []
    for k, (a, b) in enumerate(_frames(rig, traj)):
        for e in wrong.step(0.1 * k, a, b):
            if e["kind"] == "rel":
                T_rel = np.linalg.inv(traj[k - 1]) @ traj[k]
                errs.append(np.linalg.norm(e["dp"] - T_rel[:3, 3]))
    assert not errs or max(errs) > 0.01


def test_blackout_breaks_tracking_and_restart_emits_keyframe_first():
    rig = _rig(EQUI)
    vo = StereoVO(rig)
    traj = _trajectory(5)
    frames = list(_frames(rig, traj))
    black = np.zeros_like(frames[0][0])
    seq = [frames[0], frames[1], (black, black), frames[3], frames[4]]
    kinds = [[e["kind"] for e in vo.step(0.1 * k, a, b)] for k, (a, b) in enumerate(seq)]
    assert kinds[0] == ["kf"] and kinds[1] == ["rel", "kf"]
    assert kinds[2] == []                  # no tracks, no stereo map: nothing emitted
    assert kinds[3] == ["kf"]              # restart: clone first, no relative measurement
    assert kinds[4] == ["rel", "kf"]


def test_wide_angle_pnp_uses_image_pixel_inliers():
    rng = np.random.default_rng(7)
    angles = rng.uniform(np.deg2rad(55), np.deg2rad(70), 30)
    az = rng.uniform(-np.pi, np.pi, 30)
    z = rng.uniform(2, 6, 30)
    X = np.c_[np.tan(angles) * z * np.cos(az), np.tan(angles) * z * np.sin(az), z]
    uv = EQUI.project(X) + rng.normal(0, .5, (30, 2))
    result = StereoVO(_rig(EQUI))._pnp(X, np.tile(np.eye(3) * 1e-8, (30, 1, 1)), uv)
    assert result is not None
    T, covariance, inliers, rms = result
    assert inliers >= 25 and rms < 1.
    assert np.linalg.norm(T[:3, 3]) < .1
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_degenerate_patch_cannot_emit_exploded_pose():
    rng = np.random.default_rng(1)
    X = np.c_[rng.uniform(-.05, .05, (40, 2)), np.full(40, 12.)]
    uv = EQUI.project(X) + rng.normal(0, .5, (40, 2))
    result = StereoVO(_rig(EQUI))._pnp(X, np.tile(np.eye(3) * 1e-8, (40, 1, 1)), uv)
    if result is not None:
        T, cov, _, _ = result
        assert np.isfinite(T).all() and np.linalg.norm(T[:3, 3]) < 1.
        assert np.linalg.eigvalsh(cov).min() > 0
