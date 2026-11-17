"""Regressions for the external-VIO baseline and calibration fingerprints."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

import evaluate_vision
from pathfinder.data_tumvi_raw import Reference
from pathfinder.vision_eval import Frozen, system_configs
from pathfinder.vision_frontend import _src_hash
from tools.calibrate_vo import select_keyframe
from tools.external_vio import as_estimator_output, load_pose_table, module_imports, refuse_fusion, score_output
from tools.verify_real_images import assess_png
import cv2


def test_keyframe_choice_ignores_fused_ate():
    rows = [
        {"kf_every": 1, "vo_only_ate_rmse_m": 0.2, "vo_only_rot_rmse_deg": 1.0, "selected_ate_rmse_m": 0.01},
        {"kf_every": 4, "vo_only_ate_rmse_m": 0.1, "vo_only_rot_rmse_deg": 2.0, "selected_ate_rmse_m": 9.0},
    ]
    assert select_keyframe(rows)["kf_every"] == 4


def test_keyframe_tie_uses_vision_only_rotation():
    rows = [
        {"kf_every": 1, "vo_only_ate_rmse_m": 0.1, "vo_only_rot_rmse_deg": 2.0},
        {"kf_every": 2, "vo_only_ate_rmse_m": 0.1, "vo_only_rot_rmse_deg": 0.4},
    ]
    assert select_keyframe(rows)["kf_every"] == 2


def test_vision_configs_leave_covariance_unvalidated():
    fz = Frozen(imu_noise={"sigma_g": 1e-4, "sigma_a": 1e-3, "sigma_bg": 1e-5, "sigma_ba": 1e-4})
    for name, (cfg, _cam, _model) in system_configs(fz).items():
        assert cfg.vo_covariance_validated is False, name


def _cal(tmp_path, **extra):
    body = {"schema": "pathfinder.vo-calibration/1", "frontend_src_sha256": _src_hash(),
            "selected_vo_cov_scale": 1.0, "selected_kf_every": 1, "camera_time_offset_s": 0.0,
            "vo_covariance_validated": False}
    body.update(extra)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_stale_frontend_hash_is_rejected(tmp_path):
    path = _cal(tmp_path, frontend_src_sha256="0" * 64)
    with pytest.raises(SystemExit, match="front-end source changed"):
        evaluate_vision.main(["--calibration", str(path), "--no-model", "--seq", "room1"])


def test_opencv_fingerprint_mismatch_is_rejected(tmp_path):
    path = _cal(tmp_path, opencv="0.0.0")
    with pytest.raises(SystemExit, match="OpenCV"):
        evaluate_vision.main(["--calibration", str(path), "--no-model", "--seq", "room1"])


def test_calibration_cannot_assert_validated_covariance(tmp_path):
    path = _cal(tmp_path, vo_covariance_validated=True)
    with pytest.raises(SystemExit, match="vo_covariance_validated"):
        evaluate_vision.main(["--calibration", str(path), "--no-model", "--seq", "room1"])


def test_external_module_does_not_import_the_filter():
    banned = {"estimator", "filter", "replay", "pathfinder.estimator", "pathfinder.filter", "pathfinder.pipeline"}
    assert not module_imports() & banned
    tree = ast.parse(Path("tools/external_vio.py").read_text(encoding="utf-8"))
    assert "refuse_fusion" in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_fusion_of_external_poses_is_rejected():
    with pytest.raises(RuntimeError, match="counted twice"):
        refuse_fusion([0.0, 0.0, 0.0])


def test_external_pose_table_and_score_keep_covariance_invalid(tmp_path):
    stamp = np.array([1_520_530_308_000_000_000, 1_520_530_308_050_000_000], float)
    pose = np.array([[0, 0, 0], [0.1, 0, 0]])
    q_wxyz = np.tile([1, 0, 0, 0], (2, 1))
    text = "\n".join(f"{int(t)} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}"
                     for t, p, q in zip(stamp, pose, q_wxyz))
    path = tmp_path / "traj.txt"
    path.write_text("# euroc\n" + text + "\n", encoding="utf-8")
    t, p, q, unit = load_pose_table(path)
    assert unit == "ns"
    assert np.allclose(t, [1_520_530_308.0, 1_520_530_308.05])
    assert np.allclose(q, [[0, 0, 0, 1], [0, 0, 0, 1]])
    ref = Reference(t=t, p=p.copy(), q=q.copy())
    res, _ = score_output(as_estimator_output(t, p, q), ref)
    assert res["valid_fraction"] == 0
    assert res["start_se3"]["ate_all"]["rmse"] < 1e-9


def test_xyzw_table_is_not_silently_read_as_wxyz(tmp_path):
    yaw = Rot.from_euler("z", 90, degrees=True)
    xyzw = yaw.as_quat()
    wxyz = np.r_[xyzw[3], xyzw[:3]]
    path = tmp_path / "traj.txt"
    path.write_text(f"0 0 0 0 {xyzw[0]} {xyzw[1]} {xyzw[2]} {xyzw[3]}\n"
                    f"1 0 0 0 {xyzw[0]} {xyzw[1]} {xyzw[2]} {xyzw[3]}\n", encoding="utf-8")
    _, _, q_as_wxyz, _ = load_pose_table(path, quat_order="wxyz", time_unit="s")
    _, _, q_as_xyzw, _ = load_pose_table(path, quat_order="xyzw", time_unit="s")
    assert np.allclose(q_as_xyzw[0], xyzw)
    assert not np.allclose(q_as_wxyz[0], xyzw)


def test_flat_and_eight_bit_images_fail_the_real_image_audit(tmp_path):
    flat = np.full((512, 512), 1000, np.uint16)
    photo = np.zeros((512, 512), np.uint16)
    rng = np.random.default_rng(0)
    photo[:] = rng.integers(0, 4000, photo.shape, dtype=np.uint16)
    eight = (photo >> 8).astype(np.uint8)
    for img, expect in ((flat, False), (photo, True), (eight, False)):
        ok, buf = cv2.imencode(".png", img)
        assert ok
        assert assess_png(buf.tobytes())["ok"] is expect
        if img is photo:
            assert assess_png(buf.tobytes())["bit_depth"] == 16
