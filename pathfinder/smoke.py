"""Small redistributable smoke fixture in the TUM-VI/EuRoC directory layout (issue #3 B/D).

``make_fixture`` renders a deterministic synthetic recording (stereo PNGs through an equidistant
camera model, 200 Hz IMU derived from the same trajectory, 100 Hz reference poses, Kalibr camchain and
IMU noise file) so CI and fresh-setup checks run the *same* loader -> image front end -> estimator ->
scorer path as real data, in seconds and without any dataset download. Synthetic by construction:
smoke results are plumbing checks, never accuracy evidence.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import GRAVITY

CAM = dict(model="equidistant", fx=110.0, fy=110.0, cx=127.5, cy=127.5,
           dist=(0.0035, 0.0007, -0.0021, 0.0002), width=256, height=256)
# IMU -> cam0: camera z (optical axis) along IMU +x, camera y down = IMU -z.
R_C0_I = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
BASELINE = 0.10


def trajectory(t):
    """Smooth motion inside the renderer's 6 x 6 x 3 m room: positions (N, 3), rotations body->world."""
    w = 2 * np.pi / 8.0
    p = np.stack([0.8 * np.sin(w * t), 0.5 * np.sin(2 * w * t), 0.1 * np.sin(3 * w * t)], 1)
    eul = np.stack([0.6 * np.sin(w * t), 0.08 * np.sin(1.7 * w * t), 0.05 * np.sin(2.3 * w * t)], 1)
    return p, Rot.from_euler("zyx", eul)


def imu_from_trajectory(t, rng, gyro_std=0.002, acc_std=0.02):
    """Body-frame gyro and specific force, by central differences on a 4x finer grid."""
    dt = t[1] - t[0]
    h = dt / 4
    p_m, R_m = trajectory(t - h)
    p_0, R_0 = trajectory(t)
    p_p, R_p = trajectory(t + h)
    acc_w = (p_p - 2 * p_0 + p_m) / h ** 2
    gyro = (R_m.inv() * R_p).as_rotvec() / (2 * h)
    f = R_0.inv().apply(acc_w + np.array([0.0, 0.0, GRAVITY]))
    return gyro + rng.normal(0, gyro_std, gyro.shape), f + rng.normal(0, acc_std, f.shape)


def make_fixture(root, name: str = "smoke", duration: float = 6.0, cam_rate: float = 10.0, seed: int = 0) -> Path:
    import cv2

    from .camera_model import CameraModel
    from .render import render
    root = Path(root)
    d = root / f"dataset-{name}_512_16"
    m = d / "mav0"
    for sub in ("cam0/data", "cam1/data", "imu0", "mocap0"):
        (m / sub).mkdir(parents=True, exist_ok=True)
    (d / "dso").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0_ns = 1_000_000_000_000_000_000
    t_imu = np.arange(0, duration, 0.005)
    gyro, acc = imu_from_trajectory(t_imu, rng)
    ns = t0_ns + np.round(t_imu * 1e9).astype(np.int64)
    with open(m / "imu0" / "data.csv", "w") as fh:
        fh.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                 "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for k in range(len(t_imu)):
            fh.write(f"{ns[k]}," + ",".join(f"{v:.9f}" for v in (*gyro[k], *acc[k])) + "\n")
    t_ref = np.arange(0, duration, 0.01)
    p, R = trajectory(t_ref)
    q = R.as_quat()
    with open(m / "mocap0" / "data.csv", "w") as fh:
        fh.write("#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n")
        for k in range(len(t_ref)):
            fh.write(f"{t0_ns + int(round(t_ref[k] * 1e9))}," + ",".join(
                f"{v:.9f}" for v in (*p[k], q[k, 3], q[k, 0], q[k, 1], q[k, 2])) + "\n")
    cam = CameraModel(CAM["model"], CAM["fx"], CAM["fy"], CAM["cx"], CAM["cy"], CAM["dist"], CAM["width"], CAM["height"])
    T_c0_i = np.eye(4)
    T_c0_i[:3, :3] = R_C0_I
    T_c1_c0 = np.eye(4)
    T_c1_c0[:3, 3] = [-BASELINE, 0, 0]
    T_i_c0 = np.linalg.inv(T_c0_i)
    t_cam = np.arange(0.0, duration - 1e-9, 1.0 / cam_rate) + 0.0021      # cameras not on IMU ticks
    pc, Rc = trajectory(t_cam)
    rows = []
    for k, tc in enumerate(t_cam):
        T_w_i = np.eye(4)
        T_w_i[:3, :3], T_w_i[:3, 3] = Rc[k].as_matrix(), pc[k]
        T_w_c0 = T_w_i @ T_i_c0
        stamp = t0_ns + int(round(tc * 1e9))
        cv2.imwrite(str(m / "cam0" / "data" / f"{stamp}.png"), render(cam, T_w_c0, seed=seed))
        cv2.imwrite(str(m / "cam1" / "data" / f"{stamp}.png"), render(cam, T_w_c0 @ np.linalg.inv(T_c1_c0), seed=seed))
        rows.append(f"{stamp},{stamp}.png")
    for c in ("cam0", "cam1"):
        (m / c / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")

    def mat(T):
        return "\n".join("  - [" + ", ".join(f"{v:.12f}" for v in row) + "]" for row in T)
    T_c1_i = T_c1_c0 @ T_c0_i
    cam_yaml = (f"  camera_model: pinhole\n  distortion_coeffs: [{', '.join(map(str, CAM['dist']))}]\n"
                f"  distortion_model: equidistant\n  intrinsics: [{CAM['fx']}, {CAM['fy']}, {CAM['cx']}, {CAM['cy']}]\n"
                f"  resolution: [{CAM['width']}, {CAM['height']}]\n")
    (d / "dso" / "camchain.yaml").write_text(
        "cam0:\n  T_cam_imu:\n" + mat(T_c0_i) + "\n" + cam_yaml +
        "cam1:\n  T_cam_imu:\n" + mat(T_c1_i) + "\n  T_cn_cnm1:\n" + mat(T_c1_c0) + "\n" + cam_yaml)
    (d / "dso" / "imu_config.yaml").write_text(
        "# synthetic smoke fixture (noise actually injected: gyro 0.002 rad/s, acc 0.02 m/s^2 per 5 ms sample)\n"
        f"accelerometer_noise_density: {0.02 * np.sqrt(0.005):.6f}\naccelerometer_random_walk: 0.0001\n"
        f"gyroscope_noise_density: {0.002 * np.sqrt(0.005):.6f}\ngyroscope_random_walk: 0.00001\n")
    (d / "FIXTURE.txt").write_text("Synthetic smoke fixture generated by pathfinder.smoke.make_fixture "
                                   f"(seed={seed}, duration={duration}). Not real data.\n")
    return d
