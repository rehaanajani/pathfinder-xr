"""A head-worn walk with exact ground truth, for tests and for the demo when no dataset is present.

The person walks a smooth loop at about 1.2 m/s, the head bobs with each step and looks
around (yaw up to +-35 deg, small pitch and roll). IMU readings are derived from the
same discrete motion the integrator uses, so clean data integrates back almost exactly.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import GRAVITY
from .sequence import Sequence

G = np.array([0.0, 0.0, -GRAVITY])


def head_walk(duration: float = 60.0, rate: float = 200.0, n_imus: int = 1, seed: int = 0,
              imu_noise: bool = True) -> Sequence:
    rng = np.random.default_rng(seed)
    n = int(duration * rate) + 1
    t = np.arange(n) / rate

    # Path: two incommensurate loops, so it never exactly repeats.
    a1, a2 = 6.0, 2.5
    w1, w2 = 2 * np.pi / 40.0, 2 * np.pi / 13.0
    ph = rng.uniform(0, 2 * np.pi, 2)
    x = a1 * np.sin(w1 * t + ph[0]) + a2 * np.sin(w2 * t)
    y = a1 * np.cos(w1 * t + ph[0]) * 0.8 + a2 * np.sin(w2 * t + ph[1])
    z = 1.6 + 0.03 * np.sin(2 * np.pi * 1.9 * t)          # step bob
    p = np.stack([x, y, z], 1)

    v = np.gradient(p, t, axis=0)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    yaw = heading + np.deg2rad(35) * np.sin(2 * np.pi * t / 7.0) * np.sin(2 * np.pi * t / 23.0)
    pitch = np.deg2rad(8) * np.sin(2 * np.pi * t / 5.0) + np.deg2rad(2) * np.sin(2 * np.pi * 1.9 * t)
    roll = np.deg2rad(3) * np.sin(2 * np.pi * 0.95 * t)
    rot = R.from_euler("ZYX", np.stack([yaw, pitch, roll], 1))
    q = rot.as_quat()

    # Discrete-consistent IMU: rotation increments and finite-difference acceleration.
    dt = 1.0 / rate
    Rm = rot.as_matrix()
    gyro = np.zeros((n, 3))
    gyro[:-1] = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt
    gyro[-1] = gyro[-2]
    vel = np.zeros((n, 3))
    vel[:-1] = (p[1:] - p[:-1]) / dt
    vel[-1] = vel[-2]
    a = np.zeros((n, 3))
    a[:-1] = (vel[1:] - vel[:-1]) / dt
    a[-1] = a[-2]
    acc = np.einsum("nji,nj->ni", Rm, a - G)

    gyros, accs = [], []
    for k in range(n_imus):
        g_k, a_k = gyro.copy(), acc.copy()
        if imu_noise:
            r = np.random.default_rng(seed * 100 + k + 1)
            g_k += r.normal(0, 0.003, g_k.shape) + r.normal(0, 0.0005, 3)
            a_k += r.normal(0, 0.03, a_k.shape) + r.normal(0, 0.01, 3)
        gyros.append(g_k)
        accs.append(a_k)

    return Sequence(name=f"synthetic_walk_s{seed}", t=t, gyro=np.stack(gyros), acc=np.stack(accs),
                    p=p, q=q, v=vel, meta={"source": "synthetic", "rate": rate})
