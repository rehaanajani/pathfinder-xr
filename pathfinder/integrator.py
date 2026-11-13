"""Plain strapdown integration: the "trust the IMU blindly" baseline."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import GRAVITY

G = np.array([0.0, 0.0, -GRAVITY])


def strapdown(gyro: np.ndarray, acc: np.ndarray, dt: float, p0, v0, q0,
              bg=None, ba=None) -> np.ndarray:
    """Integrate one IMU stream from a known start. Returns positions (N, 3)."""
    n = len(gyro)
    bg = np.zeros(3) if bg is None else bg
    ba = np.zeros(3) if ba is None else ba
    rot = R.from_quat(q0)
    p, v = np.array(p0, float), np.array(v0, float)
    out = np.zeros((n, 3))
    for k in range(n):
        out[k] = p
        a = rot.apply(acc[k] - ba) + G
        p = p + v * dt
        v = v + a * dt
        rot = rot * R.from_rotvec((gyro[k] - bg) * dt)
    return out
