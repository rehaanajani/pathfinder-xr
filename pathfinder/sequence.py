"""The one data shape every module shares.

Frames: world is z-up with gravity (0, 0, -9.81). Each IMU reports, in the body frame,
angular rate (rad/s) and specific force f = R^T (a - g) (m/s^2).
Rotations are scipy-style quaternions (x, y, z, w), body to world.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sequence:
    name: str
    t: np.ndarray                 # (N,) seconds, uniform rate
    gyro: np.ndarray              # (K, N, 3) one block per IMU, body frame
    acc: np.ndarray               # (K, N, 3)
    p: np.ndarray                 # (N, 3) true position
    q: np.ndarray                 # (N, 4) true orientation, xyzw
    v: np.ndarray                 # (N, 3) true velocity
    meta: dict = field(default_factory=dict)

    @property
    def n_imus(self) -> int:
        return self.gyro.shape[0]

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    @property
    def rate(self) -> float:
        return 1.0 / self.dt

    def copy(self) -> "Sequence":
        return Sequence(self.name, self.t.copy(), self.gyro.copy(), self.acc.copy(),
                        self.p.copy(), self.q.copy(), self.v.copy(), dict(self.meta))

    def slice(self, start: int, stop: int) -> "Sequence":
        return Sequence(self.name, self.t[start:stop], self.gyro[:, start:stop],
                        self.acc[:, start:stop], self.p[start:stop], self.q[start:stop],
                        self.v[start:stop], dict(self.meta))
