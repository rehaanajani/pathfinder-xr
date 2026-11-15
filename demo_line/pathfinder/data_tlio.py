"""Loader for Meta's TLIO headset dataset (CC BY-NC, see README for the download).

Each sequence folder holds ``imu0_resampled.npy`` at 200 Hz with columns:
ts_us, gyro (world frame, bias-compensated), accel (world frame, bias-compensated),
q_World_Device (xyzw), pos_World, vel_World. Ground truth comes from Meta's offline VIO.

We rotate the IMU back into the device (body) frame with the ground-truth orientation, so
it looks like what a headset IMU really reports. The dataset has one IMU per sequence;
``redundant_imus`` builds K copies of it with independent noise and bias (disclosed in the
README) so the multi-IMU logic can be tested on real head motion.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sequence import Sequence

ROOT = Path(__file__).resolve().parent.parent / "local_data" / "tlio_golden"


def split(name: str, root: Path = ROOT) -> list[str]:
    return [s.strip() for s in (root / f"{name}_list.txt").read_text().split() if s.strip()]


def load(seq_id: str, root: Path = ROOT, max_seconds: float | None = None) -> Sequence:
    d = np.load(root / seq_id / "imu0_resampled.npy")
    if max_seconds:
        d = d[: int(max_seconds * 200) + 1]
    t = (d[:, 0] - d[0, 0]) * 1e-6
    gw, aw = d[:, 1:4], d[:, 4:7]
    q, p, v = d[:, 7:11], d[:, 11:14], d[:, 14:17]
    rot = R.from_quat(q)
    gb = rot.inv().apply(gw)
    ab = rot.inv().apply(aw)
    return Sequence(name=seq_id, t=t, gyro=gb[None], acc=ab[None], p=p, q=q, v=v,
                    meta={"source": "tlio_golden"})


def redundant_imus(seq: Sequence, k: int, seed: int = 0, gyro_std=0.002, acc_std=0.02,
                   gyro_bias=0.0005, acc_bias=0.01) -> Sequence:
    """K IMUs from one real IMU: each gets its own white noise and small constant bias."""
    rng = np.random.default_rng(seed)
    base_g, base_a = seq.gyro[0], seq.acc[0]
    gs, as_ = [], []
    for _ in range(k):
        gs.append(base_g + rng.normal(0, gyro_std, base_g.shape) + rng.normal(0, gyro_bias, 3))
        as_.append(base_a + rng.normal(0, acc_std, base_a.shape) + rng.normal(0, acc_bias, 3))
    out = seq.copy()
    out.gyro, out.acc = np.stack(gs), np.stack(as_)
    out.meta = dict(seq.meta, redundant_imus=k)
    return out
