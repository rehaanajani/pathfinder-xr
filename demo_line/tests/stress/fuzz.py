"""Random fault-schedule generator for the fuzz test (seeded, so every case is replayable)."""
import numpy as np

from harness import ALL_KINDS, make_case
from pathfinder.faults import Fault

IMU_PARAMS = {
    "imu_noise": lambda r: {"gyro_std": float(r.uniform(0.001, 0.3)), "acc_std": float(r.uniform(0.01, 3))},
    "imu_bias": lambda r: {"gyro": r.normal(0, 0.1, 3).tolist(), "acc": r.normal(0, 1.5, 3).tolist()},
    "imu_bias_drift": lambda r: {"gyro_rw": float(r.uniform(0.001, 0.05)), "acc_rw": float(r.uniform(0.01, 0.5))},
    "imu_saturation": lambda r: {"gyro_max": float(r.uniform(0.2, 3)), "acc_max": float(r.uniform(9.9, 20))},
    "imu_stuck": lambda r: {"gyro_value": float(r.choice([0, 0.5])), "acc_value": float(r.choice([0, 9.81]))},
    "imu_spikes": lambda r: {"rate": float(r.uniform(0.005, 0.2))},
    "camera_dark": lambda r: {"level": float(r.uniform(0.01, 0.6))},
    "camera_blur": lambda r: {"pixels": int(r.integers(3, 40))},
    "camera_wrong_fix": lambda r: {"step": float(r.uniform(0, 0.2)), "offset": r.normal(0, 1, 3).tolist()},
}


def random_case(i, duration=20.0, rate=200.0):
    r = np.random.default_rng(10_000 + i)
    n_imus = int(r.integers(1, 7))
    faults = []
    for _ in range(int(r.integers(0, 6))):
        kind = str(r.choice(ALL_KINDS))
        edge = r.random()
        if edge < 0.15:
            start = 0.0                                     # fault at t = 0
        elif edge < 0.30:
            start = duration - 1.0 / rate                   # on the very last sample
        else:
            start = float(r.uniform(0, duration))
        dur = float(r.choice([0.0, 1e-3, r.uniform(0.05, 10), 1e9]))
        imu = None if (kind in ("camera_blackout",) or r.random() < 0.3) else int(r.integers(0, n_imus))
        if kind.startswith("camera"):
            imu = None
        faults.append(Fault(kind, start, dur, imu, IMU_PARAMS.get(kind, lambda r: {})(r)))
    keep = float(r.choice([1.0, 1.0, r.uniform(0, 1), 0.0]))
    return make_case(seed=i, duration=duration, rate=rate, n_imus=n_imus, faults=faults, cam_keep=keep)
