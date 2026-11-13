"""Fault injector: breaks sensor inputs on purpose, never the ground truth.

Every fault is a named, seeded, repeatable recipe, so a benchmark run is reproducible.
IMU faults act on one IMU (``imu``) or all of them (``imu=None``).
Camera faults produce a per-frame schedule that the visual front end turns into
dropouts, darkness, blur, or wrong fixes.
"""

from dataclasses import dataclass, field

import numpy as np

from .sequence import Sequence


@dataclass
class Fault:
    kind: str
    start: float = 0.0            # seconds from sequence start
    duration: float = 1e9         # seconds
    imu: int | None = None        # which IMU; None = all
    params: dict = field(default_factory=dict)

    def label(self) -> str:
        who = "all IMUs" if self.imu is None else f"IMU {self.imu}"
        if self.kind.startswith("camera"):
            who = "camera"
        extra = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind} ({who}, {self.duration:g}s{', ' + extra if extra else ''})"


IMU_KINDS = {"imu_noise", "imu_bias", "imu_bias_drift", "imu_dropout", "imu_saturation",
             "imu_stuck", "imu_spikes"}
CAMERA_KINDS = {"camera_blackout", "camera_dark", "camera_blur", "camera_wrong_fix"}


def _mask(seq: Sequence, f: Fault) -> np.ndarray:
    rel = seq.t - seq.t[0]
    return (rel >= f.start) & (rel < f.start + f.duration)


def apply_imu_faults(seq: Sequence, faults: list[Fault], seed: int = 0) -> Sequence:
    out = seq.copy()
    rng = np.random.default_rng(seed)
    for f in faults:
        if f.kind not in IMU_KINDS:
            continue
        m = _mask(out, f)
        idx = range(out.n_imus) if f.imu is None else [f.imu]
        for k in idx:
            g, a = out.gyro[k], out.acc[k]
            n = int(m.sum())
            if n == 0:
                continue
            p = f.params
            if f.kind == "imu_noise":
                g[m] += rng.normal(0, p.get("gyro_std", 0.05), (n, 3))
                a[m] += rng.normal(0, p.get("acc_std", 0.5), (n, 3))
            elif f.kind == "imu_bias":
                g[m] += np.asarray(p.get("gyro", [0.02, -0.015, 0.01]))
                a[m] += np.asarray(p.get("acc", [0.3, -0.2, 0.15]))
            elif f.kind == "imu_bias_drift":
                dt = out.dt
                g[m] += np.cumsum(rng.normal(0, p.get("gyro_rw", 0.005) * np.sqrt(dt), (n, 3)), 0)
                a[m] += np.cumsum(rng.normal(0, p.get("acc_rw", 0.05) * np.sqrt(dt), (n, 3)), 0)
            elif f.kind == "imu_dropout":
                # Packets lost: the driver repeats the last good sample.
                first = int(np.argmax(m))
                hold = max(first - 1, 0)
                g[m] = g[hold]
                a[m] = a[hold]
            elif f.kind == "imu_saturation":
                g[m] = np.clip(g[m], -p.get("gyro_max", 1.0), p.get("gyro_max", 1.0))
                a[m] = np.clip(a[m], -p.get("acc_max", 12.0), p.get("acc_max", 12.0))
            elif f.kind == "imu_stuck":
                g[m] = p.get("gyro_value", 0.0)
                a[m] = p.get("acc_value", 0.0)
            elif f.kind == "imu_spikes":
                hit = m & (rng.random(len(m)) < p.get("rate", 0.02))
                h = int(hit.sum())
                g[hit] += rng.normal(0, p.get("gyro_mag", 3.0), (h, 3))
                a[hit] += rng.normal(0, p.get("acc_mag", 30.0), (h, 3))
    out.meta = dict(out.meta, faults=[f.label() for f in faults])
    return out


def camera_schedule(t_frames: np.ndarray, faults: list[Fault]) -> dict:
    """Per camera frame: is it available, how dark, how blurred, is the fix corrupted."""
    rel = t_frames - t_frames[0]
    n = len(t_frames)
    sched = {"available": np.ones(n, bool), "brightness": np.ones(n), "blur": np.zeros(n),
             "wrong_fix": np.zeros((n, 3))}
    rng = np.random.default_rng(1234)
    for f in faults:
        if f.kind not in CAMERA_KINDS:
            continue
        m = (rel >= f.start) & (rel < f.start + f.duration)
        if f.kind == "camera_blackout":
            sched["available"][m] = False
        elif f.kind == "camera_dark":
            sched["brightness"][m] = f.params.get("level", 0.08)
        elif f.kind == "camera_blur":
            sched["blur"][m] = f.params.get("pixels", 15)
        elif f.kind == "camera_wrong_fix":
            # Visual tracking locks onto a moving object / repeated texture and drifts away.
            drift = np.cumsum(rng.normal(0, f.params.get("step", 0.05), (int(m.sum()), 3)), 0)
            sched["wrong_fix"][m] = drift + np.asarray(f.params.get("offset", [0.6, -0.4, 0.0]))
    return sched


# Named scenarios used by the benchmark: Samsung's ICCV-2023 conditions plus IMU faults.
def scenarios(blackout_s: float = 10.0) -> dict[str, list[Fault]]:
    mid = 20.0
    return {
        "clean": [],
        "camera_blackout": [Fault("camera_blackout", mid, blackout_s)],
        "camera_dark": [Fault("camera_dark", mid, blackout_s)],
        "camera_blur": [Fault("camera_blur", mid, blackout_s)],
        "camera_wrong_fix": [Fault("camera_wrong_fix", mid, blackout_s)],
        "imu_noise_all": [Fault("imu_noise", mid, blackout_s, None)],
        "one_imu_bias": [Fault("imu_bias", mid, 1e9, 0)],
        "one_imu_stuck": [Fault("imu_stuck", mid, blackout_s, 0)],
        "one_imu_spikes": [Fault("imu_spikes", mid, 1e9, 0)],
        "one_imu_saturation": [Fault("imu_saturation", mid, 1e9, 0, {"gyro_max": 0.6, "acc_max": 10.5})],
        "blackout_plus_one_imu_bias": [Fault("camera_blackout", mid, blackout_s),
                                       Fault("imu_bias", mid - 5, 1e9, 0)],
        "dark_plus_imu_drift": [Fault("camera_dark", mid, blackout_s),
                                Fault("imu_bias_drift", 0, 1e9, None)],
        # Common-mode faults hit every IMU on the board at once (shared vibration, heat, a fast
        # head turn past the sensor range), so voting cannot remove them. This is where the
        # fault-aware network has to carry the load.
        "blackout_plus_vibration_all": [Fault("camera_blackout", mid, blackout_s),
                                        Fault("imu_noise", mid, blackout_s, None,
                                              {"gyro_std": 0.08, "acc_std": 0.8})],
        "blackout_plus_spikes_all": [Fault("camera_blackout", mid, blackout_s),
                                     Fault("imu_spikes", mid, blackout_s, None, {"rate": 0.02})],
        "blackout_plus_saturation_all": [Fault("camera_blackout", mid, blackout_s),
                                         Fault("imu_saturation", mid, blackout_s, None,
                                               {"gyro_max": 0.5, "acc_max": 10.3})],
        "blackout_plus_dropout_all": [Fault("camera_blackout", mid, blackout_s),
                                      Fault("imu_dropout", mid + 3, 0.3, None),
                                      Fault("imu_dropout", mid + 6, 0.3, None)],
    }
