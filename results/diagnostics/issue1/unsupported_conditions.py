"""Characterize (do not certify) multiple IMU faults and unestimated clock offsets.

    python -m results.diagnostics.issue1.unsupported_conditions --out results/runs/unsupported.json

All data are synthetic. In particular a faulty majority can agree and cause a healthy sensor to be
excluded. This is an observability limit outside the single-fault isolation contract, not a score to hide.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.imu_health import ImuMonitor, STATE_NAMES
from pathfinder.synthetic import head_walk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for seed in range(3):
        for case, k in (("two_different_biases", 5), ("faulty_majority_agrees", 3),
                        ("clock_offset_2ms", 3), ("clock_offset_20ms", 3)):
            s = head_walk(18., n_imus=k, seed=seed)
            if case.startswith("clock"):
                bad = s.copy()
                offset = .002 if case == "clock_offset_2ms" else .02
                for field in ("gyro", "acc"):
                    a = getattr(bad, field)
                    for axis in range(3):
                        a[0, :, axis] = np.interp(s.t - offset, s.t, getattr(s, field)[0, :, axis])
                affected = [0]
                region = s.t >= 6.
            else:
                params = {"gyro": [.08, 0., 0.], "acc": [.8, 0., 0.]}
                other = params if k == 3 else {"gyro": [0., -.08, 0.], "acc": [0., -.8, 0.]}
                bad = apply_imu_faults(s, [Fault("imu_bias", 6., 10., 0, params),
                                           Fault("imu_bias", 6., 10., 1, other)])
                affected, region = [0, 1], (s.t >= 7.) & (s.t < 16.)
            g, a, info = ImuMonitor(k).run_batch(s.t, bad.gyro, bad.acc)
            windows = (info["window_t0"] >= 7.) & (info["window_t0"] < 16.)
            healthy = [i for i in range(k) if i not in affected]
            rows.append({"seed": seed, "case": case, "imus": k,
                         "state_fraction": {name: float(np.mean(info["state"][region] == i))
                                            for i, name in enumerate(STATE_NAMES)},
                         "healthy_sensor_exclusion_fraction": float(np.mean(np.any(info["weights"][healthy][:, windows] == 0., axis=0))),
                         "gyro_fusion_rmse": float(np.sqrt(np.nanmean((g[region] - s.gyro[:, region].mean(0)) ** 2))),
                         "acc_fusion_rmse": float(np.sqrt(np.nanmean((a[region] - s.acc[:, region].mean(0)) ** 2))),
                         "supported": False})
    result = {"data": "synthetic head_walk, simulated co-located IMUs", "rows": rows,
              "conclusion": "Multiple/colluding faults and inter-sensor time offsets are outside the supported single-fault/synchronized-input contract."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
