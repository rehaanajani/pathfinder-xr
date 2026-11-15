"""Copy EuRoC recordings with the IMU frame rotated by a fixed 90 deg about x (ground truth untouched), the same
rotation as tests/stress/test_main_contract.py::test_unknown_gravity_rotated_input. Used by tools/sweep_airio.sh
to check how a body-frame network copes with an IMU mounted another way.

    python tools/make_rotated.py [--out local_data/rot] [--seq MH_04_difficult V1_03_difficult]
    PFX_DATA=local_data/rot python tools/eval_track3.py --data euroc:MH_04_difficult ...
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import DATA  # noqa: E402

RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "rot"))
    ap.add_argument("--seq", nargs="+", default=["MH_04_difficult", "V1_03_difficult"])
    a = ap.parse_args()
    for s in a.seq:
        src = DATA / "euroc" / s / "mav0"
        dst = Path(a.out) / "euroc" / s / "mav0"
        (dst / "imu0").mkdir(parents=True, exist_ok=True)
        (dst / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "state_groundtruth_estimate0" / "data.csv", dst / "state_groundtruth_estimate0" / "data.csv")
        d = np.loadtxt(src / "imu0" / "data.csv", delimiter=",", comments="#")
        d[:, 1:4] = d[:, 1:4] @ RX.T
        d[:, 4:7] = d[:, 4:7] @ RX.T
        with open(dst / "imu0" / "data.csv", "w") as f:
            f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
            for r in d:
                f.write(f"{int(r[0])}," + ",".join(f"{x:.9g}" for x in r[1:]) + "\n")
        print(dst)


if __name__ == "__main__":
    main()
