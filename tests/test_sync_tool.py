"""tools/check_sync.py recovers a known injected time offset and frame convention (synthetic fixture)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_sync  # noqa: E402

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.smoke import make_fixture  # noqa: E402
from pathfinder.vision_frontend import run_frontend  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("sync")
    make_fixture(root, duration=6.0)
    return root


@pytest.mark.parametrize("shift_ms", [0.0, 7.0])
def test_offsets_are_recovered(fx, shift_ms):
    s = D.load_streams("smoke", fx)
    ref = D.load_reference("smoke", fx)
    rig = RigCalibration.from_kalibr(s.calib_path)
    grid = np.arange(-20, 20.001, 0.25) * 1e-3
    s.imu_t = s.imu_t - shift_ms * 1e-3          # IMU clock lags the others by shift_ms
    m = check_sync.mocap_imu(s, ref, grid)
    assert m["offset_s"] == pytest.approx(shift_ms * 1e-3, abs=0.5e-3)
    assert m["mocap_to_imu_rate_rotation_deg"] < 0.5
    fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], cache=False)
    c = check_sync.camera_imu(s, rig, fe["events"], grid)
    assert c["offset_s"] == pytest.approx(-shift_ms * 1e-3, abs=1.5e-3)
    g = check_sync.gravity(s, ref, m["offset_s"])
    # 6 s of an 8 s-period motion: the mean acceleration is not zero, so only a coarse check.
    assert g["angle_from_plus_z_deg"] < 3.0
