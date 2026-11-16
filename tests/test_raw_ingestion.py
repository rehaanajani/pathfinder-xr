"""Reject corrupt recording indices before they enter the estimator."""
import numpy as np
import pytest

from pathfinder.data_tumvi_raw import load_streams, load_reference, seq_dir
from pathfinder.smoke import make_fixture


@pytest.fixture
def recording(tmp_path):
    make_fixture(tmp_path, duration=.2)
    return tmp_path, seq_dir("smoke", tmp_path)


def test_stereo_indices_must_agree(recording):
    root, directory = recording
    index = directory / "mav0/cam1/data.csv"
    index.write_text(index.read_text().replace(".png", "-wrong.png"))
    with pytest.raises(ValueError, match="stereo indices"):
        load_streams("smoke", root)


@pytest.mark.parametrize("name", ["../escape.png", "C:\\escape.png", "image.png:stream"])
def test_camera_index_cannot_escape_data_directory(recording, name):
    root, directory = recording
    for camera in ("cam0", "cam1"):
        (directory / f"mav0/{camera}/data.csv").write_text(f"#timestamp,filename\n1000000000,{name}\n")
    with pytest.raises(ValueError, match="basename"):
        load_streams("smoke", root)


def test_nonmonotonic_imu_is_rejected(recording):
    root, directory = recording
    path = directory / "mav0/imu0/data.csv"
    rows = np.loadtxt(path, delimiter=",")
    rows[1, 0] = rows[0, 0]
    np.savetxt(path, rows, delimiter=",")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_streams("smoke", root)


def test_zero_reference_quaternion_is_rejected(recording):
    root, directory = recording
    path = directory / "mav0/mocap0/data.csv"
    rows = np.loadtxt(path, delimiter=",")
    rows[0, 4:8] = 0
    np.savetxt(path, rows, delimiter=",")
    with pytest.raises(ValueError, match="nonzero quaternions"):
        load_reference("smoke", root)
