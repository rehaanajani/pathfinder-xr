"""Monado headset data (real VR headsets): loader, calibration and stereo odometry.
Skipped where the dataset isn't downloaded (CI)."""

import numpy as np
import pytest

from pathfinder import data_monado as M

HAVE = set(M.available())


@pytest.mark.skipif("MOO11_short_3_backandforth" not in HAVE, reason="Monado data not downloaded")
def test_odyssey_plus_calibration_and_timing():
    s = M.load("MOO11_short_3_backandforth")
    c = M.calib(s)
    baseline = np.linalg.norm(c.T_c1_c0[:3, 3])
    assert 0.08 < baseline < 0.14                         # ~10 cm stereo baseline
    ft, fp = M.frames(s)
    assert len(ft) > 300 and np.all(np.diff(ft) > 0)
    assert abs(np.median(np.diff(ft)) - 1 / 30) < 0.005  # 30 fps, CSV (aligned) clock, not file names


@pytest.mark.skipif("MOO11_short_3_backandforth" not in HAVE, reason="Monado data not downloaded")
def test_stereo_vo_works_on_odyssey_despite_47_degree_cameras():
    # Before calibration-guided stereo matching: 0 of 392 frames solved. Measured now: 44 of the
    # first 119 frame pairs (the start of this clip is hard), about 60% over the whole clip.
    from pathlib import Path
    from pathfinder.stereo_vo import StereoVO, load_pair
    s = M.load("MOO11_short_3_backandforth")
    ft, fp = M.frames(s)
    vo = StereoVO(M.calib(s))
    solved = sum(vo.step(*load_pair(p, Path(s.meta["dir"]) / "cam1" / "data")) is not None for p in fp[:120])
    assert solved >= 30, solved
