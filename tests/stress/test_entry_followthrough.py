"""Regressions for the Track 3 entry follow-through: conv workspace and the heading-gauge reset.

The runtime budget itself is test_runtime_under_20ms_per_second_of_imu in test_main_contract.py.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import main as entry  # noqa: E402
from main import G, PathFinder, args  # noqa: E402
from velnet import VelNet, conv1d  # noqa: E402


def test_conv_workspace_matches_conv1d_including_reuse():
    net = VelNet(str(ROOT / "src" / "velnet_weights.bin"))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((net.n_in, net.window)).astype(np.float32)
    first = net._conv_stack(x.copy()).copy()
    second = net._conv_stack(x.copy())          # same buffers, next forward
    ref = x.copy()
    for w, b, stride, pad in net.convs:
        ref = np.maximum(conv1d(ref, w, b, stride, pad), 0)
    assert np.array_equal(first, ref)
    assert np.array_equal(second, ref)


@pytest.mark.skipif(entry._HOT is None, reason="build with python src/build_hotstep.py")
@pytest.mark.parametrize("extra", [(), ("--model", "none", "--preset", "physics"), ("--extend-blind", "1")])
def test_native_and_python_cli_match_faults_across_batch_boundaries(tmp_path, extra):
    from tests.stress.test_main_contract import walk_rows, write_csv
    ts, g, a, _ = walk_rows(2400, seed=3)
    g[500:530] = 0; a[500:530] = 0
    g[1010:1030] = np.nan; a[1010:1030] = np.nan
    g[1530:1550] = g[1529]; a[1530:1550] = a[1529]
    keep = np.r_[0:1800, 1900:2400]
    source = tmp_path / "imu.csv"
    write_csv(source, ts[keep], g[keep], a[keep])
    paths = []
    for native in (True, False):
        out = tmp_path / f"{native}.txt"
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("PF_NO_HOT", None)
        if not native:
            env["PF_NO_HOT"] = "1"
        subprocess.run([sys.executable, str(ROOT / "src/main.py"), str(source), "-o", str(out), *extra], env=env, check=True, capture_output=True)
        paths.append(np.loadtxt(out))
    assert paths[0].shape == paths[1].shape
    assert np.isfinite(paths[0]).all()
    np.testing.assert_allclose(paths[0][:, 0], paths[1][:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(paths[0][:, 1:4], paths[1][:, 1:4], rtol=0, atol=1.000001e-6)
    np.testing.assert_allclose(paths[0][:, 4:], paths[1][:, 4:], rtol=0, atol=1.000001e-8)


@pytest.mark.skipif(entry._HOT is None, reason="build with python src/build_hotstep.py")
def test_native_state_can_be_closed_twice():
    pf = PathFinder(args(["unused.csv"]))
    pf._hot_open()
    assert pf._hot
    pf.close()
    pf.close()
    assert pf._hot is None


def _imu(i):
    """Healthy, not frozen, about 1 g. Varies every sample so the frozen-repeat check stays quiet."""
    g = (0.01 * math.sin(i * 0.017), 0.008 * math.cos(i * 0.013), 0.004 * math.sin(i * 0.011))
    a = (0.05 * math.sin(i * 0.019), -0.04 * math.cos(i * 0.015), G + 0.03 * math.sin(i * 0.009))
    return g, a


def _run_until(pf, n, t, i0, zero=False):
    dt = 5_000_000
    yaw_at_update = []
    real = PathFinder.learned_update

    def wrapped(self):
        before = self.yaw_var
        real(self)
        if self.since == 0 and self.net_ok:
            yaw_at_update.append((self.good_time, before, self.yaw_var))

    PathFinder.learned_update = wrapped
    try:
        for k in range(n):
            i = i0 + k
            g, a = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) if zero else _imu(i)
            ok = pf.healthy(g, a)
            if not pf.initialised:
                if ok:
                    pf.init_buf.append((g, a))
                if len(pf.init_buf) >= pf.a.init_samples:
                    pf.init_from(pf.init_buf)
            if pf.initialised:
                pf.step(t, g, a, ok)
            t += dt
    finally:
        PathFinder.learned_update = real
    return t, i0 + n, yaw_at_update


def test_yaw_variance_survives_a_healthy_window_after_a_dropout():
    """Heading variance accumulated in a dropout is not cleared when healthy data returns.

    Clearing it after one network window was tried. On both exposed rooms it raised the blind
    5 s error (and room 2's mission error). The entry keeps the variance, so a later healthy
    update is still shrunk. See results/runs/20261117-yaw-gauge/REPORT.md.
    """
    pf = PathFinder(args([str(ROOT / "src" / "velnet_weights.bin")]))
    # args() treats the positional as the input path and still loads the model next to main.py.
    assert pf.net is not None
    span = pf.W * pf.dec / pf.net.rate
    t, i, _early = _run_until(pf, 600, 0, 0)         # init plus 2 s healthy; yaw still 0 on a clean stream
    assert pf.yaw_var == 0.0
    t, i, _dropped = _run_until(pf, 80, t, i, zero=True)  # 0.4 s dead IMU
    assert pf.yaw_var > 0.0
    held = pf.yaw_var
    _t, _i, late = _run_until(pf, 400, t, i)         # 2 s healthy: past one window
    assert pf.good_time >= span
    assert late, "the network should have updated again after the window refilled"
    assert late[-1][0] >= span
    assert late[-1][1] >= held * 0.5                 # variance presented to the update is still the dropout's
    assert late[-1][2] > 0.0
    assert pf.yaw_var > 0.0


def test_yaw_q_blind_does_not_move_a_trajectory_with_no_dropout(tmp_path):
    """Clearing a variance that never grew is a no-op. Two clean runs, one with the blind
    growth forced to 0, must match to the printed micrometre."""
    sys.path.insert(0, str(ROOT))
    from tests.stress.test_main_contract import walk_rows, write_csv, run

    ts, g, a, _ = walk_rows(200 * 8, seed=1)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    p0, _, _ = run(csv, tmp_path / "a.txt", ("--yaw-q-blind", "0"))
    p1, _, _ = run(csv, tmp_path / "b.txt")
    assert p0.returncode == 0 and p1.returncode == 0
    assert (tmp_path / "a.txt").read_bytes() == (tmp_path / "b.txt").read_bytes()
