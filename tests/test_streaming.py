"""Live streaming contract with a slow camera front end (issue #3 E, docs/STREAMING.md)."""

import numpy as np
import pytest

from pathfinder import data_tumvi_raw as D
from pathfinder.camera_model import RigCalibration
from pathfinder.estimator import ReferenceFreeInit, StreamingEstimator, replay
from pathfinder.filter import FilterConfig
from pathfinder.smoke import make_fixture
from pathfinder.streaming import LaggedStream
from pathfinder.vision_frontend import run_frontend


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("stream")
    make_fixture(root, duration=5.0)
    s = D.load_streams("smoke", root)
    rig = RigCalibration.from_kalibr(s.calib_path)
    fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], cache=False)
    return s, fe["events"]


def _make():
    return StreamingEstimator(FilterConfig(), ReferenceFreeInit(), camera_mode="relative")


def _live(s, events, latency, max_lag):
    """Frame stamped t finishes processing at t + latency (IMU clock); IMU arrives on time."""
    by_frame = {}
    for e in events:
        by_frame.setdefault(e["t"], []).append(e)
    frames = sorted(set(s.frame_t) | set(by_frame))
    ready = [(t + latency, t) for t in frames]
    stream = LaggedStream(_make, max_lag_s=max_lag)
    rows, j = [], 0
    for i, t in enumerate(s.imu_t):
        while j < len(ready) and ready[j][0] <= t:
            stream.add_vision_events(by_frame.get(ready[j][1], []))
            rows += stream.camera_done(ready[j][1])
            j += 1
        rows += stream.add_imu(t, s.gyro[i], s.acc[i])
    live = len(rows)
    rows += stream.finish()
    return stream, rows, live


def test_lagged_stream_matches_zero_latency_replay_when_lag_covers_the_front_end(data):
    s, ev = data
    _, _, ref = replay(s.imu_t, s.gyro, s.acc, FilterConfig(), ReferenceFreeInit(), vision=ev)
    stream, rows, live = _live(s, ev, latency=0.04, max_lag=0.10)
    assert len(rows) == len(s.imu_t) and stream.est.log.cam_late == 0
    p = np.array([r["p"] for r in rows])
    np.testing.assert_allclose(p, ref["p"], atol=1e-9)
    # Steady state (after the initialisation buffer is replayed, before the end-of-stream flush):
    # the front-end latency is paid, and bounded by max_lag_s.
    t_init = s.imu_t[0] + ReferenceFreeInit().duration_s + 0.2
    lag = np.array([r["output_lag_s"] for r in rows[:live] if r["t"] > t_init])
    assert len(lag) > 200 and lag.max() <= 0.10 + 0.0051 and np.median(lag) > 0.03


def test_too_small_lag_drops_late_measurements_instead_of_applying_them_out_of_order(data):
    s, ev = data
    stream, rows, _ = _live(s, ev, latency=0.04, max_lag=0.01)
    assert len(rows) == len(s.imu_t)
    assert stream.est.log.cam_late > 0 and stream.stats["released_by_lag"] > 0


def test_overflow_bound_reset_and_order_rejection(data):
    s, _ = data
    stream = LaggedStream(_make, max_lag_s=10.0, max_buffer=50)        # camera never reports
    n = 0
    for i in range(300):
        n += len(stream.add_imu(s.imu_t[i], s.gyro[i], s.acc[i]))
    assert len(stream.buf) <= 50 and stream.stats["overflow"] > 0
    assert stream.add_imu(s.imu_t[10], s.gyro[10], s.acc[10]) == []   # out of order: rejected, not reordered
    assert stream.stats["imu_rejected_order"] == 1
    info = stream.reset()
    assert info["dropped_imu"] > 0 and len(stream.buf) == 0 and stream.stats["resets"] == 1
    with pytest.raises(ValueError):
        LaggedStream(_make, max_lag_s=-1)


def test_reset_accepts_restarted_clock_and_preserves_reset_event_time():
    stream = LaggedStream(_make, max_lag_s=100)
    for i in range(10):
        stream.add_imu(1.0 + i * .005, [0, 0, 0], [0, 0, 9.81])
    event = stream.reset()
    assert event["t"] == pytest.approx(1.045)
    stream.add_imu(0.0, [0, 0, 0], [0, 0, 9.81])
    stream.add_imu(.005, [0, 0, 0], [0, 0, 9.81])
    assert stream.stats["imu_rejected_order"] == 0
    assert len(stream.buf) == 2
    stream.add_imu(.005, [0, 0, 0], [0, 0, 9.81])
    assert stream.stats["imu_rejected_order"] == 1


@pytest.mark.parametrize("gyro,quality", [([0, 0], {}), ([0, 0, 0], {"q_add_g": -1}),
    ([0, 0, 0], {"imu_conf": 2}), ([0, 0, 0], {"imu_state": 9}),
    ([0, 0, 0], {"unexpected": 1})])
def test_bad_sample_does_not_mutate_buffer_or_clock(gyro, quality):
    stream = LaggedStream(_make, max_lag_s=10)
    stream.add_imu(0., [0, 0, 0], [0, 0, 9.81])
    with pytest.raises(ValueError):
        stream.add_imu(.1, gyro, [0, 0, 9.81], **quality)
    assert stream.newest_t == 0 and len(stream.buf) == 1
    stream.add_imu(.1, [0, 0, 0], [0, 0, 9.81])
    rows = stream.finish()
    assert len(rows) == 2
    assert [r["output_lag_s"] for r in rows] == pytest.approx([.1, 0])


def test_invalid_vision_batch_queues_nothing():
    stream = LaggedStream(_make)
    with pytest.raises(ValueError):
        stream.add_vision_events([{"kind": "kf", "t": 0, "kf": 0}, {"kind": "typo"}])
    assert not stream.est.pending


def test_failed_estimator_call_preserves_fifo_sample(monkeypatch):
    stream = LaggedStream(_make, max_lag_s=10)
    stream.add_imu(0., [0, 0, 0], [0, 0, 9.81])
    def fail(*args, **kwargs):
        raise RuntimeError("estimator failure")
    monkeypatch.setattr(stream.est, "add_imu", fail)
    with pytest.raises(RuntimeError):
        stream.finish()
    assert len(stream.buf) == 1
