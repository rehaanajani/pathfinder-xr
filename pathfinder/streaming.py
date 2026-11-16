"""Causal live wrapper around ``StreamingEstimator`` for a camera front end with processing latency
(issue #3 E). Contract in docs/STREAMING.md.

A stereo frame stamped t is only turned into VO events tens of milliseconds later. The estimator
cannot apply a measurement older than its state (it would be counted ``cam_late`` and dropped), so a
live system must hold IMU samples until the front end has reported every frame up to their time.
``LaggedStream`` does exactly that, with explicit bounds:

* IMU samples wait in a FIFO until ``camera_done(t_frame)`` has covered them, or until they are
  ``max_lag_s`` older than the newest IMU sample (camera stalled/slow: proceed without it; its late
  results are then dropped and counted, never applied out of order);
* the FIFO holds at most ``max_buffer`` samples; on overflow the oldest are released immediately
  (``overflow`` counter) - bounded memory and bounded output latency under overload;
* each output carries ``output_lag_s`` = newest IMU time - output time, so the consumer sees the
  latency it is getting; ``reset()`` starts a new estimator (new odometry frame) and reports it.
Out-of-order or non-finite IMU timestamps are rejected (counted), never reordered.
"""

from collections import deque

import numpy as np

from .estimator import StreamingEstimator


class LaggedStream:
    def __init__(self, make_estimator, max_lag_s: float = 0.15, max_buffer: int = 400):
        if not (np.isfinite(max_lag_s) and max_lag_s >= 0) or int(max_buffer) < 1:
            raise ValueError("max_lag_s must be >= 0 and max_buffer >= 1")
        self.make_estimator = make_estimator
        self.max_lag_s, self.max_buffer = float(max_lag_s), int(max_buffer)
        self.est: StreamingEstimator = make_estimator()
        self.buf = deque()
        self.cam_done_t = -np.inf
        self.newest_t = -np.inf
        self.stats = {"imu_rejected_order": 0, "released_by_camera": 0, "released_by_lag": 0, "overflow": 0,
                      "resets": 0, "max_buffered": 0}

    # ---------------------------------------------------------------- inputs
    def add_imu(self, t, gyro, acc, **quality) -> list:
        # Validate before advancing the clock or touching the FIFO. Non-finite
        # sensor values remain valid inputs: the estimator reports them as faults.
        gyro, acc = np.array(gyro, float), np.array(acc, float)
        if gyro.shape != (3,) or acc.shape != (3,):
            raise ValueError("gyro and acc must each have 3 components")
        if set(quality) - {"q_add_g", "q_add_a", "imu_state", "imu_conf"}:
            raise ValueError("unknown IMU quality field")
        for key in ("q_add_g", "q_add_a"):
            value = quality.get(key, 0.0)
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError("IMU noise densities must be finite/nonnegative")
        conf = quality.get("imu_conf", 1.0)
        if not np.isscalar(conf) or not np.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("IMU confidence must be in [0, 1]")
        state = quality.get("imu_state")
        if state is not None and state not in range(7):
            raise ValueError("unknown IMU monitor state")
        if not np.isfinite(t) or t <= self.newest_t:
            self.stats["imu_rejected_order"] += 1
            return []
        self.newest_t = float(t)
        self.buf.append((float(t), gyro, acc, quality.copy()))
        self.stats["max_buffered"] = max(self.stats["max_buffered"], len(self.buf))
        return self._release()

    def add_vision_events(self, events) -> None:
        events = list(events)
        for e in events:
            if not isinstance(e, dict) or e.get("kind") not in ("kf", "rel"):
                raise ValueError("vision events must be keyframes or relative poses")
            required = {"t", "kf"} | ({"dp", "dR", "cov"} if e["kind"] == "rel" else set())
            if not required <= e.keys():
                raise ValueError("incomplete vision event")
        for e in events:
            if e["kind"] == "kf":
                self.est.add_camera_keyframe(e["t"], e["kf"])
            else:
                self.est.add_camera_relative(e["t"], e["kf"], e["dp"], e["dR"], e["cov"], e.get("health", 1.0))

    def camera_done(self, t_frame) -> list:
        """The front end has finished every frame stamped <= t_frame (events already added)."""
        if np.isfinite(t_frame):
            self.cam_done_t = max(self.cam_done_t, float(t_frame))
        return self._release()

    def reset(self) -> dict:
        """Lifecycle reset (e.g. sensor reconnection): drop buffered samples, new estimator/odometry frame."""
        reset_t = self.newest_t
        dropped = len(self.buf)
        self.buf.clear()
        self.est = self.make_estimator()
        self.cam_done_t = -np.inf
        self.newest_t = -np.inf
        self.stats["resets"] += 1
        return {"event": "reset", "t": reset_t, "dropped_imu": dropped}

    def finish(self) -> list:
        out = []
        while self.buf:
            out += self._emit(self.buf[0])
            self.buf.popleft()
        tail = self.est.finish()
        for row in tail:
            row["output_lag_s"] = self.newest_t - row["t"]
        return out + tail

    # ---------------------------------------------------------------- release policy
    def _release(self) -> list:
        out = []
        while self.buf:
            t = self.buf[0][0]
            if t <= self.cam_done_t:
                self.stats["released_by_camera"] += 1
            elif t <= self.newest_t - self.max_lag_s:
                self.stats["released_by_lag"] += 1
            elif len(self.buf) > self.max_buffer:
                self.stats["overflow"] += 1
            else:
                break
            out += self._emit(self.buf[0])
            self.buf.popleft()
        return out

    def _emit(self, item) -> list:
        t, g, a, q = item
        rows = self.est.add_imu(t, g, a, **q)
        for r in rows:
            r["output_lag_s"] = self.newest_t - r["t"]
        return rows
