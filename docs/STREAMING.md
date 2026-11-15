# Streaming runtime contract

Applies to `pathfinder.estimator.StreamingEstimator` and the live wrapper `pathfinder.streaming.LaggedStream`.
Tested by `tests/test_runtime_safety.py`, `tests/test_relative_pose.py`, `tests/test_streaming.py` and
`tests/test_smoke_pipeline.py`. This is a research runtime in Python. No device integration has been
validated (issue #3 real-device phase).

## Inputs

| Call | Units / frame | Rules |
|---|---|---|
| `add_imu(t, gyro, acc, q_add_g, q_add_a, imu_state, imu_conf)` | s (one clock), rad/s and m/s² specific force in the IMU body frame | Timestamps must be strictly increasing. Non-finite samples or `imu_state=LOST` make the output invalid immediately (`IMU_LOST`). Gaps longer than `max_imu_gap_s` are propagated with envelope noise (`IMU_GAP`). |
| `add_camera_keyframe(t, kf_id)` | IMU clock | Clones the pose at `t`. It is emitted at visual start-up and after every tracking restart, so later relative measurements have an anchor. |
| `add_camera_relative(t, kf_id, dp, dR, cov6, health)` | IMU pose at `t` in the IMU frame of keyframe `kf_id`, metres and a rotation matrix; `cov6` is the covariance of [dp, dθ] with a right-perturbed rotation | Must come from images only (`stereo_vo`). A relative measurement whose keyframe clone is gone is counted `vo_no_anchor` and never applied. It is gated by χ²₆(0.999) and never re-anchors absolute position or heading. |
| `add_camera(t, fix, health)` | world position fix (legacy/simulated mode) | Fix mode and relative mode are mutually exclusive per estimator. |

Camera time = camera stamp + Kalibr `timeshift_cam_imu` (an absent value is recorded as an assumption).
Calibration (intrinsics, distortion model, `T_cam_imu`, `T_cn_cnm1`) is loaded by
`RigCalibration.from_kalibr`. The distortion model always comes from the file: equidistant or
radtan4/5/8. Its SHA-256 is recorded in every run manifest.

## Outputs (one per IMU sample)

These are `t, p, v, q, P_pos, P_vel, valid, track_state, camera_state, imu_state, reasons`
(see `estimator.OUTPUT_FIELDS`, `REASONS`). In relative mode the frame is the **start pose**
(`POSITION_FRAME_ARBITRARY`). Velocity starts unknown (`VELOCITY_ASSUMED_ZERO` until it converges).
`valid` stays false while `COVARIANCE_UNRELIABLE` holds: an uncalibrated learned model without
anchoring, degraded vision, or relative-mode VO with `vo_covariance_validated=False` (the default in
this release). Best-effort positions are still output. Consumers must not treat an invalid pose as
certified.

## Timing, buffering and overload

* **Initialisation** buffers `ReferenceFreeInit.duration_s` (3 s). Those samples are replayed and
  reported `TRACK_INITIALIZING`/invalid.
* **Front-end latency.** VO results for a frame stamped `t` exist only after processing
  (p50 ≈ 33 ms, p99 ≈ 88 ms on the profiled laptop; see the vision report). `StreamingEstimator`
  rejects measurements older than its state (`cam_late`), so a live system wraps it in `LaggedStream`.
  The wrapper holds IMU samples until `camera_done(t_frame)` covers them, or until they are
  `max_lag_s` (default 0.15 s) behind the newest IMU sample. Each output carries `output_lag_s`. With
  `max_lag_s` at least the front-end latency, live output is identical to the zero-latency offline
  replay (tested). With a smaller lag, late measurements are dropped and counted. They are never
  applied out of order.
* **Overload.** At most `max_buffer` IMU samples wait. On overflow the oldest are released without
  camera input (`overflow` counter), so memory and latency stay bounded. A front end that cannot keep
  up therefore degrades to inertial propagation with `CAMERA_OUTAGE` reported. It does not stall.
* **Out-of-order or duplicate IMU stamps** are rejected and counted, never reordered.
* **Stale data.** A camera measurement older than the estimator state is counted `cam_late`.

## Lifecycle

* **Start:** reference-free, with no truth input (`TruthInit` exists only for labelled diagnostics).
* **Visual loss and restart:** the front end emits no `rel` events while tracking is lost. The next
  usable stereo frame emits a `kf` first. Outputs report `CAMERA_OUTAGE` after `outage_gap_s` (0.5 s)
  and `CAMERA_RECOVERING` or nominal afterwards. A relative measurement cannot cause a position jump
  that re-anchors the frame. The measured re-anchoring discontinuity is in the report.
* **Reset / sensor reconnection:** `LaggedStream.reset()` drops the buffer and starts a new estimator,
  which means a new odometry frame. The consumer must treat poses before and after a reset as
  different frames.
* **Relocalisation / loop closure:** not implemented (there is no map). The odometry frame drifts and
  is never corrected.
* **Calibration persistence:** calibration files are inputs with recorded hashes. Changing the front
  end invalidates `calibration/vo_*.json` (source-hash check) and requires recalibration.

## Not provided

Clock synchronisation between independent devices, inter-IMU time-offset estimation, multi-threaded
scheduling, NPU offload, thermal management, and telemetry export. Diagnostics stay in memory
(`FilterLog`). No recording leaves the process by default.
