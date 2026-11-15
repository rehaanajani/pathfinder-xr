# Mount guard: keep the AirIO network's gain without its rotated-IMU risk

The body-frame (AirIO-style) network in `src/velnet_weights.bin` learned the training IMUs' axes. On EuRoC and TUM-VI
as recorded it beats the old gravity-aligned network (held-out ATE 3.08 vs 3.19 m, `docs/AIRIO.md`). If the
organisers' "unknown gravity alignment" means the IMU axes are rotated, it loses badly on the blind 5 s.

**The guard.** At start-up the entry already measures gravity in the IMU frame (to find "down"). Gravity at the start
of every training recording sits in two tight clusters:

| Recordings | "Up" in the IMU frame at start |
|---|---|
| EuRoC, all 11 | (+0.94, −0.01, −0.34), every sequence within ~6° |
| TUM-VI rooms 1-6 | (0.02, 0.00, 1.00) |

If the new file's start-up gravity is more than 25° from both (`--mount-guard-deg`, 0 = off), the entry loads
`velnet_weights_world.bin` (the gravity-aligned network, which does not depend on the mount) before any network input
is collected, and says so on stderr. A rotation about gravity itself cannot be seen this way.

## Measured (C++ entry, `tools/degrade_imu.py`)

- **Normal data:** on all 16 recordings (11 EuRoC + TUM-VI rooms 1-4, 6, seed 1) the guard stays quiet and the output
  is byte-identical to the guard switched off. Every AirIO number stands.
- **Rotated data:** validation MH_04 + V1_03, seeds 1-2, IMU frame rotated (90° about x as in `tools/make_rotated.py`,
  and three random rotations). The guard fires on all four. Checkpoints / blind 5 s / dropouts, metres:

| Rotation | Start-up angle from EuRoC mount | AirIO only | Guard (= old network) |
|---|---:|---|---|
| 90° about x | 28° | 3.81 / 2.94 / 4.21 | 3.93 / **2.66** / 4.27 |
| random 0 | 140° | 3.90 / 2.84 / 4.25 | 3.93 / **2.62** / 4.26 |
| random 1 | 111° | 3.95 / 2.72 / 4.31 | 3.97 / **2.55** / **4.20** |
| random 2 | 170° | 3.56 / 3.54 / 4.31 | 3.94 / **2.59** / **4.26** |

On rotated data AirIO keeps a small checkpoint edge but its blind 5 s collapses (up to 3.54 vs 2.59 m). Weighted by
the rubric (checkpoints 30, blind 20, dropouts 10), the guard is ahead in every rotated case and costs nothing on
normal data.

Tests: `tests/stress/test_mount_guard.py` (both training mounts stay on AirIO; flipped, y-up and x-down switch; the
guard can be turned off). Python and C++ agree within 1e-6 m on a rotated recording.
