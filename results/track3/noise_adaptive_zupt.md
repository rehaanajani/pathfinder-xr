# Noise-adaptive still detection: tested, **not kept**

**Idea** (brief item B1; vortexxx scales its process noise by the measured excess gyro scatter): our zero-velocity
detector uses fixed per-axis thresholds (gyro std 0.01 rad/s, accel std 0.15 m/s²), but the organisers' white noise
grows to 0.015 rad/s and 0.2 m/s², so late in a file (where the IMU dies, often after landing) rest can no longer be
detected. Implementation (commit "Added an optional noise-adaptive still detector ...", reverted in the next commit):
excess noise per sensor = row-to-row variance smoothed over 2 s minus its quietest value in the first 10 s (white-noise
scale), and the still thresholds become sqrt(threshold² + (c × excess)²).

**Why it fails.** The measured excess is not just damage. On the drone it is mostly rotor vibration: on V1_01 the
accelerometer excess is about 1 m/s² from take-off on (gyro 0.03-0.06 rad/s), while on TUM-VI room1 it follows the
damage ramp (gyro 0.012 → 0.035 rad/s, accel 0.04 → 0.22 m/s²). Widened thresholds then call hover "still". c = 0.5
changes nothing; c ≥ 1 makes every metric worse on the development set. Not checked on held-out data.
"entry" is `main` with dropout bridging (`gap_bridge.md`).


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | zupt-adapt 0.5 | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | zupt-adapt 1 | 1.920 | 1.722 | 2.253 | 1.723 | 0.483 | 3.91 / 4.50 | 1.74 / 2.18 | 0.60 / 0.64 | 11/11 |
| dev, seed 42 | zupt-adapt 2 | 2.079 | 1.846 | 2.375 | 1.857 | 0.598 | 4.07 / 4.53 | 1.96 / 2.30 | 0.71 / 0.83 | 11/11 |
| dev, seed 42 | zupt-adapt 3 | 2.176 | 1.969 | 2.327 | 1.927 | 0.643 | 4.09 / 4.54 | 1.96 / 2.12 | 0.95 / 0.88 | 11/11 |

