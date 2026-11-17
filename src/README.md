# Path Finder: Track 3 entry (IMU-only)

Team California Burrito, SRM Institute of Science and Technology. Samsung PRISM XR Hackathon 2026, Track 3.

Reads a degraded IMU stream in EuRoC MAV format (`imu0/data.csv`) and writes the estimated 6-DoF trajectory,
`estimated_trajectory.txt`, in TUM format: one pose per input IMU row, including the 5 s after the IMU dies.

## Run

**C++ (the submission; ≤ 8 MB, ≤ 20 ms per second of IMU, one thread):**

```bash
make -C src                                   # needs a C++17 compiler and zlib
src/pathfinder <input> -o estimated_trajectory.txt
```

**Docker:**

```bash
docker build -t pathfinder src/
docker run --rm -v "$PWD/data:/data" pathfinder /data -o /data/estimated_trajectory.txt
```

**Python reference (same output; needs numpy for the network):**

```bash
python3 src/main.py <input> -o estimated_trajectory.txt
```

`<input>` is a folder containing `imu0/data.csv` (or `mav0/imu0/data.csv`), a `.zip` of such a folder, or the CSV
file itself.

## Input and output

Input, comma-separated, `#` header optional:
`timestamp [ns], w_x, w_y, w_z [rad/s], a_x, a_y, a_z [m/s², including gravity]`.

Output, space-separated, with a `# timestamp tx ty tz qx qy qz qw` header: timestamp in seconds (input ns × 1e-9),
position in metres, unit quaternion. The world frame has gravity along −z; origin and heading are those of the first
sample (an IMU alone cannot observe them).

Rows that are missing values, NaN/inf, all zero, frozen (bit-identical repeats) or physically impossible still get
a pose: the estimator dead-reckons through them. If the file simply ends when the IMU dies, `--extend-blind 5`
appends 5 s of predicted poses after the last row.

## How it works

1. **Sample health**: bad readings and timestamp gaps are detected per row.
2. **Start without ground truth**: gravity direction and gyro bias come from the data (the first still stretch).
3. **Strapdown integration** with gravity correction (complementary filter), gyro-bias learning, zero-velocity
   updates when still, residual-acceleration removal, a soft speed limit and a pull back toward the start point.
4. **Learned velocity**: a 42,716-parameter 1-D CNN (`velnet_weights.bin`, 171 KB) reads the last second of
   body-frame IMU with gravity kept plus our own "up" direction (AirIO style, `docs/AIRIO.md`) and predicts
   body-frame displacement and velocity with their uncertainty; rotated into the world frame, these correct the
   filter, weighted by that uncertainty. On by default; `--model velnet_weights_world.bin` is the previous
   gravity-aligned network, `--model none --preset physics` the physics-only filter.
   **Mount guard:** the body-frame network learned the training IMUs' axes, so if gravity at start-up points more
   than 25° away from every training mount (a rotated IMU), the entry switches to `velnet_weights_world.bin` and says
   so on stderr (`--mount-guard-deg 0` turns it off; `docs/MOUNT_GUARD.md`). Keep both weight files next to the binary.
5. **Dropouts and IMU death**: dead reckoning from the last trusted state, with turn rate and velocity decaying.

`main.py` is the commented reference; `main.cpp` is a line-by-line port (exact match without the network, within
1 µm with it).

The Python entry can run its strapdown step through an optional native C helper;
the learned update stays in Python. Build from the repository root with
`python src/build_hotstep.py` (MSVC C++ tools on Windows, a C compiler on Linux/macOS).
The build emits a platform library plus source/binary hashes and an ABI identifier.
Compiled binaries are not committed. CI builds them from source. Missing libraries
use the Python fallback; incompatible or stale libraries warn and use that fallback.
`PF_NO_HOT=1` explicitly selects the pure-Python implementation.

The native path processes bounded batches of 512 samples after initialization and
releases its state when the input finishes. The conv stack reuses its workspaces.
Report whether the helper was loaded when quoting Python runtime: accelerated
Python and pure Python are distinct benchmark configurations. Neither is the
standalone C++ submission. Cross-platform parity tests include missing, zero,
NaN and frozen data across batch boundaries, plus explicit blind extension.

A dropout still leaves a heading variance that is never cleared: the horizontal network update stays shrunk for
the rest of the file. Clearing that variance after one healthy window was measured on both exposed rooms and
made the blind 5 s and, on room 2, the mission error worse, so it is not what the entry does
(`results/runs/20261117-yaw-gauge/REPORT.md`).

## Measured (Apple M5, macOS; EuRoC / TUM-VI degraded with our stand-in for the organisers' script)

| | C++ | Python |
|---|---:|---:|
| Peak RAM | 2.2-2.3 MB | 41 MB |
| Time per second of IMU | 2.0 ms | 5.7 ms |

C++ re-measured on 16 Nov with the AirIO network on degraded MH_05 and TUM-VI room6. On a Windows laptop the C++
entry takes 6.0 ms per second of IMU with a 5.2 MB peak working set, most of it the Windows static runtime
(`docs/AIRIO.md`); Docker on Linux 3.2 MB (`docs/TRACK3_EVAL.md`).

Options: `python3 src/main.py --help` lists them; the C++ entry accepts the same ones.
