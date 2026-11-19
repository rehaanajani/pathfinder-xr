# Path Finder: Track 3 entry (IMU-only)

Team California Burrito, SRM Institute of Science and Technology. Samsung PRISM XR Hackathon 2026, Track 3.

Reads a degraded IMU stream in EuRoC MAV format (`imu0/data.csv`) and writes the estimated 6-DoF trajectory,
`estimated_trajectory.txt`, in TUM format: one pose per input IMU row, including the 5 s after the IMU dies.

## Run

**Python** (Python 3 + numpy; the network weights are embedded, so
`main.py` can be copied anywhere on its own):

```bash
python3 src/main.py <input> -o estimated_trajectory.txt
```

**Docker** (the image contains only `main.py`):

```bash
docker build -t pathfinder src/
docker run --rm -v "$PWD/data:/data" pathfinder /data -o /data/estimated_trajectory.txt
```

**The C++ port** (same algorithm, output within 1 µm; for on-device use; reads `velnet_weights*.bin` next to it):

```bash
make -C src                                   # needs a C++17 compiler and zlib
src/pathfinder <input> -o estimated_trajectory.txt
```

`<input>` is a folder containing `imu0/data.csv` (or `mav0/imu0/data.csv`), a `.zip` of such a folder, or the CSV
file itself. After retraining the network, `python tools/embed_weights.py` refreshes the copy inside `main.py`.

## Input and output

Input, comma-separated, any header line optional:
`timestamp [ns], w_x, w_y, w_z [rad/s], a_x, a_y, a_z [m/s², including gravity]` (EuRoC).

The organisers' degradation script documents the other column order (`timestamp, accel xyz, gyro xyz`) and accepts
timestamps in s/ms/us/ns and acceleration in g. The entry therefore checks the first rows: the accelerometer is the
triplet that reads gravity (6-14 m/s², or 0.6-1.4 g if neither reads m/s²); epoch-nanosecond timestamps (> 1e17) stay
ns, others take their unit from the row spacing. It says what it found on stderr when it is not the EuRoC layout. To
force it: `--columns gyro-first|accel-first`, `--ts-unit ns|us|ms|s`, `--accel-unit ms2|g` (also accepted by
`tools/make_bundle.py`).

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

## Measured

`main.py`, worst of 5 recordings with the organisers' damage, MacBook (Apple M4), Python 3.11
([`results/track3/python_budget.json`](../results/track3/python_budget.json), `tools/measure_python_budget.py`):
peak heap of the algorithm **1.47 MiB** (5.87 MiB counting everything after numpy, including compiling `main.py`
with its embedded network), CPU **4.9 ms** per second of IMU. The C++ port: 2.5 MB and 1.8 ms on the same Mac, 4.0 MB and 5.0 ms on a Galaxy M36
phone CPU ([`docs/reports/`](../docs/reports/README.md)).

Options: `python3 src/main.py --help` lists them; the C++ entry accepts the same ones.
