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
5. **Dropouts and IMU death**: dead reckoning from the last trusted state, with turn rate and velocity decaying.

`main.py` is the commented reference; `main.cpp` is a line-by-line port (exact match without the network, within
1 µm with it).

## Measured (Apple M5, macOS; EuRoC / TUM-VI degraded with our stand-in for the organisers' script)

| | C++ | Python |
|---|---:|---:|
| Peak RAM | 2.2 MB | 28 MB |
| Time per second of IMU | 4 ms | 4.5 ms |

These were measured with the earlier 27,020-parameter network. On a Windows laptop where that network takes 4.1 ms
per second of IMU in C++, the AirIO network takes 6.0 ms, peak working set about 5.2 MB for both
(`docs/AIRIO.md`); not yet re-measured on the M5.

Options: `python3 src/main.py --help` lists them; the C++ entry accepts the same ones.
