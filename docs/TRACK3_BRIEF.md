# Track 3 official brief (what Samsung will actually score)

Source: *XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf* (Samsung R&D Institute India, Bangalore), sent
by Dr. S. Poornima (SRMIST PRISM coordinator) on 15 Nov 2026, slides 14-18 and 25-27. This page restates it; the PDF
is the authority.

## The task

> In GPS-denied and hostile environments, an aeroplane navigates through waypoints capturing reconnaissance
> images. As it proceeds, enemy jamming causes escalating IMU noise, bias drift, and frame dropouts until the IMU
> sensor fails entirely.

1. **Estimate the 6-DoF trajectory** (position + orientation) through all mission checkpoints **using only the
   degraded IMU stream**.
2. **After complete IMU failure** (after the final checkpoint), **dead-reckon the next 5 seconds** of the path.

Degradation: progressive noise, bias drift, frame dropouts (including **2-second signal loss**), **unknown gravity
alignment**, then total failure. Datasets: **EuRoC MAV and TUM-VI**. An organizer degradation script is mentioned
("provided") — we have not received it; see *Assumptions*.

## Input (slide 16)

Zipped folder, EuRoC MAV format, file `imu0/data.csv`, comma-separated:

```
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,0.0124,-0.0023,0.0051,0.1241,0.0312,9.8105
```

Timestamps: 64-bit integer nanoseconds since Unix epoch. Gyro in rad/s, accelerometer in m/s² **including gravity**.
Non-conforming submissions fail the automated pipeline.

## Output (slide 17)

`estimated_trajectory.txt`, TUM format, space-separated, optional `#` header, exactly 8 columns:

```
# timestamp tx ty tz qx qy qz qw
1403636579.7585 0.0012 0.0045 0.0102 0.0001 0.0023 0.0011 0.9999
```

- timestamp in **seconds** (input ns × 1e-9), position in metres, unit quaternion (world frame).
- **Exactly one pose per IMU frame.** Dropped or misaligned rows incur heavy penalties or failure.

## Budgets and deliverable (slides 14-15)

- **≤ 8 MB peak RAM, ≤ 20 ms compute per 1 s of IMU, single CPU thread, offline.**
- A **standalone script**: degraded IMU CSV → 6-DoF trajectory (`src/main.py` or `src/main.cpp`).
- Pipeline walkthrough video, 2-3 min.

## Scoring (100 points)

| Part | Points |
|---|---:|
| Execution & efficiency: peak memory | 20 |
| Execution & efficiency: runtime | 20 |
| Trajectory closeness: waypoints / checkpoints (T1) | 30 |
| Trajectory closeness: blind 5 s after failure (T2) | 20 |
| Trajectory closeness: dropout resilience (T3) | 10 |

Ranked with the open-source **`evo`** suite against a hidden high-accuracy ground truth: **ATE (RMSE)** for global
consistency and **RPE** for local drift and rotation error.

## Submission bundle (slide 18) and general rules (slide 25)

```
team_name.zip
├── estimated_trajectory.txt   # final generated trajectory
├── metadata.json              # optional: runtime perf, compute specs
└── src/
    ├── main.py / main.cpp     # entry point / build system
    └── README.md              # how to compile & run
```

Plus, for every team: working code in a **public or shared GitHub repo**; README with reproducible setup **and
Docker files**; **demo video ≤ 3 min** (YouTube or Drive link); presentation (PPT or PDF) named
**`CollegeName_TeamName`** → `SRMIST_CaliforniaBurrito`; one submission per team via the Google Form. The
presentation covers: theme ID, title, team; problem in our own words; solution and architecture diagram; tools and
tech stack; innovation highlights, results and limitations. *Not following the guideline = disqualification.*

## Dates

| | Deck (slide 27, "tentative") | Earlier email (Prakhar, 12 Nov) |
|---|---|---|
| Final submission | **18 Nov 2026, 11:59 PM** | 22 Nov 2026, 11:59 PM |
| Top 15 announced | 23 Nov | 29 Nov (top 12) |
| Final demo | 27 Nov | 9 Dec |
| Results | 2 Dec | 12 Dec |

**We plan for 18 Nov.** Prizes: ₹1.0 L for the top 4, PRISM worklet and possible Samsung internship.

## What this changes for us

**Scored now:** an IMU-only estimator in the exact I/O format, fast and small. **Not scored (innovation material
for the slides, video and live demo):** camera health, visual odometry, real VR headsets, the live phone demo,
multi-IMU fault isolation, OpenXR-style flags.

| Gap | Plan |
|---|---|
| 5 s prediction after IMU death (20 pts) | Motion-model extrapolation from the last trusted state (velocity / turn-rate decay, speed limit) |
| Start with no ground truth, unknown gravity | Gravity and gyro bias from the first static period; world frame fixed by gravity |
| Exact I/O, one pose per row, through dropouts | `src/main.py` streams the CSV and writes a pose for every row, bridging gaps |
| ≤ 8 MB, ≤ 20 ms/s, 1 thread (40 pts) | Streamed, bounded-memory design (one short pre-scan for gyro bias); C++ `src/main.cpp` port (Python itself exceeds 8 MB) |
| EuRoC data | Download IMU + ground truth for all 11 sequences |
| Their degradation | `tools/degrade_imu.py` following the description until we get theirs |
| Their scoring | `evo` ATE/RPE harness with waypoint, blind-5 s and dropout sub-scores |
| Packaging | Dockerfile, `team_name.zip`, README, video, PPT name |

## Assumptions (to replace when the organizer script/kit arrives)

- **Checkpoints** are times along the hidden ground truth where the pose error is measured; their positions are
  *not* given to us.
- **IMU death**: the input either keeps rows for the blind 5 s with invalid values (NaN / zeros / frozen), or ends.
  `main.py` handles both: it outputs one pose per input row, and with `--extend-blind 5` appends 5 s of poses after
  the last row.
- **Dropouts**: rows missing (timestamp gaps) or values frozen/zero/NaN for ~2 s.
- **Evaluation alignment**: `evo` with alignment (the world frame of an IMU-only estimate is arbitrary in yaw and
  origin). We report aligned and unaligned numbers.
