# Track 3 evaluation: real `evo`, blind-5 s study, Docker

Branch `eval-mayank`, on `track3-entry` @ 332b4df. Run on an Apple M4 (macOS, arm64), `evo` 1.38.1, Python 3.11.
Everything below is the **C++ entry** (`src/pathfinder`, built with `make -C src`), single thread. Inputs are the 11 EuRoC
sequences degraded with `tools/degrade_imu.py` (our stand-in for the organisers' script, defaults, seeds 1, 2, 3, so
33 runs). The organisers' script has not arrived, so absolute numbers are not what the judges will see.

**TUM-VI: not scored.** There are no TUM-VI rooms on this machine (`local_data/` has the 11 EuRoC sequences and TLIO only).
I did not download them. `tools/eval_evo.py` is EuRoC-only for now.

## Findings

1. **The entry is only 2.6 % better than standing still on ATE.** Mean `evo_ape -a` is 3.35 m for ours and
   3.44 m for stand still (`--v-max 0.0000001`). Ours wins on SE(3) ATE in 33/33 runs and on RPE in 33/33,
   but with Sim(3) alignment ours wins only 21/33 (stand still's `-as` is lower on average for MH_03, V1_02, V2_02 and V2_03). On EuRoC, with this degradation, the ATE is mostly the size of
   the room, not the filter.
2. **Dead reckoning after the IMU dies does not beat holding still** (60 random kills, section 3): 1.32 m vs 1.31 m error
   of the motion since the kill. It is not hurting much either: the worst single loss is +0.15 m.
3. **The C++ entry meets both budgets in the container:** peak RSS 3.2 MB (limit 8 MB), 4.2 ms per second of IMU
   (limit 20 ms), one pose per input row, and the memory does not grow with the length of the file (10 x longer file: 3.2 MB).
4. **`score_track3.py` and `evo` agree on ATE to 0.01 % but disagree on RPE by more than 5 % in 6 of 33 runs** (section 2).
   The reason is the definition (below), not a bug.
5. **PR #24 review:** all seven fixes verified, approved (section 1).

## 1. PR #24 (fixes for #13-#19)

`STRESS_SLOW=1 pytest tests/stress` on `stress-fixes` @ 9700374 with both phone sessions in `runs/live`
(release `phone-sessions-v1`): **61 passed, 1 skipped, 6 xfailed, 16 min.** The skip is the fuzz with the learned model (not
downloaded on this machine). The 6 xfails are limits I measured but did not file as bugs: out-of-order delivery and
accelerometer clock offset in the live server (sessions A and B), a camera pose that never changes is still trusted, and ARCore
never tracking buffers IMU without bound. Each fix was also re-run against its issue's repro (#13 and #14 scripts no longer
raise; #15 on TLIO val, blind from 0 s: `valid=False` at 10, 20 and 40 s on all four clips, claimed std 2.6 / 5.2 / 10.4 m against a
real 5.6-11.4 / 9.9-18.4 / 1.6-16 m). Review: <https://github.com/rehaanajani/pathfinder-xr/pull/24>, approved with three
non-blocking notes (the uncertainty floor is still 2-4 x too small at 10 s; `well_formed` drops unknown message kinds; exact-zero IMU
rows now count as bad).

```bash
git worktree add ../stressfix-wt origin/stress-fixes && cd ../stressfix-wt
ln -s <main checkout>/runs runs        # phone sessions: gh release download phone-sessions-v1 -D runs/live
STRESS_SLOW=1 PYTHONPATH=. python -m pytest tests/stress -v
```

## 2. evo scoring

Commands (all from the repo root, venv with `evo` installed):

```bash
make -C src
python tools/eval_evo.py --data MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
    V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult --seeds 1 2 3 --tag eval --jobs 3
python tools/eval_tables.py            # prints the tables below
```

For each recording and seed the script runs `tools/degrade_imu.py <mav0> <dir> --seed N`, `src/pathfinder <dir> -o est.txt`
and the stand-still run (`... --v-max 0.0000001`), converts the ground truth to TUM, and calls the real tools:

```bash
evo_ape tum gt.txt est.txt -a  --t_max_diff 0.01        # SE(3) alignment
evo_ape tum gt.txt est.txt -as --t_max_diff 0.01        # Sim(3) alignment
evo_rpe tum gt.txt est.txt -d 200 -u f --t_max_diff 0.01   # translation RPE over 200 frames (1 s at 200 Hz)
```

`evo_rpe` has no seconds unit, so 1 s is 200 frames (a dropout of missing rows makes a few windows slightly longer than
1 s). `evo_ape -a` aligns on the whole trajectory, including the blind 5 s, as the brief's tool would. All 33 + 33 runs
exited 0 and wrote one pose per input row (`rows_out == rows_in`, timestamps equal, unit quaternions).

Mean of 3 seeds (m). The seeds differ by at most 0.40 m on any metric; raw per-seed numbers are in `docs/track3_eval_results.json` (blind study: `docs/track3_blind_results.json`).

| recording | evo ATE -a (ours) | ATE -as (ours) | evo RPE 1 s (ours) | ATE -a (still) | ATE -as (still) | RPE 1 s (still) |
|---|---:|---:|---:|---:|---:|---:|
| MH_01_easy | 4.24 | 3.86 | 0.48 | 4.36 | 4.34 | 0.52 |
| MH_02_easy | 4.55 | 4.42 | 0.58 | 4.66 | 4.66 | 0.58 |
| MH_03_medium | 3.44 | 3.17 | 1.19 | 3.63 | 3.16 | 1.28 |
| MH_04_difficult | 6.63 | 6.32 | 1.22 | 6.74 | 6.48 | 1.24 |
| MH_05_difficult | 6.72 | 6.46 | 1.14 | 6.80 | 6.62 | 1.15 |
| V1_01_easy | 1.79 | 1.76 | 0.47 | 1.85 | 1.76 | 0.48 |
| V1_02_medium | 1.70 | 1.70 | 0.93 | 1.78 | 1.62 | 1.05 |
| V1_03_difficult | 1.50 | 1.45 | 0.71 | 1.60 | 1.58 | 0.73 |
| V2_01_easy | 2.21 | 2.14 | 0.36 | 2.29 | 2.19 | 0.37 |
| V2_02_medium | 2.04 | 2.04 | 0.71 | 2.11 | 1.92 | 0.78 |
| V2_03_difficult | 2.02 | 2.02 | 0.79 | 2.05 | 1.95 | 0.84 |
| **mean (33 runs)** | **3.35** | **3.21** | **0.78** | **3.44** | **3.30** | **0.82** |

### Where `score_track3.py` and `evo` disagree by more than 5 %

ATE agrees to 0.01 % on all 33. RPE differs by more than 5 % on these runs (the larger of the entry's and stand still's difference is shown):

| recording | seed | evo ATE -a | score_track3 ATE | diff | evo RPE | score_track3 RPE | diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| MH_02_easy | 2 | 4.50 | 4.50 | -0.0 % | 0.55 | 0.58 | +5.2 % |
| V1_03_difficult | 1 | 1.49 | 1.49 | +0.0 % | 0.71 | 0.76 | +6.9 % |
| V1_03_difficult | 2 | 1.49 | 1.49 | +0.0 % | 0.71 | 0.75 | +5.7 % |
| V1_03_difficult | 3 | 1.50 | 1.50 | +0.0 % | 0.70 | 0.75 | +6.3 % |
| V2_03_difficult | 1 | 2.01 | 2.01 | +0.0 % | 0.79 | 0.89 | +13.0 % |
| V2_03_difficult | 2 | 2.06 | 2.06 | +0.0 % | 0.78 | 0.84 | +8.3 % |

Cause: `tools/score_track3.py` measures RPE as the difference of **world-frame** displacements after an alignment fitted on
the mission part only, whereas `evo_rpe -r trans_part` rotates the relative translation into the **estimate's own frame**
at the start of each window, so a heading error does not count. Recomputing evo's definition in numpy on V2_03_difficult s1
gives 0.789 (evo: 0.79) and V1_03_difficult s1 0.708 (evo: 0.71); the world-frame number is 0.82 / 0.86 unaligned, so the gap
is largest on the recordings with fast turns (V1_03, V2_03). Neither is wrong; if the judges use the `evo` default, trust the
`evo` column.

## 3. Blind 5 s study

Three recordings (MH_01_easy, V1_02_medium, V2_03_difficult), 20 random kill times each, drawn uniformly in [20 s, T-10 s]
(`numpy` seed 7, 8, 9). For each kill the recording is degraded with seed 1 and `--blind T-t_kill` (the noise ramp ends at the
kill; zeros afterwards), cut 5 s after the kill, and run through the entry. SE(3) alignment is fitted on the part **before** the kill only, then the 5 s after it are
scored (mean over the 5 s). Three predictors:

* **ours**: `src/pathfinder`.
* **still**: `src/pathfinder --v-max 0.0000001`.
* **hold**: our own aligned position at the kill, held fixed (stand still where we were).

*abs* is what a scorer sees (it includes the drift already accumulated when the IMU died). *moved-since-kill* subtracts that:
it is the error of the displacement since the kill, so for hold it is simply how far the vehicle really travelled
(mean 2.05 m in 5 s).

```bash
python tools/blind_study.py --data MH_01_easy V1_02_medium V2_03_difficult --kills 20 --seed 7 --tag blind --jobs 2
python tools/eval_tables.py
```

| recording | kills | ours abs (m) | still abs (m) | hold abs (m) | ours moved-since-kill err (m) | still/hold moved-since-kill err (m) | kills ours worse than still (abs) | worse than hold (rel) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01_easy | 20 | 4.41 | 4.49 | 4.41 | 1.00 | 0.99 | 2 | 14 |
| V1_02_medium | 20 | 1.68 | 1.62 | 1.68 | 1.56 | 1.57 | 11 | 10 |
| V2_03_difficult | 20 | 1.89 | 1.88 | 1.87 | 1.38 | 1.37 | 8 | 11 |
| **all** | 60 | 2.66 | 2.66 | 2.66 | 1.32 | 1.31 | 21 | 35 |

* **Does dead reckoning ever lose to standing still? Yes, often, and it never wins by much.** In absolute error ours is worse than
  `still` in 21 of 60 kills (worst loss +0.29 m, best gain -0.19 m); against `hold` on the moved-since-kill
  error it is worse in 35 of 60 (worst loss +0.15 m, best gain -0.07 m). On average the two are equal to 0.01 m.
* The numbers say the entry behaves almost like holding still after the IMU dies (I did not trace why; its `--blind-vel-tau` setting
  controls how fast the blind velocity decays). That is why the 5 s score (20 points) is
  set by the drift at the moment of failure, not by the extrapolation: the absolute error is 1.7-4.4 m, the part added in the last 5 s 1.0-1.6 m.
* The loss is not tied to speed: the correlation between how far the vehicle travelled in the 5 s and (ours minus hold) is 0.18.

## 4. Docker

```bash
docker build -t pathfinder src/
docker run --rm -v "$PWD/runs/track3/docker:/data" pathfinder /data -o /data/estimated_trajectory.txt
# measurement: the same image with GNU time installed in a throw-away container
docker run --rm --entrypoint sh -v "$PWD/runs/track3/docker:/data" pathfinder -c \
  'apt-get update -qq && apt-get install -y -qq time && /usr/bin/time -v /app/pathfinder /data -o /data/est.txt'
```

Input: MH_01_easy degraded with seed 1 (`imu0/data.csv`, 35,620 rows, 184.1 s of IMU time including the 6 s of dropped rows). Docker Desktop, aarch64 Linux VM, `debian:bookworm-slim`.

| | Docker (Linux, `/usr/bin/time -v`) | macOS native (`/usr/bin/time -l`) | Budget |
|---|---:|---:|---:|
| Peak RSS | 3.19-3.25 MB (5 runs) | 2.2 MB | 8 MB |
| Wall time | 0.77-0.79 s (5 runs) | 0.82 s | |
| Per second of IMU | **4.2 ms** | 4.5 ms | 20 ms |
| User / system | 0.76-0.78 s / 0.00 s | | single thread |
| Poses out / rows in | 35,620 / 35,620 | | one per row |

* The container's output matches the native output to 1e-6 m (the 6th decimal of the output).
* A file 10 x longer (356,200 rows, built by repeating the same recording) peaks at 3,184 kB and gives 356,200 poses: memory is constant.
* The 8 MB budget is met by 2.5 x; the 20 ms budget by 4.7 x.
* Output checked: 8 columns, timestamps in seconds equal to the input's ns x 1e-9 (max difference 0), quaternion norm within 1e-8 of 1, all finite.

## Limits of this evaluation

* Our degradation script, not the organisers'; no TUM-VI; the hidden ground truth is higher accuracy than EuRoC's
  `state_groundtruth_estimate0`.
* The three blind-study recordings were chosen to span easy, medium and difficult, not at random. Kill times are random, but the
  noise ramp restarts for each kill, so the degradation level at the kill is always the maximum.
* Stand still is the entry's own `--v-max 0.0000001` setting, which keeps the start position; `hold` is the more meaningful "stand
  still after the IMU dies".
* No setting of the entry was tuned and no algorithm code was changed.
