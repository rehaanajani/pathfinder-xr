> **This folder is the camera + multi-IMU demo line, kept whole.** It is a snapshot of PRs #4-#11, Mayank's stress
> suite (#20) with its fixes (#24), and the final benchmark, taken on 16 Nov 2026. It has its **own copy** of
> `pathfinder/` (the filter as it was before the estimator redesign in #10); the repository root holds the redesigned
> estimator (#10, #26) and the scored IMU-only entry (`src/`). Run everything here from inside `demo_line/`:
> `python -m pytest tests` (86 passed on 16 Nov), `./demo.sh check`, `bash tools/live_demo/run.sh --model` (live phone
> demo, see DEMO.md). Not part of the scored submission.

# Path Finder: trajectory estimation that survives sensor failure

**Samsung PRISM XR Hackathon 2026, Track 3: Path Finder, Trajectory Estimation Under Sensor Degradation.**
Team California Burrito, SRM Institute of Science and Technology, Kattankulathur.

> Design robust trajectory-estimation techniques capable of maintaining reliable tracking and navigation
> when sensor measurements become degraded or unreliable.

A headset knows where it is by fusing its cameras with its IMUs (motion sensors). Cameras fail in the dark,
under motion blur, against blank walls, or when they lock onto the wrong thing. IMUs drift, saturate, spike
or die. When that happens the virtual world slides away from the real one.

This system **finds which sensor is lying, stops trusting it, and keeps tracking on what is left**:

1. **Camera health, measured before fusion and learned from real frames.** Each frame's brightness, sharpness
   and corners predict how many real feature tracks survive (fitted on real TUM-VI frames, tested on a room it
   never saw). How much each health level is trusted is **calibrated from measured fix errors**, not hand-set.
2. **Parity-space IMU fault detection.** All IMUs on a headset sit on one rigid frame, so their readings must
   agree. The aerospace parity test (Sturza 1988) projects out the true motion, flags a fault with a chi-square
   test and isolates the faulty IMU. With 3+ IMUs it catches small biases and slow drift that simple voting
   misses. Galaxy XR has five IMUs.
3. **A fault-aware learned motion model.** A small network (375,046 parameters, 1.5 MB) reads one second of
   IMU data and predicts how far the head moved **and how sure it is**. It is trained on broken IMU data too,
   so it reports more doubt when its input is bad.
4. **An adaptive error-state Kalman filter** blends everything by trust. A soft chi-square gate caps any reading
   that disagrees, a camera that suddenly jumps is held out for up to 30 s, and after an outage the filter
   re-anchors within 1 s. When healthy camera frames keep disagreeing with the IMU, the IMU's process noise is
   raised (adaptive Kalman filtering), so drifting IMUs are corrected faster.
5. **A self-check on the learned model.** While the camera is healthy, the filter compares the network's predictions
   with what the camera sees. If the network keeps being wrong on this kind of motion, its uncertainty is inflated
   so it is trusted less once the camera drops out.
6. **A health signal out.** Per-sensor trust and the filter's own position uncertainty in metres. OpenXR today
   only reports a yes/no "position tracked" flag; this gives a graded one.

## Where this sits in a headset

The OpenXR spec asks runtimes, during tracking loss, to keep providing an inferred pose "based on ... inertial dead
reckoning", clear the TRACKED flag, and snap back when tracking recovers. This project is that component, made
fault-aware:

```
IMUs + cameras -> Sensors HAL -> [ IMU voting ] -> perception / VIO service -> OpenXR runtime -> apps (xrLocateSpace)
                                 [ camera health ] ->        |
                                 [ learned inertial model + robust filter ] -> pose when VIO is lost or degraded
                                 [ health signal ] -> graded tracking quality (beyond the TRACKED bit)
```

It is a library for the device's tracking service, not an app: users never install it, and apps only see better
poses. Shipping it means a C++ port of the filter and voting, and the network exported to TFLite / the phone's NPU.

## Submission package

| Item | Where |
|---|---|
| Source code | this repository |
| Presentation | [`deck/PathFinder_SRMIST_CaliforniaBurrito.pptx`](deck/PathFinder_SRMIST_CaliforniaBurrito.pptx) (Samsung template, built by `deck/build_deck.py`) |
| Video | [`results/demo.mp4`](results/demo.mp4) (84 s, rendered from real runs by `make_video.py`) |
| Interactive demo | [`demo/index.html`](demo/index.html), open in any browser |
| Results | [`results/report.html`](results/report.html), [`results/summary.json`](results/summary.json) |
| Trained models | [release `models-v1`](https://github.com/rehaanajani/pathfinder-xr/releases/tag/models-v1) |
| AI disclosure | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) |

## Results
### On real sensor data (TUM-VI)

Real Bosch BMI160 IMU, real fisheye camera frames, OptiTrack truth. Degradations are applied to the real frames; camera health and trust are fitted on one room and tested on the other. Visual fixes come from motion capture plus error driven by how many real KLT feature tracks survive. The motion network was trained on Meta's headset data, so this is also a cross-device test. 60 s clips, ATE in metres.

| Scenario | Naive fusion | **Ours** | Ours w/o model | Gain |
|---|---:|---:|---:|---:|
| No fault | 0.043 | **0.049** | 0.049 | 0.9x |
| Camera off 10 s | 0.881 | **0.498** | 0.798 | 1.8x |
| Dark room 10 s | 0.276 | **0.245** | 0.236 | 1.1x |
| Motion blur 10 s | 0.153 | **0.136** | 0.136 | 1.1x |
| Dark room + all IMUs drift | 0.277 | **0.273** | 0.271 | 1.0x |
| Camera off + one IMU biased | 4.004 | **0.500** | 0.804 | 8.0x |
| Camera off + all IMUs saturate | 10.919 | **3.055** | 11.004 | 3.6x |

**Camera health on real frames:** the learned score predicts the share of correct real KLT tracks with correlation **0.86** on a room it never saw (249 frame pairs), up from 0.64 for the hand-tuned score.

### IMU only, full TLIO test split, next to published results

All 36 test sequences start to end (mean 318 s), one real IMU, no camera. Published numbers use their own filter and alignment, so this is indicative, not like-for-like.

| Method | ATE (m) |
|---|---:|
| TLIO (paper) (published) | 1.722 |
| TLIO + rot. aug. (EqNIO table) (published) | 1.812 |
| EqNIO O(2) (published) | 1.433 |
| Our network + our filter, TLIO-style training | 6.980 |
| Our network + our filter, fault-aware training (first filter) | 6.266 |
| **Same network, cloned poses + uncertainty x1600 (chosen on validation)** | **2.000** (median 1.44) |

**On-device readiness:** exported to ONNX (1.5 MB, 0.40 ms) and INT8 (0.40 MB, 0.28 ms) per prediction on one CPU core; INT8 window error 0.174 m vs 0.166 m. Both files are in the `models-v1` release.


All numbers come from `python evaluate.py` on 36 held-out test sequences of Meta's TLIO headset dataset
(never seen in training), 60 s each, faults starting at 20 s. ATE = absolute trajectory error after
yaw + translation alignment, the metric used by RoNIN and TLIO. Full table and plots: `results/report.html`.

**Headline.** Camera off + one IMU biased: naive fusion drifts **2.16 m**, ours **0.20 m** (11.1x lower). After a 20 s blackout: **9.08 m vs 1.20 m**. Ours is clearly better in 9 of 16 scenarios, level in 3, worse in 4 (see Honest limits).

| Scenario | IMU only | Naive fusion | Ours w/o model | Ours w/ TLIO-trained model | **Ours** | Ours median | Gain vs naive |
|---|---:|---:|---:|---:|---:|---:|---:|
| No fault | 29.8 | 0.03 | 0.03 | 0.04 | **0.04** | 0.04 | 0.8x |
| Camera off 10 s | 29.8 | 0.43 | 0.29 | 0.19 | **0.20** | 0.13 | 2.2x |
| Dark room 10 s | 29.8 | 0.22 | 0.21 | 0.13 | **0.12** | 0.11 | 1.8x |
| Motion blur 10 s | 29.8 | 0.20 | 0.18 | 0.12 | **0.11** | 0.11 | 1.7x |
| Camera tracks the wrong thing 10 s | 29.8 | 0.31 | 0.31 | 0.29 | **0.30** | 0.30 | 1.0x |
| Noise on all IMUs | 33.7 | 0.03 | 0.03 | 0.04 | **0.04** | 0.04 | 0.8x |
| One IMU biased | 122.6 | 0.03 | 0.03 | 0.04 | **0.04** | 0.04 | 0.8x |
| One IMU dies | 448.8 | 0.15 | 0.03 | 0.04 | **0.04** | 0.04 | 3.7x |
| One IMU spikes | 87.5 | 0.04 | 0.03 | 0.04 | **0.04** | 0.04 | 0.9x |
| One IMU saturates | 116.1 | 0.04 | 0.03 | 0.04 | **0.04** | 0.04 | 1.0x |
| Camera off + one IMU biased | 174.7 | 2.16 | 0.30 | 0.19 | **0.20** | 0.12 | 11.1x |
| Dark room + all IMUs drift | 697.6 | 0.22 | 0.27 | 0.24 | **0.25** | 0.22 | 0.9x |
| Camera off + vibration on all IMUs | 38.5 | 0.73 | 0.74 | 0.48 | **0.21** | 0.15 | 3.4x |
| Camera off + spikes on all IMUs | 112.6 | 2.49 | 1.73 | 0.40 | **0.22** | 0.15 | 11.5x |
| Camera off + all IMUs saturate | 418.3 | 8.90 | 8.91 | 1.50 | **1.26** | 0.56 | 7.0x |
| Camera off + IMU packets dropped | 157.1 | 2.33 | 2.32 | 0.34 | **0.34** | 0.25 | 6.8x |

ATE in metres, mean over sequences (median for ours alongside). Per-sequence values are in `results/summary.json`.

**Error when the camera comes back, by blackout length (m, mean):**

| Blackout | 1 s | 2 s | 5 s | 10 s | 20 s |
|---|---:|---:|---:|---:|---:|
| IMU only | 7.35 | 8.24 | 11.29 | 17.78 | 37.03 |
| Naive fusion | 0.10 | 0.20 | 0.73 | 2.35 | 9.08 |
| **Ours** | 0.13 | 0.16 | 0.35 | 0.76 | 1.20 |

**Finding the faulty IMU** (share of faulty moments where the bad IMU is dropped and every good one kept):

| Fault | 2 IMUs | 3 IMUs | 5 IMUs |
|---|---:|---:|---:|
| bias | 0% | 100% | 100% |
| small bias | 0% | 100% | 100% |
| slow drift | 0% | 100% | 100% |
| stuck | 99% | 100% | 100% |
| spikes | 99% | 100% | 100% |
| noise | 99% | 100% | 100% |

### Methods compared

| Method | What it is |
|---|---|
| IMU only | Integrate the averaged IMUs, no correction. Shows the raw problem. |
| Naive fusion | Kalman filter, camera + averaged IMUs, every frame trusted equally. Typical simple VIO fusion. |
| Learned IMU only (TLIO-style) | Our network trained with TLIO's augmentation, filter, no camera. |
| Ours without learned model | Voting + camera health + gate, no network (ablation). |
| **Ours** | Everything above + the fault-aware network. |

## Run it

```bash
bash reproduce.sh               # set up, download data (3.8 GB), train or reuse models, test, benchmark, demo
bash reproduce.sh --synthetic   # no download, synthetic head walks, ~3 min
```

Outputs: `results/report.html` (all numbers and plots), `results/summary.json`, `demo/index.html`
(interactive, open in any browser).

## Repository map

| Path | What |
|---|---|
| `pathfinder/faults.py` | Seeded fault injector: camera blackout, dark, blur, wrong fix; IMU noise, bias, drift, dropout, saturation, stuck, spikes |
| `pathfinder/camera.py` | Image health score and the visual fix model |
| `pathfinder/imu_vote.py` | Multi-IMU consistency voting and fault isolation |
| `pathfinder/filter.py` | Error-state Kalman filter with health weighting, chi-square gate and lock-out guard |
| `pathfinder/model.py`, `train.py` | Displacement network and fault-aware training |
| `pathfinder/pipeline.py` | Runs every method on a sequence and scenario |
| `evaluate.py`, `make_demo.py`, `make_video.py` | Benchmark and report, interactive demo, video |
| `make_readme.py`, `deck/build_deck.py` | README results and the slide deck, generated from `results/summary.json` |
| `tests/` | Unit tests (`pytest`) |

## Honest limits

- **Not every scenario is a win.** In *dark room + all IMUs drifting* our mean error is higher than naive fusion
  (median is lower); one held-out recording with fast, unusual motion is outside what the network learned and
  dominates the mean. When the camera already rescues naive fusion (one biased IMU with a working camera, plain
  noise) the two are level.
- **Fault-aware training helps less than the architecture.** Swapping in the TLIO-style-trained network gives
  similar numbers in most scenarios; the fault-aware network is clearly better only under vibration and saturation
  on every IMU. Most of the gain comes from voting, camera health, the robust gate and the model self-check.
- **60 s clips.** Each of the 36 test sequences is scored on one 60 s clip; longer runs are a next step.

- **Clean tracking costs about 1 cm.** With a healthy camera and healthy IMUs our filter is slightly behind naive
  fusion (real TUM-VI: 4.9 vs 4.3 cm), the price of the extra caution that wins every failure case.
- **Real camera, but fixes still come from motion capture.** On TUM-VI we use the real IMU and the real fisheye
  frames: degradations are applied to the real images, the health score is computed from them, and fix noise follows
  the measured fraction of correctly tracked features. The position fix itself is motion-capture truth plus that
  noise, not a full visual odometry front end.
- **Extra IMUs are simulated.** TLIO and TUM-VI have one IMU each. Multi-IMU tests copy the real IMU with its own
  noise and bias per copy, all at the same point. The parity detector compensates lever arms when the IMU
  positions are given (`lever_arms`, tested on a simulated rigid head: false alarms 99 % without it, 1-2.6 %
  with it), assumes the IMUs are already rotated into one body frame and time-aligned, and has not been
  validated on real multi-IMU hardware. A fault shared by every IMU (common vibration, same-model temperature
  drift) cancels in the parity test and is not detected there.
- **Two IMUs detect, three isolate.** With two IMUs a smooth bias is detected but cannot always be attributed; the
  parity test isolates it with three or more, which is why Galaxy XR's five matter.
- **IMU-only accuracy is close to, not yet at, published TLIO.** Full TLIO test split, IMU only, no
  alignment: published TLIO 1.72 m, EqNIO 1.43 m. Ours went from 6.27 m to **2.00 m (median 1.44 m)** by
  keeping past poses in the filter (stochastic cloning) and calibrating the network's claimed uncertainty
  on the validation split (it overlaps 10 one-second windows per second and its errors are correlated).
  The headset benchmark above still uses the earlier filter setting; switching it over is being tested.
  TLIO's yaw-frame measurement was also implemented and did not help IMU-only tracking (a common heading
  error cancels in it); it stays as an option. See `results/diagnostics/`.
- **A wider network is training.** 1.46M parameters, 0.120 m per-window error after 17 of 30 epochs vs 0.136 m
  for the current network; not yet scored end to end.
- **Not on a headset yet.** The network runs in ONNX Runtime at 2.0 ms on one CPU core (INT8: 0.4 MB); the
  next step is ONNX Runtime Mobile or Qualcomm QNN on Snapdragon XR2+ Gen 2.

## Prior work we build on

- W. Liu et al., *TLIO: Tight Learned Inertial Odometry*, RA-L 2020. Dataset (CC BY-NC 4.0) and network design (BSD).
- H. Yan et al., *RoNIN*, ICRA 2020. Evaluation metrics.
- R. K. Jayanth et al., *EqNIO*, ICLR 2025, and *Neural Inertial Odometry from Lie Events*, RSS 2025.
- Y. Qiu et al., *AirIMU* (2023) and *AirIO* (2025). Learned IMU uncertainty.
- M. Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*, ICLR 2022 (beta-NLL).
- X. Peng et al., *DVI-SLAM: A Dual Visual Inertial SLAM Network*, Samsung R&D Institute China, ICRA 2024. Learns
  visual vs IMU confidence inside dense bundle adjustment. Our work differs in isolating faults *between IMUs*,
  scoring the camera *before* fusion, and staying small enough for a phone-class processor, which the DVI-SLAM
  authors name as their next step.

## AI disclosure

See [AI_DISCLOSURE.md](AI_DISCLOSURE.md).

## Team

Rehaan Ajani (lead), Nikhil Solanki, Debadrita Haldar, Mayank Yadav. B.Tech CSE (AI/ML), SRM IST Kattankulathur.
