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
3. **A fault-aware learned motion model.** A small network (__PARAMS__ parameters, __SIZE__ MB) reads one second of
   IMU data and predicts how far the head moved **and how sure it is**. It is trained on broken IMU data too,
   so it reports more doubt when its input is bad.
4. **An adaptive error-state Kalman filter** blends everything by trust. A soft chi-square gate caps any reading
   that disagrees, a camera that suddenly jumps is held out for up to 10 s, and after an outage the filter
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
__EXTRA__

All numbers come from `python evaluate.py` on __NSEQ__ held-out test sequences of Meta's TLIO headset dataset
(never seen in training), __CLIP__ s each, faults starting at 20 s. ATE = absolute trajectory error after
yaw + translation alignment, the metric used by RoNIN and TLIO. Full table and plots: `results/report.html`.

__RESULTS__

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
  noise and bias per copy. The parity-space detector models lever arms but is not tuned on real multi-IMU hardware.
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
- **Not on a headset yet.** The network runs in ONNX Runtime at __MS__ ms on one CPU core (INT8: 0.4 MB); the
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
