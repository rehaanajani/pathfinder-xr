# Path Finder: trajectory estimation that survives sensor failure

**Samsung PRISM XR Hackathon 2026, Track 3: Path Finder, Trajectory Estimation Under Sensor Degradation.**
Team California Burrito, SRM Institute of Science and Technology, Kattankulathur.

> Design robust trajectory-estimation techniques capable of maintaining reliable tracking and navigation
> when sensor measurements become degraded or unreliable.

A headset knows where it is by fusing its cameras with its IMUs (motion sensors). Cameras fail in the dark,
under motion blur, against blank walls, or when they lock onto the wrong thing. IMUs drift, saturate, spike
or die. When that happens the virtual world slides away from the real one.

This system **finds which sensor is lying, stops trusting it, and keeps tracking on what is left**:

1. **Camera health, measured before fusion.** Every frame is scored for light, sharpness and trackable
   corners. A dark or blurred frame is down-weighted before it can pull the path off.
2. **IMU voting.** All IMUs on a headset sit on one rigid frame, so they must agree. A faulty IMU (stuck,
   biased, spiking, saturated, noisy) is outvoted and dropped. Galaxy XR has five IMUs; we show isolation
   improves from 2 to 3 to 5 IMUs.
3. **A fault-aware learned motion model.** A small network (375,046 parameters, 1.5 MB) reads one second of
   IMU data and predicts how far the head moved **and how sure it is**. It is trained on broken IMU data too,
   so it reports more doubt when its input is bad.
4. **An error-state Kalman filter** blends everything by trust. A soft chi-square gate caps any reading that
   disagrees, a camera that suddenly jumps is held out for up to 10 s, and after an outage the filter re-anchors
   within 1 s instead of locking itself out.
5. **A self-check on the learned model.** While the camera is healthy, the filter compares the network's predictions
   with what the camera sees. If the network keeps being wrong on this kind of motion, its uncertainty is inflated
   so it is trusted less once the camera drops out.
6. **A health signal out.** Per-sensor trust and the filter's own position uncertainty, the tracking-quality
   signal that apps can use and that today's Android XR pose API does not expose.

## Results

All numbers come from `python evaluate.py` on 10 held-out test sequences of Meta's TLIO headset dataset
(never seen in training), 60 s each, faults starting at 20 s. ATE = absolute trajectory error after
yaw + translation alignment, the metric used by RoNIN and TLIO. Full table and plots: `results/report.html`.

**Headline.** Camera off + one IMU biased: naive fusion drifts **2.10 m**, ours **0.27 m** (7.8x lower). After a 20 s blackout: **7.24 m vs 1.76 m**. Ours is clearly better in 10 of 16 scenarios, level in 5, worse in 1 (see Honest limits).

| Scenario | IMU only | Naive fusion | Ours w/o model | Ours w/ TLIO-trained model | **Ours** | Ours median | Gain vs naive |
|---|---:|---:|---:|---:|---:|---:|---:|
| No fault | 29.4 | 0.03 | 0.03 | 0.03 | **0.03** | 0.03 | 1.0x |
| Camera off 10 s | 29.4 | 0.37 | 0.70 | 0.23 | **0.24** | 0.12 | 1.6x |
| Dark room 10 s | 29.4 | 0.24 | 0.70 | 0.23 | **0.24** | 0.12 | 1.0x |
| Motion blur 10 s | 29.4 | 0.21 | 0.21 | 0.14 | **0.14** | 0.13 | 1.5x |
| Camera tracks the wrong thing 10 s | 29.4 | 0.31 | 0.32 | 0.28 | **0.29** | 0.32 | 1.1x |
| Noise on all IMUs | 30.7 | 0.03 | 0.04 | 0.04 | **0.04** | 0.04 | 0.8x |
| One IMU biased | 123.6 | 0.03 | 0.03 | 0.03 | **0.03** | 0.03 | 1.0x |
| One IMU dies | 460.7 | 0.15 | 0.03 | 0.03 | **0.03** | 0.03 | 5.0x |
| One IMU spikes | 84.9 | 0.04 | 0.03 | 0.03 | **0.03** | 0.03 | 1.2x |
| One IMU saturates | 83.4 | 0.04 | 0.03 | 0.03 | **0.03** | 0.03 | 1.3x |
| Camera off + one IMU biased | 175.4 | 2.10 | 2.42 | 0.26 | **0.27** | 0.15 | 7.8x |
| Dark room + all IMUs drift | 703.8 | 0.24 | 3.86 | 0.34 | **0.36** | 0.21 | 0.7x |
| Camera off + vibration on all IMUs | 34.0 | 0.59 | 0.97 | 0.54 | **0.35** | 0.24 | 1.7x |
| Camera off + spikes on all IMUs | 77.0 | 2.18 | 0.69 | 0.29 | **0.30** | 0.13 | 7.3x |
| Camera off + all IMUs saturate | 255.6 | 4.74 | 4.52 | 0.67 | **0.58** | 0.45 | 8.2x |
| Camera off + IMU packets dropped | 119.4 | 1.63 | 1.72 | 0.35 | **0.35** | 0.13 | 4.7x |

ATE in metres, mean over sequences (median for ours alongside). Per-sequence values are in `results/summary.json`.

**Error when the camera comes back, by blackout length (m, mean):**

| Blackout | 1 s | 2 s | 5 s | 10 s | 20 s |
|---|---:|---:|---:|---:|---:|
| IMU only | 5.26 | 5.98 | 8.58 | 14.49 | 33.25 |
| Naive fusion | 0.08 | 0.18 | 0.64 | 2.04 | 7.24 |
| **Ours** | 0.08 | 0.14 | 0.33 | 0.99 | 1.76 |

**Finding the faulty IMU** (share of faulty moments where the bad IMU is dropped and every good one kept):

| Fault | 2 IMUs | 3 IMUs | 5 IMUs |
|---|---:|---:|---:|
| bias | 0% | 99% | 99% |
| stuck | 99% | 99% | 99% |
| spikes | 99% | 99% | 99% |
| noise | 99% | 99% | 99% |

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
| `evaluate.py`, `make_demo.py` | Benchmark, report and demo |
| `tests/` | Unit tests (`pytest`) |

## Honest limits

- **Not every scenario is a win.** In *dark room + all IMUs drifting* our mean error is higher than naive fusion
  (median is lower); one held-out recording with fast, unusual motion is outside what the network learned and
  dominates the mean. When the camera already rescues naive fusion (one biased IMU with a working camera, plain
  noise) the two are level.
- **Fault-aware training helps less than the architecture.** Swapping in the TLIO-style-trained network gives
  similar numbers in most scenarios; the fault-aware network is clearly better only under vibration and saturation
  on every IMU. Most of the gain comes from voting, camera health, the robust gate and the model self-check.
- **10 test sequences x 60 s.** Run `python evaluate.py --seqs 36` for the full test split.

- **The camera is simulated on TLIO.** The TLIO dataset has IMU and ground truth but no images. Visual fixes are
  ground truth plus noise, drift and jumps that grow with darkness and blur; the health score is measured from
  degraded synthetic frames with the same function that runs on real images. The same model is applied to every
  method. The next step is real images (Aria, EuRoC, TUM-VI).
- **Extra IMUs are simulated.** TLIO has one IMU per sequence. For the multi-IMU tests we copy the real IMU and
  give each copy its own noise and bias. Real IMUs at different positions also differ by lever-arm effects;
  our tolerance covers typical head motion but is not tuned on real multi-IMU hardware.
- **With two IMUs, a smooth bias on one is detected but not always attributed** to the right IMU (see the
  isolation table). Three or more IMUs fix this, which is why Galaxy XR's five matter.
- **The learned displacement update is simplified.** TLIO keeps past states in the filter (stochastic cloning);
  we hold the past position at its estimate and add its covariance.
- **Python, not on-device yet.** The network is 1.5 MB and takes 0.5 ms per prediction on one laptop CPU core
  at 10 predictions per second; porting to Android (TFLite / ExecuTorch) is the next step.

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
