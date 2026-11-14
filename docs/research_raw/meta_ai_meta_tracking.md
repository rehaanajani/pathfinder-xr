# Meta AI answer: Meta's head-tracking stack (14 Nov 2026)

Prompt: Oculus Insight, TLIO, Project Aria, how Meta validates tracking, Meta papers on robustness / multi-IMU. Summary of Meta AI's answer; links are the ones it gave.

## 1. Oculus Insight (Quest)
- Visual-inertial SLAM for the headset, IR-LED constellation tracking for controllers; a new pose **every 1 ms**. ([Meta AI blog](https://ai.meta.com/blog/powered-by-ai-oculus-insight/))
- Fuses headset + controller IMUs, camera features (corners, floor patterns, furniture edges) triangulated into a 3D map, and controller IR LEDs. Also mapping, place recognition and floor detection.
- **Three asynchronous threads:** mapper (edits the map), tracker (poses from camera frames), and a higher-priority **IMU thread** writing the latest state to shared memory for low latency. Quest-specific DSP optimisations.
- Prediction: extrapolation with dynamic damping + a kinematic motion model, to cut jitter, latency and "swimminess".
- Compute budget: about 10x less compute and 100x less power than a PC.
- Hardware (product specs, not Meta tracking papers): Quest 1 four ultra-wide cameras; Quest 2 four; **Quest 3 four IR tracking + two RGB passthrough, XR2 Gen 2**; Quest Pro 10 sensors; Touch Pro controllers self-track with 3 cameras each.
- Darkness / occlusion / fast motion: IMU integration + prediction bridge short gaps, then re-localise against the map. **Meta has published no thresholds** (feature detector, bundle adjustment, darkness survival time are proprietary).

## 2. TLIO ([arXiv 2007.01867](http://arxiv.org/abs/2007.01867v1), [project](https://cathias.github.io/TLIO/), RA-L 2020)
- Motivated by dark rooms, occlusion, power and privacy.
- Single MEMS IMU; network + **stochastic-cloning EKF** (pose, velocity, gyro/accel biases + window of past poses).
- Measurement: 3D displacement + diagonal covariance in a gravity-aligned, yaw-invariant frame anchored at the older clone; **a chi-squared test rejects bad network updates**.
- Network: 1D ResNet18, 1 s window at 200 Hz, two heads (displacement, uncertainty). MSE for 10 epochs then Gaussian NLL; Adam, lr 1e-4, dropout 0.5; about 4 h on an NVIDIA DGX.
- The network gets no initial velocity, so it learns a "typical human motion" prior; the EKF handles velocity.
- Augmentation: random bias perturbations, up to 5 deg gravity tilt, random yaw.
- Data: Bosch BMI055 IMU on a headset rig, 400+ sequences, 60 h, 5+ people; supervised by a 1000 Hz VIO (not motion capture). Split 80/10/10.
- Result: 27 % less yaw drift, 33 % less position drift than RoNIN; learned (adaptive) covariance keeps tracking where fixed covariance loses it.
- Follow-ups: RNIN-VIO (2021), IDOL (AAAI 2021), EqNIO, AirIO (56.2 % better ATE than TLIO on drones).

## 3. Project Aria ([hardware spec](https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/hardware_spec), [docs](https://facebookresearch.github.io/projectaria_tools/))
- Gen 1: 2 mono SLAM cameras (150x120 deg, 640x480, global shutter, up to 30 fps), 1 RGB (up to 2880x2880), 2 eye cameras, **2 IMUs at 1000 Hz and 800 Hz**, magnetometer 10 Hz, barometer 50 Hz, 7 mics, GPS/Wi-Fi/BT; all on one nanosecond clock. Snapdragon 835, 75 g.
- Gen 2 (2025): 4 CV cameras, on-device SLAM / eye / hand tracking, 6-8 h battery.
- **MPS** (offline cloud processing): 1 kHz closed-loop and open-loop trajectories, semi-dense point cloud, multi-sequence alignment, eye gaze, hand tracking. Offline with loop closure, so not the same as on-device tracking.
- **AEA:** 143 sequences, 5 locations, about 7.3 h, MPS trajectories as pseudo ground truth. ([AEA paper](https://arxiv.org/abs/2402.13349))
- **ADT:** 200 sequences, 2 digitised scenes, 398 objects; ground truth from **OptiTrack** with sub-10-microsecond sync; object-pose system error about 6.8 mm / 1.29 deg. ([ADT paper](https://arxiv.org/abs/2306.06362), [dataset](https://www.projectaria.com/datasets/aria-digital-twin))
- Meta publishes no single "MPS trajectory error" number.

## 4. How Meta validates tracking
- OptiTrack rigs in offices and employees' homes, hundreds of environments; tune on discrepancies.
- Target: **sub-millimetre jitter**; Insight "exceeds this target in most environments". No public ATE or latency number for Quest.
- Replays thousands of hours of recordings to catch regressions, with a mobile-performance simulator on servers.

## 5. Related Meta papers
- TLIO; RNIN-VIO (ISMAR 2021, [arXiv 2109.08416](https://arxiv.org/abs/2109.08416)); GNSS carrier-phase + VIO fusion ([arXiv 2303.01291](https://arxiv.org/abs/2303.01291)); LaMaRia city-scale egocentric VI-SLAM benchmark, 2025 ([site](https://www.lamaria.ethz.ch)); Project Aria ([arXiv 2307.05973](https://arxiv.org/abs/2307.05973)).
- **No public Meta paper describes a multi-IMU fusion algorithm.** Insight only says it fuses "multiple IMUs"; Aria has two IMUs.

## What this means for us
- Our filter mirrors TLIO's design choices (1 s / 200 Hz window, displacement + covariance, chi-square gating, bias augmentation), so judges can map it to a known Meta method.
- Our additions (fault-aware augmentation, multi-IMU voting, camera health, model self-check) fill gaps Meta has not published.
- ADT (OptiTrack ground truth, 2 IMUs) is the best next dataset for real dual-IMU tests.
