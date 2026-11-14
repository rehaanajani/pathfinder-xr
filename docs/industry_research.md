# Industry research: the maths, the competitors, the benchmarks

Collected 14 Nov 2026 from the pages linked below. Anything marked *unverified* comes from a secondary source.

## 1. The maths in the sources

### Samsung DVI-SLAM ([paper](https://arxiv.org/abs/2309.13814), [blog](https://research.samsung.com/blog/DVI-SLAM-A-Dual-Visual-Inertial-SLAM-Network))
Built on DROID-SLAM. Three parts: a CNN **feature extraction** module, a **multi-factor data association** module (a 4D **correlation volume** between frames fed into a **ConvGRU** that iteratively updates optical flow and confidence maps), and a differentiable **dense bundle adjustment (DBA)** layer solved with Gauss-Newton.

The DBA layer minimises three weighted residuals. Each weight `w` is a **learned confidence map**:

- Re-projection factor (does pixel `x_i`, projected with depth `d_i` and poses `T_i, T_j`, land where the flow says?):
  `E^r(T,d) = || x*_ij − Π(T_i, T_j, d_i, x_i) ||²_Σr`,  `Σr = diag(w^r_ij)`
- Feature-metric factor (do the learned features match at the projected pixel?):
  `E^f(T,d) = || F_i(x_i) − F_j(Π(T_i, T_j, d_i, x_i)) ||²_Σf`
- IMU factor (IMU **pre-integration** between keyframes i and j, with bias Jacobians):
  rotation `Log( (ΔR_ij · Exp(ΔJ^r_ij (bg_i − b̂g_i)))ᵀ R_iᵀ R_j )`,
  position `R_iᵀ(p_j − p_i − v_i Δt − ½ g Δt²) − (Δp_ij + ΔJ^p_ij (b_i − b̂_i))`,
  velocity `R_iᵀ(v_j − v_i − g Δt) − (Δv_ij + ΔJ^v_ij (b_i − b̂_i))`,
  bias random walk `ba_j − ba_i`, `bg_j − bg_i`
- Total: `E(T,M,d) = Σ E^r + Σ E^f + Σ_(j=i+1) E^u`, over the frame graph edges.
- Training loss: `L = w1·L_pose + w2·L_flow + w3·L_repro + w4·L_metric`, trained on 4× RTX 3090.
- Initialisation follows VINS-Mono (gravity direction, IMU motion, scale for monocular).
- Their own failure case: TartanAir MH004, "mostly a white wall". Their own next step: "dense to sparse tracking" for real-time on "consumer-grade mobile processor platforms".

### Meta TLIO ([project page](https://cathias.github.io/TLIO/))
Facebook Reality Labs Redmond + UPenn. IMU-only. A network regresses **displacement + uncertainty** from a window of gravity-aligned IMU. A tightly coupled **EKF with stochastic cloning** fuses those with strapdown propagation, estimating pose, velocity and **sensor biases**. Trained on 60 h of pedestrian data. Loss: MSE, then Gaussian negative log-likelihood.

### OpenXR 1.1 ([spec](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html))
On tracking loss, runtimes "should continue to provide inferred or last-known position", for example by "inertial dead reckoning". They must keep `POSITION_VALID_BIT` set, may clear `POSITION_TRACKED_BIT`, and should "snap the pose ... back" on recovery. This is the exact slot our module fills. The only quality signal is that yes/no bit.

### Android Sensors HAL 2.0 ([docs](https://source.android.com/docs/core/interaction/sensors/sensors-hal2))
IMU events go from the HAL to the framework through **Fast Message Queues** (an Event FMQ plus a Wake Lock FMQ). Each sensor is configured with `batch()` (sampling period + max reporting latency) and "must be able to be reconfigured at any time ... without the loss of sensor data". Our IMU voting would sit right after this layer.

### Qualcomm Snapdragon XR2+ Gen 2 ([product page](https://www.qualcomm.com/products/mobile/snapdragon/xr-vr-ar/snapdragon-xr2-plus-gen-2-platform))
The chip in Galaxy XR: "12 or more concurrent cameras" and on-device AI "to track your head, hands, controllers, eyes". Qualcomm's older VIO reference used two 1 MP global-shutter fisheye cameras plus an IMU, and its advice on lost tracking was to **fade to black** ([2017 deck](https://www.qualcomm.com/media/documents/files/on-device-motion-tracking-for-immersive-vr.pdf)). Our pitch beats that: about 1 m of error after 20 s of blackout instead of a black screen.

## 2. The maths in our system
- **Error-state Kalman filter**, 15 states `[δθ, δv, δp, δb_g, δb_a]`, `R_true = R·Exp(δθ)`.
  Propagation: `p += v·dt + ½a·dt²`, `v += a·dt`, `a = R(f − b_a) + g`, `R ← R·Exp((ω − b_g)dt)`;
  linearised `F` with `−[ω]×`, `−R[f]×`, process noise scaled up when IMU confidence drops.
- **Updates** on position (camera fix, learned displacement) with Joseph-form covariance.
- **Soft chi-square gate**: NIS `= yᵀS⁻¹y`; above 16.27 (χ², 3 dof, 99.9 %) the noise is inflated by `(NIS/16.27)²`.
- **Camera health** = brightness × sharpness (variance of the Laplacian after a Gaussian blur, exposure-normalised) × corner count (Shi-Tomasi minimum eigenvalue of the structure tensor, with non-max suppression).
- **IMU voting**: per 0.1 s window, element-wise median of the healthy IMUs; disagreement measured against gyro (0.05 rad/s) and accel (0.6 m/s²) tolerances; stuck / saturated / high-frequency-energy checks.
- **Network**: 1D ResNet, loss `MSE + β-NLL` with `β = 0.5` ([Seitzer et al., ICLR 2022](https://arxiv.org/abs/2203.09168)).
- **Metrics**: ATE after 4-DoF (yaw + translation) Umeyama alignment, RTE over 1 s / 10 s, drift %.

## 3. Who builds this, and with what

| Company | Tracking stack (public) | Data / validation | Results we can cite |
|---|---|---|---|
| **Meta** | "Insight": visual-inertial SLAM, 4 wide-FoV fisheye cameras + IMUs, runs on the low-power CPU cores ([Reality Labs](https://tech.facebook.com/reality-labs/2019/08/the-story-behind-oculus-insight-technology/), [Game Developer](https://www.gamedeveloper.com/extended-reality/c-mon-and-slam-how-oculus-tackled-portable-6dof-tracking-for-the-quest)). Research: TLIO, Project Aria + MPS | Tuned against OptiTrack motion capture; TLIO 60 h; Aria datasets | Quest 3 is sub-cm in most cases but >10 cm APE in hard scenarios (Duke, below) |
| **Apple** | ARKit "world tracking" via visual-inertial odometry ([Apple docs](https://developer.apple.com/documentation/arkit/understanding-world-tracking)) | Not public | Vision Pro beat Quest 3 by 33.9 % RPE and 14.6 % APE ([Duke, ImmerCom 2024](https://par.nsf.gov/biblio/10634403-apple-meta-comparative-study-spatial-tracking-sota-xr-headsets)) |
| **Google + Qualcomm** (Android XR, Galaxy XR) | Android XR is built on OpenXR 1.1; tracking runs on Qualcomm's perception stack on XR2+ Gen 2 (*unverified detail*, [secondary](https://www.framesixty.com/android-xr)); Snapdragon Spaces runtime | Not public | No public accuracy numbers found |
| **Samsung Research** | DVI-SLAM (learned dense VI-SLAM, confidence-weighted factors) | TartanAir, EuRoC, ETH3D-SLAM | 1st of 29 teams, ICCV 2023 VI-SLAM challenge (darkness, smoke, lighting changes); mean ATE 0.547 m ([news](https://research.samsung.com/news/Samsung-R-D-Institute-China-ndash-Beijing-Secures-First-Place-in-ICCV-2023-SLAM-Challenge)) |
| **Spectacular AI** (startup, Finland) | Commercial VISLAM SDK for headsets, hardware-agnostic, < 50 MB RAM ([docs](https://spectacularai.github.io/docs/sdk/)) | Not public | Not public |
| **Lynx** (headset startup) | Open-sourced an Android ORB-SLAM3 fork: 640×400 input, 90 fps on XR2 Gen 1, visual-only, GPLv3; their default is Qualcomm's 6DoF ([Lynx blog](https://portal.lynx-r.com/blog/view/open-sourcing-lynx-6dof/)) | - | - |
| **Open source** | ORB-SLAM3, OpenVINS, Basalt, VINS-Mono; Monado (open OpenXR runtime) | EuRoC, TUM-VI | ORB-SLAM3 about 1 cm ATE monocular-inertial on EuRoC rooms ([paper](https://arxiv.org/pdf/2007.11898)); Basalt strong on EuRoC with low memory |

**Who's best?** By the only head-to-head study we found, Apple's tracking is the most accurate shipping headset, and Meta's is close. Nobody publishes how their tracking behaves when a sensor *fails*. That's the gap we're in.

## 4. Benchmarks and indices
There is **no single public index for headset tracking**. The closest equivalents:
- **EuRoC MAV** and **TUM-VI**: the standard VIO tables (ATE in cm). TUM-VI uses a handheld rig with head-like motion.
- **TartanAir**: synthetic, includes hard conditions (Samsung used it).
- **ETH3D-SLAM**: online leaderboard.
- **Hilti SLAM Challenge**: yearly leaderboard, construction sites ([2023 report](https://arxiv.org/abs/2404.09765)).
- **ICCV 2023 SLAM Challenge**: TartanAir + SubT-MRS, degraded conditions (Samsung won VI track).
- **Duke XR tracking testbed**: the only head-to-head of shipping headsets.
- **Inertial-only**: TLIO dataset (ours), RoNIN, Aria Everyday Activities / Aria Digital Twin.

## 5. What this means for us
1. Our method is the "inertial dead reckoning during tracking loss" that OpenXR asks for, made fault-aware. That's a standards-backed slot, not an invented one.
2. Qualcomm's older guidance was to fade to black on lost tracking. We keep a usable pose for 20 s.
3. Nobody public reports per-sensor fault isolation. Galaxy XR has 5 IMUs, so this is a Samsung-specific advantage.
4. To be taken seriously next to these players we need **real camera frames** (EuRoC / TUM-VI), and ideally a number on a phone.
