# Perplexity answers: tech stack reference (14 Nov 2026)

Two answers were visible in the Perplexity thread (prompts 3 and 4 of the split set; 44 and 31 sources). Prompts 1 (filter maths) and 2 (IMU side) were not in that thread. Links are the ones Perplexity gave.

## Heteroscedastic regression (Gaussian NLL, beta-NLL)
- The network outputs a mean and an input-dependent variance; train by maximising Gaussian likelihood:
  `L_NLL = ½ log σ²(x) + (y − μ(x))² / (2σ²(x))`
- Plain NLL can inflate σ² early and hurt the mean. **beta-NLL** weights the loss by a detached `σ^(2β)`: `β = 0` is ordinary NLL, `β = 1` makes the mean gradient match plain squared error.
- Paper: Seitzer, Tavakoli, Antic, Martius, ICLR 2022, [arXiv 2203.09168](https://arxiv.org/abs/2203.09168). Code: [martius-lab/beta-nll](https://github.com/martius-lab/beta-nll).

## 1D ResNet / temporal CNNs for IMU
- Treats a 200x6 IMU window as a 6-channel time series; residual blocks `x_{l+1} = x_l + F(x_l)` keep deep nets trainable.
- Papers: ResNet ([arXiv 1512.03385](https://arxiv.org/abs/1512.03385)); RoNIN's 1D ResNet / TCN / LSTM ([arXiv 1905.12853](https://arxiv.org/abs/1905.12853)); TCN study by Bai, Kolter, Koltun ([arXiv 1803.01271](https://arxiv.org/abs/1803.01271)).

## Image quality and corners
- **Variance of Laplacian** `Var(∇²I)`: cheap blur score; sensitive to texture and exposure, so compare relatively.
- Structure tensor `M = [[Ix², IxIy], [IxIy, Iy²]]`; **Harris** `R = det(M) − k·tr(M)²`; **Shi-Tomasi** `R = λ_min(M)`.
- Papers: Harris & Stephens 1988; Shi & Tomasi, "Good Features to Track", CVPR 1994. Tutorial: [OpenCV Shi-Tomasi](https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html).

## Trajectory metrics
- `ATE_RMSE = sqrt(1/N Σ ||p̂_i − (sR p_i + t)||²)`; RPE measures local drift over a fixed interval.
- Umeyama alignment: least-squares `s, R, t`. Papers: Sturm et al., TUM RGB-D benchmark, IROS 2012; Umeyama, IEEE TPAMI 1991. Tool: [evo](https://github.com/MichaelGrupp/evo).

## Open-source VIO / SLAM
| System | What it is | Paper | EuRoC note |
|---|---|---|---|
| ORB-SLAM3 | Feature-based visual / visual-inertial / multi-map SLAM | [arXiv 2007.11898](https://arxiv.org/abs/2007.11898) | Strong in all modes |
| OpenVINS | MSCKF-style filter VIO platform | [arXiv 1908.03549](https://arxiv.org/abs/1908.03549) | Community eval: 0.079 m mono avg on Vicon rooms (excl. V2_03) |
| Basalt | Optimisation VIO with non-linear factor recovery | [arXiv 1904.04080](https://arxiv.org/abs/1904.04080) | Repo reproduces all 11 sequences |
| VINS-Mono | Sliding-window monocular VIO, pre-integration, loop closure | Qin, Li, Shen, T-RO 2018 | Beats compared methods on most sequences |
| DROID-SLAM | Learned dense SLAM, recurrent flow + differentiable BA | [arXiv 2108.10869](https://arxiv.org/abs/2108.10869) | 0.022 m mono / 0.024 m stereo avg ATE (visual only) |
- Caveat: EuRoC numbers only compare when sensors, mode, loop closure, alignment and failure policy match.

## Shipping a model on Android
- Export (ONNX or TFLite) → quantise (usually INT8 for the NPU) → pick a backend → benchmark → package in the APK/AAB.
- ONNX Runtime on Android, with Qualcomm AI Engine Direct for the Hexagon NPU ([ONNX Runtime mobile](https://onnxruntime.ai/docs/tutorials/mobile/)).
- TFLite / LiteRT with Qualcomm's QNN TFLite delegate ([TFLite Android](https://ai.google.dev/edge/lite-inference/android)).
- Qualcomm AI Runtime SDK (QAIRT) compiles ONNX / PyTorch / TF for CPU, GPU or Hexagon NPU ([Qualcomm docs](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/overview.html)). Use the NPU for the convolution net, the CPU for the filter logic.

## Android Sensors HAL 2.0
- Flow: IMU hardware → Sensors HAL → Event FMQ → SensorManager → app callback. The framework calls `initialize()` with two FMQ descriptors; sensors are configured with `batch()` and `activate()`. ([AOSP](https://source.android.com/docs/core/interaction/sensors/sensors-hal2))

## OpenXR tracking-loss rules
- `POSITION_VALID_BIT` = position is usable; `POSITION_TRACKED_BIT` = actively tracked. Same pair for orientation.
- On tracking loss, keep returning inferred / last-known pose (inertial dead reckoning, neck model) and clear the TRACKED bit; if the pose is no longer meaningful, clear VALID. ([spec](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html), [flags](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrSpaceLocationFlagBits.html))

## Books (Perplexity's picks)
Groves (2nd ed.); Bar-Shalom, Li, Kirubarajan; Thrun, Burgard, Fox; Barfoot; Solà et al. micro Lie theory.

## PhD theses (Perplexity's picks, *titles and years not yet verified*)
- Christian Forster, University of Zurich, 2016 (IMU pre-integration).
- Anastasios Mourikis, University of Minnesota, 2008 (MSCKF).
- Shaojie Shen, 2014 (VIO for micro aerial vehicles; VINS lineage).
- Tong Qin, HKUST (VINS-Mono).
- Patrick Geneva, University of Delaware (OpenVINS).
