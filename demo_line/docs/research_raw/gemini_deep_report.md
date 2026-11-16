# Gemini (Argon High) answer: maths, lineage, FDI, benchmarks (14 Nov 2026)

The full answer is about 56k characters. Below are the parts that change what we build or let us compare our numbers. Links and citations are Gemini's; numbers marked *unverified* by Gemini stay unverified.

## Maths we use (confirms our implementation)
- ESKF on SO(3) (Solà 2017): nominal + 15-dim error state, `R = R̂·Exp(δθ)`, discrete `F_k`, `G_k`, Joseph-form update, and a **reset Jacobian** `J_reset = diag(I, I, I − ½[δθ]×, I, I)` after injection. *We do not apply the reset Jacobian yet (small-angle approximation).*
- Forster pre-integration: ΔR, Δv, Δp with first-order bias correction and recursive 9x9 noise covariance.
- TLIO measurement: yaw-only gravity-aligned frame `h(X) = R_γiᵀ(p_j − p_i)` with Jacobians on clone i and j (stochastic cloning). *We approximate this by holding the past position.*
- NIS gating thresholds (3 dof): 7.815 (95 %), 11.345 (99 %); soft gating via IRLS / robust Kalman (Chang, IEEE TAES 2014; Agamennoni et al., ICRA 2011).
- beta-NLL: mean gradient scaled by `σ^(2β−2)`; β = 0.5 recommended; explains why plain NLL collapsed (matches what we saw).

## Learned inertial odometry numbers (from EqNIO's tables, arXiv 2408.06321)
| Method | TLIO dataset ATE / RTE | Aria (AEA) ATE / RTE |
|---|---|---|
| TLIO (+ rotation aug.) | 1.812 m / 0.500 m | 1.285 m / 0.464 m |
| TLIO (paper setting) | 1.722 m | - |
| EqNIO O(2) | **1.433 m** / 0.458 m | **1.118 m** / 0.416 m |
| TLIO, no rotation aug. | - | 1.969 m / 0.834 m |
These are IMU-only, full test sequences (3-7 min), no camera.

Other lineage numbers: RoNIN-ResNet ATE 5.14 m (unseen) / 3.54 m (seen); AirIMU on TUM-VI RPE 0.18 m; AirIO up to 86.6 % better than prior on Blackbird; KISS-IMU APE about 2.5 m vs TLIO 21-66 m on Botanic Garden; SSIO within 0.07 m/s of supervised AirIO.

## Multi-IMU fault detection: the aerospace method (most useful)
- **Lever-arm compensation** before comparing accelerometers: `a_B = R(ã − b) − ([ω]×² + [ω̇]×) r`.
- **Parity space** (Sturza 1988; RAIM / redundant strapdown IMU literature): stack all IMU axes `z = Hx + η + f`; parity matrix `V` spans the left nullspace of `H` (`VH = 0`); parity vector `p = Vz` **does not depend on the true motion**.
- **Detection (GLRT):** `D = pᵀp / σ²` ~ χ²(M−3) when healthy; alarm when above the χ² threshold for the chosen false-alarm rate.
- **Isolation:** per-IMU block statistic `I_k = pᵀV_k(V_kᵀV_k)⁻¹V_kᵀp / σ²`; the faulty IMU maximises it.
- **Redundancy rules:** 2 IMUs can **detect** one fault; 3 IMUs are needed to **isolate** it. (Matches our 0 % vs 99 % result.)
- Soft fusion: Student-t / IRLS weights around the component-wise median.

## Benchmarks with numbers
- EuRoC (ATE): DROID-SLAM 0.022 m (mono, visual only); ORB-SLAM3 0.035 m (mono-inertial); DM-VIO 0.06 m; Basalt about 0.07 m; OpenVINS 0.06-0.15 m.
- TUM-VI: DM-VIO 0.05 m on room1-6.
- TartanAir: DROID-SLAM 0.33 m mono / 0.17 m stereo.
- ETH3D-SLAM leaderboard: FoundationSLAM 0.069 m (#1); ORB-SLAM3 0.135 m; DROID-SLAM 0.171 m.
- Hilti 2026 leaderboard (as reported): Localization #1 ACDC-K (KAIST); **SamsungPoland #10**; SLAM #1 HKU. *Verify on hilti-challenge.com.*
- ICCV 2023 SLAM Challenge: 1st Samsung R&D China-Beijing (DVI-SLAM).
- Duke XR testbed (ISMAR 2025, arXiv 2508.08642): Vision Pro vs Quest 3 and others under Vicon.

## What we take from this
1. Replace our heuristic IMU voting with **parity-space GLRT detection + isolation** (standard, citable).
2. Score our learned IMU-only model on the **full TLIO test split** to compare directly with TLIO (1.72 m) and EqNIO (1.43 m).
3. Add the ESKF **reset Jacobian** (small, correct).
