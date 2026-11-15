# Outside review of the IMU-only entry (15 Nov 2026)

We described the entry and our held-out numbers to Gemini, Grok and Meta AI and asked what we are doing wrong,
what already exists, and what to change. Then we tested every suggestion that needs only a setting, with the C++
entry on the validation recordings (EuRoC MH_04 + V1_03, seeds 1-3; room5 is not downloaded). The held-out set was
not touched.

## What they agreed on

1. **Our network is loosely coupled.** TLIO's strength is the *stochastic-cloning EKF*: the network's displacement
   over the window corrects velocity, biases and tilt in one consistent update. We nudge position and velocity.
2. **For EuRoC (a drone), pedestrian priors are weak.** All three point to **AirIO** (drone inertial odometry,
   trained and tested on EuRoC): feed the network *body-frame* IMU with gravity kept in, plus the attitude as an
   input, instead of a gravity-aligned window. They also suggest a 2 s window.
3. **Gyro denoising** (Brossard et al., a small dilated CNN on EuRoC / TUM-VI) is cheap and improves attitude,
   the main error source (1° of tilt = 0.17 m/s² of fake acceleration).
4. **Track the biases online** instead of fixing them at the start; AI-IMU's learned noise covariances are the
   pattern for degradation.
5. **Evaluate exactly like the scorer**: real `evo_ape` with the same alignment flags, per checkpoint, and the
   blind 5 s cut at many random times.

Repos and papers as the reviewers cited them (links not yet checked by us): TLIO (arxiv.org/abs/2007.01867, github.com/CathIAS/TLIO), AirIO (arxiv.org/abs/2506.10607),
AI-IMU (github.com/mbrossar/ai-imu-dr), gyro denoising (github.com/mbrossar/denoise-imu-gyro), TinyOdom
(github.com/nesl/tinyodom, small TCNs for tight budgets), SSIO (arxiv.org/abs/2506.08016).

## What we tested, and what was wrong in their advice

Validation mean, metres (ATE / checkpoints / blind 5 s / dropouts):

| Change | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| **Current default** | **4.06** | **3.96** | **2.54** | **3.84** |
| Stand still | 4.17 | 4.05 | 2.72 | 3.95 |
| No pull back to the start (`--pos-tau 0`) | 4.96 | 4.44 | 6.52 | 5.68 |
| No brakes at all (pull, leak, speed limit) | 5.07 | 4.53 | 6.79 | 5.83 |
| No velocity leak / no speed limit / no residual-acceleration removal / no still detection | 4.06 | 3.96 | 2.54 | 3.84 |
| Shrink the output 0.5-0.9x (Meta AI's idea) | 4.07-4.11 | | | |
| Blind 5 s: velocity decay 0.3-3 s, turn-rate decay 0.1-2 s | 4.06 | 3.96 | 2.54-2.56 | 3.84 |
| Pull back to start 5 s instead of 3 s | 4.00 | 3.92 | 2.44 | 3.80 |
| Gravity-correction trust 0.3-3 m/s², gain 0.1-1, low-pass 0.5 s (Gemini: drone banking fools it) | 4.06-4.07 | 3.96-3.97 | 2.52-2.55 | 3.83-3.87 |
| Separate, stronger pull for height (0.3-10 s) | 4.05-4.07 | 3.93-3.97 | 2.55-2.57 | 3.84-3.87 |
| Pull toward our own running average instead of the start (15-120 s) | 4.04-4.06 | 3.85-3.95 | 2.62-3.20 | 3.88-4.01 |
| Stop dead the moment the IMU dies (5 recordings incl. TUM-VI) | 2.40 vs 2.40 | 2.30 vs 2.30 | 1.90 vs 1.91 | 2.54 vs 2.54 |

- **"Remove the brakes" is wrong for us.** All three said the pull-to-start, leak and speed limit fight the
  network. Removing the pull makes everything worse (4.06 -> 4.96 m); the others do nothing, because the network's
  velocity already replaces them.
- **Output shrinkage does not help** (ATE gets slightly worse).
- **The blind 5 s cannot be fixed after the fact**: its error is the position error already accumulated when the
  IMU dies (Mayank found the same). Better tracking before the failure is the only lever.
- `--pos-tau 5` is 1-2 % better on validation: within tuning noise; not changed.

Why nothing moves (from Gemini's derivation, consistent with these numbers): after SE(3) alignment,
ATE² = σ_true² + σ_est² − 2ρσ_trueσ_est, where σ is a trajectory's spread around its centroid and ρ how well the
shapes match. Standing still scores exactly σ_true. The pull to start keeps σ_est small, so ATE can only beat standing
still in proportion to ρ, the shape match, and ρ comes from the network. Settings that do not change the network's
output cannot change ρ.

## What we will do

The cheap knobs are exhausted. The remaining gains need a better model and a tighter filter:

| Change | Who | Why |
|---|---|---|
| AirIO-style input (body frame + gravity + attitude, try 2 s window), same weights file format | Debadrita | the one change all three reviewers rank highest for EuRoC |
| Gyro denoiser (Brossard recipe) in front of the attitude filter | Debadrita, after the above | attitude is the largest error source |
| Real `evo_ape` with the scorer's flags on all 11 EuRoC + TUM-VI rooms; blind 5 s cut at random times | Mayank | so every choice is made on the real metric |
| Stochastic-cloning update of the network displacement (TLIO) in `main.py` and `main.cpp` | Rehaan | tighter coupling; lets the network correct velocity and bias |
