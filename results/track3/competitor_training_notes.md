# How two competitors train and tune (read 20 Nov 2026)

Read line by line from public clones kept outside this repository:

- **VIO-Edge**: `rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation` at `87f5c4a`, MIT licence.
  Small pieces may be adapted with attribution.
- **vortexxx**: `RaktimChandra/pathfinder` at `a67f4ee`, no licence. Ideas only; any code here is our own.

Goal: better handheld (TUM-VI) velocity on rooms the network never saw. The "Do we" column is about the shipped
`src/main.py` on `main` (38b2481).

## First: what their TUM-VI numbers mean

VIO-Edge's shipped ensemble is trained on **all 17 recordings, rooms 5 and 6 included**:

- `devlog.md` D-044, "Shipped model: 3 members trained on all 17 sequences".
- D-053, "trained on all 11 EuRoC sequences and TUM-VI rooms 1-6, the held-out ones included".

So its TUM-VI scores on our scoreboard are in-sample for its network. Its own held-out figures are much weaker:

- D-046: TUM-VI before failure 1.41 m, against 1.16 for a fixed pose.
- EXP-023: 1.11 / 0.41 / 0.87 m (checkpoints / blind / dropouts), with rooms 3 and 6 held out.

Our rooms 5-6 numbers come from a network that never saw them. A matched comparison would either retrain ours on all
17 or score theirs with its held-out ensemble (`weights/mix/ho_ens.bin`, not in their repository). The VIO-Edge
devlog says the hidden test may reuse public sequences (D-012), and the organisers' FAQ allows any public EuRoC /
TUM-VI sequence for development.

vortexxx has **no learned model**. Its handheld result comes from a smoother with a workspace-centre state,
zero-velocity updates and a damage-following noise model (below).

## VIO-Edge (MIT)

| # | Idea | Where | What it does | Why it might help on unseen rooms | Do we |
|---|---|---|---|---|---|
| V1 | Soft Mahalanobis gate on the network velocity | `src/main.py:633-650` (`update_velocity`); `nn_gate = 16` at `:96` | d² = rᵀ(P_vv + σ²I)⁻¹r for the innovation r. If d² > 16, σ is inflated by √(d²/16) instead of rejecting the measurement | A wrong network velocity on an unfamiliar room is trusted less | No: our gain `vref²/(vref²+σ²)` (`learned_update`) ignores the innovation. **Caveat:** their D-051 says the gate "never fires on these sets" |
| V2 | Ensemble uncertainty = mean predicted variance + member disagreement | `src/main.py:474-494` (`VelNet.forward`) | 3 MLP members; log(mean var + between-member var) | Disagreement grows off-distribution | Tested 20 Nov: same as the best single seed (`network_ensemble.md`) |
| V3 | Strong regularisation | `src/research/train_vel.py` `--dropout 0.3 --wd 0.02 --epochs 15`; D-027 ("longer training overfit: val z² rose from 1.9 to 2.9") | Dropout 0.3 on both hidden layers, weight decay 0.02, few epochs | Our body-frame network memorises its rooms (seen 0.68 m vs left-out 0.97 m ATE; z² 1 on training, 2-11 on validation) | No: weight decay 1e-4, no dropout, 20 epochs |
| V4 | Heavy synthetic augmentation on clean rows | `train_vel.py` `AUG` and `make_batch`: accel noise up to 2.5 m/s² (u²-scaled), accel bias 0.3, gyro noise 0.05, gyro bias 0.05, tilt 2° | Noise and bias drawn per window | Wider input distribution, less memorisation | Partly: degradation copies at 3 noise levels, tilt 2°; no per-window synthetic noise |
| V5 | Organiser-damage copies at random severity and both column orders | `train_vel.py` `official_copy` / `build`: seeds 1000+, max level U(0.7, 1.3), gamma U(1.3, 2.1); D-037 | Degraded copies rebuilt as `main.py` sees them (invalid rows dropped, gaps interpolated) | Matches the scored damage | Tried 17 Nov (`retrain_organiser.md`): no gain |
| V6 | Warm-up with MSE, then Gaussian NLL | `train_vel.py` main loop: MSE for 3 epochs, then 0.5(e²/σ² + log σ²) | The mean settles before the variance | Better-behaved σ | No: MSE + β-NLL from the start |
| V7 | MLP on 4:1 block-averaged input (450-256-128-6) | `train_vel.py` `Net`, `make_batch` | 50 steps × 9 features; body-frame accel, gyro, gravity direction | Simpler model; D-041 found CNN and GRU better per step but worse end to end | No: 1-D CNN, 200 slots × 10 |
| V8 | Look-ahead velocity model used only inside the offline smoother | D-054; `src/main.py:1044-1053` | Second ensemble predicts the velocity 0.6 s before the window end and replaces the logged measurement for the smoother | Uses the future, which the task allows | No. Their TUM-VI gain was 1.12 -> 1.11 m (checkpoints): negligible |
| V9 | Drone / handheld switch from rotor vibration | D-055; `src/main.py:892-925` | Median 1 s RMS of the accel first difference: 2.1-3.8 m/s² drones, 0.17-0.22 handheld; threshold 1.0 | A platform test that does not depend on the mount | We use start-up gravity vs the thrust axis (mount guard, drag); same effect on EuRoC / TUM-VI |
| V10 | Workspace prior centred on the running centroid, σ = spread (handheld 0.6 × spread, after 5 s) | D-047, D-055; `src/main.py:1204` | Pulls toward the middle of the path so far, not the start | Rooms: the start is often at the edge | We pull to the start (handheld `pos_tau` 3 s) |
| V11 | Ground-truth velocity for TUM-VI from a ±3-sample central difference; samples near mocap gaps masked | `train_vel.py` `load_seq` | Velocity labels for motion-capture data | Cleaner label than our 0.25 s mean | Different: we use the 0.25 s mean over the window end |

## vortexxx (no licence: ideas only)

| # | Idea | Where | What it does | Why it might help on unseen rooms | Do we |
|---|---|---|---|---|---|
| X1 | Process noise follows the damage | `src/main.py:741-777` (`noise_schedule`) | Accel process noise √(qa² + (6·exc)²), where exc = measured gyro high-frequency scatter above the quiet start. Accel-offset random walk adds the documented ramp sev(t) = 0.02 + 0.98(t/T)^1.7. README ablation: TUM-VI 0.54 -> 1.25 m without it | Late in the run the accelerometer is worth less, so lean on priors | No: fixed trust in the network and the accelerometer all run long |
| X2 | Zero-velocity updates at every rest ≥ 1.5 s | `src/main.py:339-374` (`detect_static_all`); `zupt_sigma = 0.02` | Block (0.4 s) static if gyro std < 0.06, accel std < 1.05, \|a\| ≈ g ± 0.25, gyro mean steady (< 0.03). Only runs ≥ 1.5 s count, trimmed one block each end | Loose thresholds survive damage noise; the long-run test keeps false positives out | Partly: our still test (40 samples) uses gyro std < 0.01 rad/s, below the organisers' late gyro noise (up to 0.015), so late rests are missed. Mayank's #43 (noise-adaptive still detection) was not kept |
| X3 | Workspace centre as a filter state; σ 3 m horizontal, 1 m vertical | `src/main.py:51-58`, `translation` (`Hr`, `Rr`, centre states 9-11 with prior variance 100) | Position is pulled toward an estimated centre, not the start; vertical much tighter | Same as V10 | We pull to the start |
| X4 | Kalman filter + RTS smoother at 0.4 s blocks | `translation` | Smoothed position / velocity | Offline | Main has a zero-phase pull on the drone only. Handheld zero-phase smoothing lost on unseen rooms (`blind_and_handheld.md`) |
| X5 | Blind velocity decay 0.2 s, chosen by leave-one-sequence-out over 17 | `src/main.py:63-66` | Velocity at failure is the least reliable, so it is not extrapolated | | Ours 0.7 s; blind decay sweeps never moved our score |

## What we take into the next experiments (in order)

1. **V1, soft gate**, on handheld mounts: our version of the innovation test, with the network σ and a fixed filter
   velocity σ (we have no covariance).
2. **X1 + X2, damage-aware trust and rests**, handheld only:
   - network-velocity and accel-tilt trust scaled by measured excess noise against the quiet start;
   - zero-velocity updates at rests ≥ 1.5 s with loose thresholds.
3. **V3 + V4 + V6**, retraining the handheld gravity-aligned network with strong regularisation and per-window
   augmentation, judged by leave-one-room-out.

If anything is kept, credit VIO-Edge (V1, V3-V6) and vortexxx (X1, X2) in the README "Ideas we learned from"; code is
our own.
