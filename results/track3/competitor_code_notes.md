# Competitor code notes: SRMIST_vortexxx and ZeroDrift

Read line by line on 20 Nov 2026, in separate clones outside this repository: `RaktimChandra/pathfinder` @ `a67f4ee`
(SRMIST_vortexxx) and `Gem2005/VITChennai_ZeroDrift` @ `9c03e6d` (ZeroDrift). Neither repository has a licence, so no
code is copied: every idea we tried is our own implementation, credited in the README. Line numbers refer to those
commits. "Ours" = `main` @ 38b2481. Scoreboard numbers: `competitors.md` (organisers' script, seed 42, all 17).

## How each gets the position at the failure right on the Machine Hall

**vortexxx (blind 1.65 m on the Machine Hall).** Translation is one offline problem over the whole file, the blind
window included: a Kalman filter plus RTS smoother on 0.4 s blocks (`translation`, `src/main.py:605-701`) with state
position, velocity, world acceleration offset, workspace centre and a 2-axis drag offset. Inputs: block-mean world
acceleration. Measurements every block: rotor-drag body velocity (Cauchy-weighted, 678-686), zero velocity at every
rest of ≥ 1.5 s anywhere (`detect_static_all` 339-374, used at 822), and a soft prior that position stays within
σ = 3.5 m (horizontal, EuRoC) / 1 m (vertical) of the free centre (`Rr`, 640; applied every block). After the IMU dies
the velocity decays with τ = 0.2 s (`Fb`, 622-628) and the smoother runs through those blocks too. So the failure
position comes from velocity measurements (drag) integrated by a smoother that sees the whole flight, and it is held.
The drag model is a body-velocity measurement, which keeps the trajectory's real extent; the landing point (within
1 m of take-off on MH_01-05) then aligns correctly.

**ZeroDrift (Machine Hall blind 2.24 m, checkpoints 2.92 m: the best).** A two-pass 13-state error-state EKF at 10 Hz
gives a dead-reckoned track, then `spline_trend` (`src/es_ekf.py:672-722`) removes the slow drift by penalised least
squares: the track stays bounded, the drift is smooth (cut-off period 30 s → 15 s along the damage ramp, 82-85), and
in flight rotor drag enters as *displacement* information over a band up to 60-120 s (`drag_aided_trend` 843-918,
drag coefficient a fixed prior 0.25 /s at 97), with the off-axis drag bias eliminated exactly as a polynomial in
time. The blind window is extrapolated from a weighted fit of the last 3 s (`blind_extrapolate` 993-1055) with damped
velocity (τ 0.75 s), acceleration and yaw rate.

## Ideas, one by one

| Idea | Where | What it does | Why it might help us | Do we do it? |
|---|---|---|---|---|
| Duplicate rows collapsed | vortexxx `main.py:213`; ZeroDrift `inverse_degrade.py:418` | rows with a repeated timestamp are dropped, the pose copied back | duplicates must not double-integrate | **Yes, already**: a zero time step is skipped (`PathFinder._step`) |
| Stuck runs treated as missing and interpolated | ZeroDrift `inverse_degrade.py:474-487`, `505-513`; vortexxx `main.py:253` (any repeat invalid), `276` (interpolated) | identical consecutive rows with dt > 0 are replaced by linear interpolation in time between healthy neighbours | we distrusted only the third repeat onward; longer runs became short dropouts that reset our filters | **Tried, PR #53**: blind 1.741 → 1.590 m (4-seed mean, all 17), but some recordings newly lose to standing still |
| Spike repair | ZeroDrift Hampel 5-point, `inverse_degrade.py:309-351`, `488-503`; vortexxx running median-9 + 8·MAD gate, `main.py:259-268` | single-row outliers on several channels replaced by neighbours | spikes went straight into integration | **Tried, PR #42**: no measurable effect (before rotor drag) |
| Noise-adaptive smoothing of the raw IMU | ZeroDrift `_smooth_chunk` 353-383, `inverse_degrade.py:516-532` (binomial passes ∝ (t/T)^1.7) | zero-phase pre-smoothing that grows with the damage | less white noise into integration | No (our tilt low-pass, network window and drag low-pass already smooth) |
| Process noise from the damage ramp | vortexxx `noise_schedule` 741-777 (sev at 748, excess gyro scatter 770-771); ZeroDrift `process_noise` 455-472 (the script's own constants) | late, damaged samples count less in the smoother/EKF | vortexxx's ablation: biggest single gain (ATE 1.39 → 2.51 m without it) | Partly: our zero-phase drone pass uses the ramp (48 → 12 s). **Tried** noise-adaptive rest detection (PR #43, worse: vibration) |
| Rest detection anywhere | vortexxx `detect_static_all` 339-374 | ZUPT at every rest ≥ 1.5 s, not only the start | anchors velocity whenever the drone lands | **Yes**: `zero_velocity` runs on every sample |
| Gyro across gaps | vortexxx trapezoidal rate 436-441; ZeroDrift RK4 sub-steps `_rk4_step` 249, `_expand_gaps` 275, `preintegrate` 325 | a turn during a dropout is integrated with both edge readings | 30° tilt loss on room5 (vortexxx) | **Yes, kept** (#45, dropout bridging) |
| Two-pass, re-linearised biases | ZeroDrift pass 1 / pass 2, `es_ekf.py:1095-1113` | biases learned on the whole file seed the second pass | better start state | **Yes, kept** (#52, `--two-pass 10`, gyro bias) |
| Accelerometer-scale state | ZeroDrift error state `dk`, clamped ±5 %, `es_ekf.py:655` | tracks the scale random walk of the damage | gravity misread by 0.1-0.2 m/s² | **Tried, PR #55**: worse on every set |
| Zero-phase drift removal, cut-off shortening with severity | ZeroDrift `spline_trend` 672-722, periods 82-85 | slow drift removed without lag | our pull to the start lagged | **Yes, kept** on the drone (#51, 48 → 12 s); on handheld not kept (`blind_and_handheld.md`) |
| Drag velocity as displacement information over long periods | ZeroDrift `drag_aided_trend` 843-918; vortexxx drag measurement 676-686 | the slow part of position comes from drag, not from a pull | the Machine Hall gap (ZeroDrift checkpoints 2.92 vs our 2.86 now; blind 2.24 vs our 2.98) | Partly: drag corrects our velocity causally (Cauchy-weighted); the offline pass only high-passes position |
| Workspace-centre state | vortexxx state `c`, prior `Rr` 640 (σ 3.5 m EuRoC, 3 m otherwise, 1 m vertical) | position pulled to a free centre, not to the start | a better bounded-space prior | No; **tried** an RTS smoother with a centre state before rotor drag (PR #41: worse than standing still on room5/room6) |
| Blind: hold position | vortexxx τ = 0.2 s, 622-628 | the last velocity is least reliable | blind error is set by the position at failure | Our blind rows keep the causal entry (pulled to start); hold from a smoothed state **tried** twice (`blind_and_handheld.md`, PR #54): worse on held-out |
| Blind: fit of the last 3 s | ZeroDrift `blind_extrapolate` 993-1055 | damped extrapolation of a smoothed terminal state | | **Tried** an averaged blackout velocity (PR #47): no gain |
| Hard-coded EuRoC platform | vortexxx `PLATFORMS` 37-46 (gravity direction, gyro bias, drag k = [−0.1665, −0.2010] fitted on EuRoC), fingerprint cone 48 | known sensor → known constants | | No, by design: our mount guard detects the drone from gravity, and our drag constants are fitted per axis (`rotor_drag.md`) |

## Constants worth knowing (and where they came from)

* vortexxx (`P_DEFAULT`, 50-70): block 0.4 s; qa 0.1, qb 0.003, qd 0.0003; room_h 3.0 (3.5 for the EuRoC platform,
  "minimax ATE/const over all 10 sequences" — chosen with the Machine Hall in view); room_v 1.0; rdrag 0.10;
  Cauchy threshold NIS 4; blind_qa 0.3; blind_tau 0.2 s (leave-one-sequence-out, same pick in 17 folds); zupt σ 0.02;
  sev_floor 0.02, sev_gamma 1.7 (the organisers' script); qa_exc 6, qb_sev 0.15, qb_exc 0.3.
* ZeroDrift (`PARAMS`, 62-103): block 0.1 s; trend periods 30 → 15 s (14 → 7 s without a standstill calibration);
  drag μ 0.25 /s (a prior; "estimating μ from the data was tried and does not work"); drag periods 120 → 60 s; flight
  test windows; blind fit 3 s, velocity damping 0.75 s; and the organisers' `CONFIG` copied verbatim
  (`inverse_degrade.py:34-54`) as the white-box noise model.
