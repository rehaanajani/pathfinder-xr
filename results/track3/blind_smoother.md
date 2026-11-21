# A whole-flight smoother for the drone's blind rows (21 Nov 2026): not kept

Goal: blind 5 s on MH_04 and MH_05, where vortexxx is far ahead (seed 1: 0.34 vs 1.11 m and 0.82 vs 3.37 m).
Rows before the failure were left exactly as shipped (zero-phase on the drone), so only blind could change.
Organisers' script; damaged inputs from `tools/competitor_scoreboard.py`; scored with `tools/score_track3.py` (SE(3)
fitted on the rows before the failure). Tool: `tools/exp_blind_smoother.py` (`diagnose`, `split`, `sweep`); raw rows
in [`blind_smoother/`](blind_smoother/). The top-down plots (`diagnose` writes them to `runs/blind_smoother/plots/`)
are not committed.

## 1. Diagnosis: the error is the position at the failure

Mean blind error (m) of each candidate for the blind rows, in the scorer's frame; and the error of our position at the
failure moment itself:

| Recording | Seed | zero-phase row at failure | causal row at failure | shipped blind rows | hold zero-phase row | hold un-pulled position | oracle: hold true failure position | oracle: best fixed point | true motion in blind |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_02_easy | 42 | 3.31 | 3.12 | 3.43 | 3.51 | 5.77 | 0.28 | 0.11 | 0.37 |
| MH_02_easy | 1 | 3.36 | 3.20 | 3.72 | 3.63 | 5.80 | 0.28 | 0.11 | 0.37 |
| MH_02_easy | 7 | 2.83 | 2.71 | 3.07 | 3.06 | 5.09 | 0.28 | 0.11 | 0.37 |
| MH_03_medium | 42 | 1.45 | 1.26 | 1.49 | 1.63 | 3.74 | 0.26 | 0.08 | 0.32 |
| MH_03_medium | 1 | 1.06 | 0.97 | 1.36 | 1.30 | 3.14 | 0.26 | 0.08 | 0.32 |
| MH_03_medium | 7 | 1.82 | 1.50 | 1.77 | 2.03 | 4.30 | 0.26 | 0.08 | 0.32 |
| MH_04_difficult | 42 | 1.71 | 1.35 | 1.50 | 1.76 | 3.14 | 0.19 | 0.08 | 0.27 |
| MH_04_difficult | 1 | 0.98 | 0.73 | 1.11 | 1.10 | 2.16 | 0.19 | 0.08 | 0.27 |
| MH_04_difficult | 7 | 3.29 | 2.81 | 2.96 | 3.35 | 4.98 | 0.19 | 0.08 | 0.27 |
| MH_05_difficult | 42 | 3.06 | 2.12 | 2.48 | 3.28 | 6.87 | 0.68 | 0.20 | 0.85 |
| MH_05_difficult | 1 | 3.88 | 3.00 | 3.37 | 4.15 | 7.73 | 0.68 | 0.20 | 0.85 |
| MH_05_difficult | 7 | 4.00 | 2.98 | 3.46 | 4.37 | 8.42 | 0.68 | 0.20 | 0.85 |

During the blind 5 s the drone moves only 0.3-0.85 m (it is landing): holding the true position at the failure would
score 0.2-0.7 m. Every candidate we have is 1-4 m off **at the failure itself**. The causal row (what ships) is closer
than the zero-phase row and far closer than the un-pulled run, which is why holding or coasting from those was
already worse.

## 2. Where that failure error comes from

Development drones, seeds 42 and 1 ([`blind_smoother/split.json`](blind_smoother/split.json)).

- **Late velocity:** the true position τ s before the failure plus our own un-pulled displacement over the last τ
  seconds. This is the error our velocity alone makes near the end.
- **Accumulated:** our shipped position τ s before the failure plus the true displacement. This is the error already
  there before that stretch.

| Recording | Seed | late velocity, 5 s | 15 s | 30 s | same, rotor drag off, 5 s | 15 s | 30 s | accumulated before last 5 s | 15 s | 30 s | hold take-off position | shipped blind |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01_easy | 42 | 1.68 | 3.04 | 3.50 | 2.14 | 3.46 | 4.03 | 1.46 | 1.15 | 1.79 | 3.80 | 2.23 |
| MH_01_easy | 1 | 1.82 | 2.46 | 4.21 | 2.14 | 3.25 | 4.87 | 1.94 | 2.84 | 0.96 | 3.68 | 3.99 |
| MH_02_easy | 42 | 1.32 | 5.63 | 3.77 | 1.46 | 5.98 | 4.51 | 2.05 | 2.90 | 3.13 | 4.05 | 3.43 |
| MH_02_easy | 1 | 1.74 | 5.16 | 4.62 | 1.89 | 5.77 | 5.84 | 1.59 | 2.44 | 2.55 | 3.81 | 3.72 |
| MH_03_medium | 42 | 0.84 | 2.98 | 2.23 | 0.98 | 3.66 | 2.46 | 1.01 | 2.49 | 0.51 | 2.91 | 1.49 |
| MH_03_medium | 1 | 0.76 | 2.35 | 2.38 | 1.10 | 2.96 | 2.79 | 0.68 | 1.88 | 1.80 | 2.91 | 1.36 |
| V1_01_easy | 42 | 0.75 | 0.67 | 3.16 | 0.57 | 1.07 | 3.10 | 0.64 | 1.02 | 0.87 | 2.06 | 1.06 |
| V1_01_easy | 1 | 0.65 | 0.63 | 3.17 | 0.55 | 0.84 | 3.18 | 0.87 | 1.55 | 0.59 | 2.15 | 1.24 |
| V1_02_medium | 42 | 0.65 | 0.86 | 2.14 | 0.90 | 0.98 | 2.84 | 1.35 | 1.15 | 0.85 | 0.74 | 0.75 |
| V1_02_medium | 1 | 0.65 | 1.60 | 2.80 | 0.84 | 1.52 | 3.41 | 1.48 | 0.65 | 0.85 | 0.57 | 1.09 |
| V2_01_easy | 42 | 0.90 | 1.83 | 2.28 | 1.14 | 2.56 | 4.98 | 1.56 | 2.60 | 1.77 | 3.07 | 2.17 |
| V2_01_easy | 1 | 0.77 | 1.10 | 2.77 | 1.09 | 1.99 | 5.45 | 1.80 | 2.57 | 1.98 | 3.27 | 2.05 |
| V2_02_medium | 42 | 1.15 | 2.28 | 2.13 | 0.85 | 3.13 | 3.63 | 1.93 | 1.45 | 0.51 | 2.63 | 1.68 |
| V2_02_medium | 1 | 0.94 | 2.42 | 2.50 | 0.86 | 3.34 | 4.31 | 1.95 | 1.01 | 0.60 | 2.73 | 1.99 |

- **Our late velocity is the problem.** Integrated over the last 15 s it misses by 0.6-5.6 m (MH_02 5.2-5.6 m), more
  than the error already accumulated by then (0.7-2.9 m). The most damaged part of the run is the end, so the end is
  where our velocity is worst.
- **Rotor drag is not to blame.** With drag off the late velocity is worse on most recordings, so dropping drag
  measurements near landing cannot help.
- **The heading shrink of the network velocity is not to blame.** It stays at 1.00 to the end of MH_05 and V2_01
  (0.995 on MH_02), measured by wrapping `learned_update`.
- **The take-off position (onset of rotor vibration) is no landing estimate either:** worse than the shipped blind
  rows on 11 of 14 cases.

## 3. The smoother

Per axis, a Kalman filter + RTS smoother at 2.5 Hz over the whole flight (our own code; the idea is vortexxx's
offline smoother with a workspace-centre prior, no code copied):

- state: position and velocity;
- velocity measurements: our un-pulled filter velocity with std `sv`, with none in the last `cut` seconds;
- a weak prior at every step: position = centre of our own shipped trajectory, std `kc` × its spread;
- the smoothed position at the failure is offset into the shipped frame, then held (`hold`) or moved with the shipped
  blind motion (`coast`).

Development, seeds 42 and 1 ([`blind_smoother/sweep.json`](blind_smoother/sweep.json)):

**Development drone recordings (EuRoC)**

| setting | seed 42 blind | seed 1 blind | mean blind | mean checkpoints |
|---|---:|---:|---:|---:|
| entry | 1.828 | 2.204 | 2.016 | 1.685 |
| rts r2.5 cut0 sv0.3 kc1 hold | 2.561 | 2.700 | 2.631 | 1.685 |
| rts r2.5 cut0 sv0.3 kc1 coast | 2.969 | 2.915 | 2.942 | 1.685 |
| rts r2.5 cut0 sv0.3 kc2 hold | 2.086 | 2.422 | 2.254 | 1.685 |
| rts r2.5 cut0 sv0.3 kc2 coast | 2.444 | 2.607 | 2.526 | 1.685 |
| rts r2.5 cut0 sv1 kc1 hold | 3.013 | 3.036 | 3.025 | 1.685 |
| rts r2.5 cut0 sv1 kc1 coast | 3.468 | 3.285 | 3.376 | 1.685 |
| rts r2.5 cut0 sv1 kc2 hold | 2.760 | 2.837 | 2.798 | 1.685 |
| rts r2.5 cut0 sv1 kc2 coast | 3.184 | 3.063 | 3.124 | 1.685 |
| rts r2.5 cut3 sv0.3 kc1 hold | 3.322 | 3.313 | 3.318 | 1.685 |
| rts r2.5 cut3 sv0.3 kc1 coast | 3.798 | 3.576 | 3.687 | 1.685 |
| rts r2.5 cut3 sv0.3 kc2 hold | 3.106 | 3.182 | 3.144 | 1.685 |
| rts r2.5 cut3 sv0.3 kc2 coast | 3.577 | 3.446 | 3.512 | 1.685 |
| rts r2.5 cut3 sv1 kc1 hold | 3.271 | 3.270 | 3.270 | 1.685 |
| rts r2.5 cut3 sv1 kc1 coast | 3.747 | 3.533 | 3.640 | 1.685 |
| rts r2.5 cut3 sv1 kc2 hold | 3.217 | 3.232 | 3.225 | 1.685 |
| rts r2.5 cut3 sv1 kc2 coast | 3.692 | 3.496 | 3.594 | 1.685 |
| rts r2.5 cut6 sv0.3 kc1 hold | 3.277 | 3.271 | 3.274 | 1.685 |
| rts r2.5 cut6 sv0.3 kc1 coast | 3.754 | 3.534 | 3.644 | 1.685 |
| rts r2.5 cut6 sv0.3 kc2 hold | 3.390 | 3.329 | 3.360 | 1.685 |
| rts r2.5 cut6 sv0.3 kc2 coast | 3.867 | 3.592 | 3.729 | 1.685 |
| rts r2.5 cut6 sv1 kc1 hold | 3.260 | 3.260 | 3.260 | 1.685 |
| rts r2.5 cut6 sv1 kc1 coast | 3.737 | 3.523 | 3.630 | 1.685 |
| rts r2.5 cut6 sv1 kc2 hold | 3.305 | 3.293 | 3.299 | 1.685 |
| rts r2.5 cut6 sv1 kc2 coast | 3.781 | 3.557 | 3.669 | 1.685 |

**All development recordings (TUM-VI rows are unchanged: no rotor drag)**

| setting | seed 42 blind | seed 1 blind | mean blind | mean checkpoints |
|---|---:|---:|---:|---:|
| entry | 1.401 | 1.629 | 1.515 | 1.245 |
| rts r2.5 cut0 sv0.3 kc1 hold | 1.868 | 1.945 | 1.906 | 1.245 |
| rts r2.5 cut0 sv0.3 kc1 coast | 2.127 | 2.082 | 2.104 | 1.245 |
| rts r2.5 cut0 sv0.3 kc2 hold | 1.565 | 1.768 | 1.667 | 1.245 |
| rts r2.5 cut0 sv0.3 kc2 coast | 1.793 | 1.886 | 1.840 | 1.245 |
| rts r2.5 cut0 sv1 kc1 hold | 2.155 | 2.159 | 2.157 | 1.245 |
| rts r2.5 cut0 sv1 kc1 coast | 2.444 | 2.317 | 2.381 | 1.245 |
| rts r2.5 cut0 sv1 kc2 hold | 1.994 | 2.032 | 2.013 | 1.245 |
| rts r2.5 cut0 sv1 kc2 coast | 2.264 | 2.176 | 2.220 | 1.245 |
| rts r2.5 cut3 sv0.3 kc1 hold | 2.352 | 2.335 | 2.343 | 1.245 |
| rts r2.5 cut3 sv0.3 kc1 coast | 2.655 | 2.503 | 2.579 | 1.245 |
| rts r2.5 cut3 sv0.3 kc2 hold | 2.215 | 2.251 | 2.233 | 1.245 |
| rts r2.5 cut3 sv0.3 kc2 coast | 2.514 | 2.420 | 2.467 | 1.245 |
| rts r2.5 cut3 sv1 kc1 hold | 2.319 | 2.308 | 2.313 | 1.245 |
| rts r2.5 cut3 sv1 kc1 coast | 2.622 | 2.475 | 2.549 | 1.245 |
| rts r2.5 cut3 sv1 kc2 hold | 2.285 | 2.284 | 2.284 | 1.245 |
| rts r2.5 cut3 sv1 kc2 coast | 2.587 | 2.452 | 2.519 | 1.245 |
| rts r2.5 cut6 sv0.3 kc1 hold | 2.323 | 2.308 | 2.316 | 1.245 |
| rts r2.5 cut6 sv0.3 kc1 coast | 2.627 | 2.476 | 2.551 | 1.245 |
| rts r2.5 cut6 sv0.3 kc2 hold | 2.395 | 2.345 | 2.370 | 1.245 |
| rts r2.5 cut6 sv0.3 kc2 coast | 2.699 | 2.512 | 2.606 | 1.245 |
| rts r2.5 cut6 sv1 kc1 hold | 2.313 | 2.302 | 2.307 | 1.245 |
| rts r2.5 cut6 sv1 kc1 coast | 2.616 | 2.469 | 2.542 | 1.245 |
| rts r2.5 cut6 sv1 kc2 hold | 2.341 | 2.323 | 2.332 | 1.245 |
| rts r2.5 cut6 sv1 kc2 coast | 2.644 | 2.490 | 2.567 | 1.245 |

**Not kept, and not taken to the 17-recording check.** Every setting is worse than the shipped blind rows. The best
(velocity std 0.3, prior 2x spread, hold) gives 2.25 m against 2.02 m on the development drones.

- **Without velocity near the end** (`cut` 3-6 s) the smoother falls back to the centre prior, about 3.3 m.
- **With it**, it inherits the late-velocity error of section 2.
- **In effect** a per-step centre prior plus velocity measurements is a forward-backward pull to the centroid: the
  same family as the zero-phase hold and the pull lengths that had already failed.

## What this means

A better blind window needs better velocity in the last 15-30 s of heavy damage, or a landing model, not a different
smoother over the same velocity. vortexxx's blind lead on MH_04 / MH_05 comes from its failure position being right,
which its drag-and-accelerometer smoother gets from velocity we do not have.
