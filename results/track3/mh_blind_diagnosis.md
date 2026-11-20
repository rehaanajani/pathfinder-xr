# Machine Hall blind 5 s: why we trail SRMIST_vortexxx (diagnosis)

Base: `main` @ 920eeb0 (rotor drag, dropout bridging, drone zero-phase pass, two-pass start, stuck-run repair).
SRMIST_vortexxx @ `a67f4ee`. Same damaged files for both (organisers' script, accel-first, seeds 42, 1, 7, 2026);
raw scores `mh_blind_scoreboard.json` (`tools/competitor_scoreboard.py run --entries ours vortexxx still --tag mhblind`),
per-recording split `mh_blind_diagnosis.json` (`tools/analyse_mh_blind.py`).

## Where the gap is

| blind 5 s, mean (m) | seed 42: ours / vortexxx | seed 1: ours / vortexxx | seed 7: ours / vortexxx | seed 2026: ours / vortexxx |
|---|---:|---:|---:|---:|
| all 17 | **1.44** / 1.49 | 1.62 / **1.58** | 1.73 / **1.58** | 1.57 / **1.33** |
| Machine Hall | 2.22 / **1.65** | 2.71 / **1.86** | 2.89 / **1.99** | 2.56 / **1.44** |
| Vicon | **1.58** / 2.19 | **1.73** / 2.31 | **1.84** / 2.19 | **1.68** / 1.94 |
| TUM-VI | 0.65 / **0.64** | **0.61** / 0.62 | **0.64** / 0.64 | 0.64 / **0.61** |

We lead blind on the Vicon rooms at every seed and are level on TUM-VI; the whole gap is the Machine Hall. Seed 2026
(not in the brief's table) trails too: 1.57 vs 1.33 m, again all from the Machine Hall (2.56 vs 1.44 m).

## Per recording: error at the failure vs motion during the 5 s

Alignment exactly as the scorer's blind window (SE(3) fitted on the mission part). "Error at failure" is the error of
the last row before the IMU dies; "motion error" is the error of the displacement since then (what the blind model
adds); "moved in blind" is how far the estimate, vortexxx's estimate and the truth move during the 5 s; the last
column is how far each estimate is, at the failure, from its own first row, and how far the truth is from its start.

| recording | seed | blind ours / vortexxx | error at failure ours / vortexxx | motion error ours / vortexxx | estimate moved in blind ours / vortexxx / truth | estimate at failure to own start: ours / vortexxx; truth to start |
|---|---|---:|---:|---:|---:|---:|
| MH_01_easy | 42 | 2.23 / 3.20 | 1.79 / 2.88 | 0.67 / 0.44 | 0.67 / 0.08 / 0.65 | 2.18 / 2.69; 0.45 |
| MH_02_easy | 42 | 3.43 / 3.04 | 3.31 / 2.75 | 0.72 / 0.31 | 0.91 / 0.11 / 0.37 | 2.83 / 4.36; 0.35 |
| MH_03_medium | 42 | 1.49 / 0.88 | 1.45 / 0.69 | 0.71 / 0.36 | 0.98 / 0.25 / 0.32 | 2.72 / 1.45; 0.27 |
| MH_04_difficult | 42 | 1.50 / 0.47 | 1.35 / 0.58 | 0.54 / 0.25 | 0.84 / 0.17 / 0.27 | 2.43 / 3.15; 0.31 |
| MH_05_difficult | 42 | 2.48 / 0.68 | 2.12 / 0.22 | 0.71 / 0.62 | 0.24 / 0.07 / 0.85 | 1.32 / 2.18; 0.91 |
| MH_01_easy | 1 | 3.99 / 4.08 | 3.64 / 3.82 | 0.40 / 0.39 | 0.25 / 0.03 / 0.65 | 2.07 / 3.86; 0.45 |
| MH_02_easy | 1 | 3.72 / 3.00 | 3.36 / 2.76 | 0.42 / 0.25 | 0.42 / 0.05 / 0.37 | 0.95 / 3.89; 0.35 |
| MH_03_medium | 1 | 1.36 / 1.07 | 1.06 / 0.94 | 0.59 / 0.27 | 0.79 / 0.09 / 0.32 | 2.13 / 0.98; 0.27 |
| MH_04_difficult | 1 | 1.11 / 0.34 | 0.73 / 0.39 | 0.50 / 0.25 | 0.78 / 0.17 / 0.27 | 2.30 / 3.25; 0.31 |
| MH_05_difficult | 1 | 3.37 / 0.82 | 3.00 / 0.52 | 0.74 / 0.61 | 0.30 / 0.09 / 0.85 | 1.98 / 1.69; 0.91 |
| MH_01_easy | 7 | 3.20 / 4.08 | 2.91 / 3.80 | 0.55 / 0.41 | 0.40 / 0.05 / 0.65 | 1.10 / 3.13; 0.45 |
| MH_02_easy | 7 | 3.07 / 2.80 | 2.83 / 2.52 | 0.88 / 0.30 | 1.16 / 0.13 / 0.37 | 3.20 / 3.98; 0.35 |
| MH_03_medium | 7 | 1.77 / 1.17 | 1.82 / 0.99 | 0.59 / 0.32 | 0.89 / 0.16 / 0.32 | 2.52 / 1.31; 0.27 |
| MH_04_difficult | 7 | 2.96 / 1.14 | 2.81 / 1.00 | 0.36 / 0.22 | 0.47 / 0.13 / 0.27 | 1.36 / 2.73; 0.31 |
| MH_05_difficult | 7 | 3.46 / 0.75 | 2.98 / 0.27 | 0.76 / 0.60 | 0.23 / 0.08 / 0.85 | 1.85 / 2.29; 0.91 |
| MH_01_easy | 2026 | 2.65 / 2.71 | 2.17 / 2.41 | 0.50 / 0.42 | 0.58 / 0.06 / 0.65 | 2.62 / 2.64; 0.45 |
| MH_02_easy | 2026 | 3.80 / 1.94 | 4.26 / 1.65 | 0.97 / 0.30 | 1.42 / 0.11 / 0.37 | 4.73 / 3.73; 0.35 |
| MH_03_medium | 2026 | 1.11 / 0.39 | 1.20 / 0.45 | 0.82 / 0.33 | 1.13 / 0.24 / 0.32 | 3.00 / 1.15; 0.27 |
| MH_04_difficult | 2026 | 1.83 / 0.44 | 1.72 / 0.51 | 0.58 / 0.26 | 0.89 / 0.18 / 0.27 | 2.48 / 3.19; 0.31 |
| MH_05_difficult | 2026 | 3.40 / 1.73 | 2.93 / 1.09 | 0.75 / 0.65 | 0.26 / 0.06 / 0.85 | 1.86 / 1.69; 0.91 |

Worst three gaps (ours minus vortexxx): MH_05 seed 7 (+2.72 m), MH_05 seed 1 (+2.55 m), MH_02 seed 2026 (+1.86 m). Plots of the last 15 s:
[`mh_blind_MH_05_difficult_s7.png`](mh_blind_MH_05_difficult_s7.png),
[`mh_blind_MH_05_difficult_s1.png`](mh_blind_MH_05_difficult_s1.png),
[`mh_blind_MH_02_easy_s2026.png`](mh_blind_MH_02_easy_s2026.png).

## What it shows

1. **The gap is the position at the failure, not the blind model.** Our error at the failure is 0.7-4.3 m against
   vortexxx's 0.2-3.8 m; on MH_05 it is 2.1-3.0 m against 0.2-1.1 m. The 5 s add a further 0.4-1.0 m for us (our
   estimate keeps moving 0.2-1.4 m, pulled to the start and coasting) against 0.2-0.6 m for vortexxx (it holds:
   ≤ 0.25 m).
2. **Why the position at the failure is wrong: our speed collapses in the last ~16 s.** On MH_05 (seeds 7 and 1) and
   MH_04 (seed 7) the drone flies ~10 m back to its landing pad in the last 15 s at 0.7-1.25 m/s. Our filter's
   horizontal speed in 2 s bins drops to 0.1-0.6 m/s and follows the velocity network almost exactly (network
   0.13-0.74 m/s); the rotor-drag reading over the same bins is 0.6-1.8 m/s. vortexxx follows the return flight with
   its drag-driven smoother. The heading-uncertainty shrink of the network velocity is not the cause (its factor stays
   at 0.96-1.00, because dropouts are now bridged).
3. **The rotor-drag reading has a near-constant offset.** Against the true body velocity, the first drag axis reads
   −0.46 to −0.72 m/s too low on all seven development drone flights (median −0.60), early and late in each file;
   the second axis is recording-dependent (−1.3 to +0.0). vortexxx carries a drag-offset state for this; we do not.
4. **The drone has not landed when the IMU dies.** Its true speed over the last 2 s is 0.25-1.0 m/s and it still
   moves 0.27-0.85 m during the blind window, so a landing (rest) detector before the failure has nothing to catch.
5. **Little reversible damage is left in the last 20 s.** Over the five Machine Hall files per seed: 35-50 duplicated
   timestamps (already skipped: a zero time step), 4-15 stuck repeats left after the stuck-run repair, and 900-950
   gaps, the longest 0.24 s, all within the 0.5 s dropout bridge.

## Attempts that followed (each has its page)

* `late_drag_trust.md`: fade the network velocity on the drone along the damage ramp. Blind better on average, but
  through the Vicon rooms; Machine Hall flat, held-out Machine Hall worse, checkpoints slightly worse. Not kept.
* `drag_bias.md`: remove the drag offset (tracked, or fixed from the development fit). The flight improves
  (checkpoints, dropouts, ATE) but blind gets worse at both seeds: our blind rows come from the causal estimate, pulled
  toward the start, and a faster, unbiased causal velocity moves them away from the landing pad (≈ the start on the
  Machine Hall). A failure-state smoother with offset-corrected drag readings is still far worse. Not kept.

**Conclusion.** Closing the Machine Hall blind gap needs the right position at the failure on the drone, which needs
both a correct late velocity (drag with its offset, network trusted less late) and a blind window that no longer
relies on the pull to the start. Each half alone makes blind worse; vortexxx does both in one smoother over the whole
file. That is a larger change than fits before the freeze.
