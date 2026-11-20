# Rotor-drag offset: tested, **not kept**

**Idea** (from `mh_blind_diagnosis.md` and vortexxx's drag-offset state, `competitor_code_notes.md`): against the
true body velocity, our first rotor-drag reading is 0.46-0.72 m/s too low on every development drone flight (median
−0.60) and the second is recording-dependent (median −0.30), so the drag correction pulls our velocity toward a biased
value. Implementation in this branch's history (commit "Added an optional rotor-drag offset ...", reverted in the
next): `--drag-off1/--drag-off2` subtract fixed offsets (fitted on the development flights against ground truth);
`--drag-bias-tau` instead tracks the offset as the slow mean of (reading − our body velocity), skipping readings
farther than `--drag-robust` from it. "entry" = `main` @ 920eeb0. Development sweep at seeds 42 and 1.

**Result: not kept.** Removing the offset improves the flight but worsens the blind window at both seeds:

* fixed −0.6 / −0.3: seed 42 checkpoints 1.226 → 1.188, dropouts 1.198 → 1.173, Machine Hall ATE 2.58 → 2.43, but
  blind 1.401 → 1.535 (seed 1: 1.629 → 1.792);
* tracker, τ = 10-90 s: checkpoints slightly better, blind 1.564-1.689 at seed 42 and 1.776-1.877 at seed 1 (worse).
  Tracking against our own velocity also absorbs the network's late under-read, so it removes real motion too.

Our blind rows come from the causal estimate, which is pulled toward the start; on the Machine Hall the drone lands
within a metre of the start. A faster, unbiased causal velocity moves the blind rows away from that point. Not checked
on held-out data, since blind — the target — got worse on development data.

**Failure-state smoother with offset-corrected drag readings.** The smoother of `failure_state.md` (#54), re-run
offline with the offset removed from the drag readings: still far worse than the entry (blind 1.9-6.6 m), so in the
last seconds the drag readings fail for more than their offset at a fixed trust.

| drag offset (m/s) | seed | setting | ATE | checkpoints | blind 5 s | dropouts | Machine Hall blind | Vicon blind |
|---|---|---|---:|---:|---:|---:|---:|---:|
| -0.6 / -0.3 | 42 | entry | 1.354 | 1.226 | 1.401 | 1.198 | 2.38 | 1.41 |
| -0.6 / -0.3 | 42 | rts 5 hold | 1.379 | 1.226 | 1.908 | 1.204 | 3.33 | 2.10 |
| -0.6 / -0.3 | 42 | rts 10 hold | 1.458 | 1.226 | 3.061 | 1.214 | 5.30 | 3.79 |
| -0.6 / -0.3 | 42 | rts 10 coast | 1.452 | 1.226 | 2.957 | 1.214 | 5.03 | 3.70 |
| -0.6 / -0.3 | 42 | rts 20 hold | 1.643 | 1.226 | 4.990 | 1.231 | 8.40 | 6.77 |
| -0.6 / -0.3 | 1 | entry | 1.383 | 1.265 | 1.629 | 1.231 | 3.02 | 1.59 |
| -0.6 / -0.3 | 1 | rts 5 hold | 1.404 | 1.265 | 2.061 | 1.236 | 3.35 | 2.53 |
| -0.6 / -0.3 | 1 | rts 10 hold | 1.464 | 1.265 | 2.767 | 1.243 | 3.22 | 4.57 |
| -0.6 / -0.3 | 1 | rts 10 coast | 1.466 | 1.265 | 2.796 | 1.243 | 3.28 | 4.60 |
| -0.6 / -0.3 | 1 | rts 20 hold | 1.538 | 1.265 | 3.422 | 1.250 | 3.48 | 6.18 |
| -0.6 / 0.0 | 42 | entry | 1.354 | 1.226 | 1.401 | 1.198 | 2.38 | 1.41 |
| -0.6 / 0.0 | 42 | rts 5 hold | 1.400 | 1.226 | 2.221 | 1.206 | 4.03 | 2.44 |
| -0.6 / 0.0 | 42 | rts 10 hold | 1.537 | 1.226 | 3.964 | 1.221 | 7.29 | 4.78 |
| -0.6 / 0.0 | 42 | rts 10 coast | 1.528 | 1.226 | 3.846 | 1.221 | 7.03 | 4.65 |
| -0.6 / 0.0 | 42 | rts 20 hold | 1.826 | 1.226 | 6.603 | 1.245 | 12.28 | 8.29 |
| -0.6 / 0.0 | 1 | entry | 1.383 | 1.265 | 1.629 | 1.231 | 3.02 | 1.59 |
| -0.6 / 0.0 | 1 | rts 5 hold | 1.399 | 1.265 | 1.952 | 1.235 | 2.87 | 2.59 |
| -0.6 / 0.0 | 1 | rts 10 hold | 1.475 | 1.265 | 2.629 | 1.242 | 2.53 | 4.71 |
| -0.6 / 0.0 | 1 | rts 10 coast | 1.477 | 1.265 | 2.666 | 1.243 | 2.61 | 4.75 |
| -0.6 / 0.0 | 1 | rts 20 hold | 1.610 | 1.265 | 3.811 | 1.255 | 3.71 | 7.08 |


### Development sweep (11 development recordings; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | drag bias 10 s | 1.353 | 1.202 | 1.689 | 1.193 | 0.311 | 2.57 / 2.83 | 1.26 / 1.87 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag bias 30 s | 1.362 | 1.213 | 1.668 | 1.205 | 0.312 | 2.58 / 2.79 | 1.29 / 1.84 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag bias 90 s | 1.360 | 1.221 | 1.564 | 1.204 | 0.313 | 2.58 / 2.63 | 1.28 / 1.68 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | entry | 1.354 | 1.226 | 1.401 | 1.198 | 0.314 | 2.58 / 2.38 | 1.26 / 1.41 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | drag offset -0.6 / 0 | 1.343 | 1.219 | 1.460 | 1.194 | 0.309 | 2.54 / 2.40 | 1.26 / 1.56 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag offset -0.6 / -0.3 | 1.318 | 1.188 | 1.535 | 1.173 | 0.308 | 2.43 / 2.56 | 1.27 / 1.65 | 0.53 / 0.65 | 11/11 |
| dev, seed 42 | drag offset -0.3 / 0 | 1.343 | 1.219 | 1.411 | 1.191 | 0.311 | 2.55 / 2.36 | 1.25 / 1.46 | 0.53 / 0.65 | 11/11 |
| dev, seed 1 | drag bias 10 s | 1.390 | 1.243 | 1.877 | 1.232 | 0.313 | 2.64 / 3.40 | 1.31 / 1.99 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag bias 30 s | 1.397 | 1.255 | 1.857 | 1.243 | 0.314 | 2.63 / 3.34 | 1.34 / 1.98 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag bias 90 s | 1.394 | 1.264 | 1.776 | 1.241 | 0.314 | 2.63 / 3.22 | 1.33 / 1.85 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | entry | 1.383 | 1.265 | 1.629 | 1.231 | 0.314 | 2.62 / 3.02 | 1.31 / 1.59 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | standing still | 2.294 | 2.139 | 2.322 | 2.105 | 0.740 | 4.19 / 4.67 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 1 | drag offset -0.6 / 0 | 1.368 | 1.251 | 1.708 | 1.223 | 0.311 | 2.57 / 3.09 | 1.30 / 1.76 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag offset -0.6 / -0.3 | 1.357 | 1.233 | 1.792 | 1.217 | 0.310 | 2.50 / 3.27 | 1.33 / 1.85 | 0.53 / 0.62 | 11/11 |
| dev, seed 1 | drag offset -0.3 / 0 | 1.370 | 1.255 | 1.646 | 1.222 | 0.312 | 2.59 / 3.02 | 1.29 / 1.64 | 0.53 / 0.62 | 11/11 |

