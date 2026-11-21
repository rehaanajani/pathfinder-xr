# Ideas tried and not kept

Each idea below was built, scored on the organisers' damage (17 recordings, several corruption seeds) and left out of
the entry because it did not improve the scored windows overall. The pull request holds the code, the numbers and the
reasoning; all were closed unmerged on 21 Nov 2026 so that `main` holds only the final entry.

| PR | Idea | By |
|---|---|---|
| [#39](https://github.com/rehaanajani/pathfinder-xr/pull/39) | Zero-phase pull back to the start: tested and not kept | Mayank Yadav |
| [#41](https://github.com/rehaanajani/pathfinder-xr/pull/41) | Offline RTS smoother over position: tested and not kept | Mayank Yadav |
| [#42](https://github.com/rehaanajani/pathfinder-xr/pull/42) | Spike repair with one-row look-ahead: tested and not kept | Mayank Yadav |
| [#43](https://github.com/rehaanajani/pathfinder-xr/pull/43) | Noise-adaptive still detection: tested and not kept | Mayank Yadav |
| [#47](https://github.com/rehaanajani/pathfinder-xr/pull/47) | Blackout start velocity from a short average: tested and not kept | Mayank Yadav |
| [#48](https://github.com/rehaanajani/pathfinder-xr/pull/48) | Offline two-pass start: tested, not kept (near miss) | Mayank Yadav |
| [#50](https://github.com/rehaanajani/pathfinder-xr/pull/50) | Network experiments: platform choice, ensemble, calibration, handheld network, corridors (none kept) | Debadrita Halder |
| [#54](https://github.com/rehaanajani/pathfinder-xr/pull/54) | Failure-state estimate for the drone blind window: tested and not kept | Mayank Yadav |
| [#55](https://github.com/rehaanajani/pathfinder-xr/pull/55) | Accelerometer-scale estimate: tested and not kept | Mayank Yadav |
| [#56](https://github.com/rehaanajani/pathfinder-xr/pull/56) | Competitor code notes, refreshed scoreboard (main @ 38b2481) and Ideas we learned from | Mayank Yadav |
| [#57](https://github.com/rehaanajani/pathfinder-xr/pull/57) | Competitor training notes, and VIO-Edge's soft gate on handheld (not kept) | Debadrita Halder |
| [#58](https://github.com/rehaanajani/pathfinder-xr/pull/58) | Rests of 1.5 s and damage-aware trust on handheld (not kept) | Debadrita Halder |
| [#59](https://github.com/rehaanajani/pathfinder-xr/pull/59) | Machine Hall blind: diagnosis of the gap to vortexxx (no entry change) | Mayank Yadav |
| [#60](https://github.com/rehaanajani/pathfinder-xr/pull/60) | Drone network fade along the damage ramp: tested and not kept | Mayank Yadav |
| [#61](https://github.com/rehaanajani/pathfinder-xr/pull/61) | Rotor-drag offset: tested and not kept (flight better, blind worse) | Mayank Yadav |
| [#64](https://github.com/rehaanajani/pathfinder-xr/pull/64) | Blind window on the drone: diagnosis and a whole-flight smoother (not kept) | Debadrita Halder |
| [#65](https://github.com/rehaanajani/pathfinder-xr/pull/65) | VIO-Edge's regularisation for the handheld network, leave-one-room-out (not kept) | Debadrita Halder |

Earlier ideas that were measured and dropped are in [`blind_and_handheld.md`](blind_and_handheld.md),
[`retrain_organiser.md`](retrain_organiser.md) and [`coupling_sweep.md`](coupling_sweep.md). Two more were tested in
the final run and left out: a loop closure for handheld recordings (walkers end 0.6 to 1.1 m from where they started) and
a reshaped drone drift removal (better checkpoints and dropouts, worse blind on seeds 1, 7 and 2026;
[`loop_power.json`](loop_power.json)).
