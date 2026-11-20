# Blind-window starting points and handheld smoothing (tested, not kept)

Raw rows in `blind_and_handheld/`, from `tools/exp_zero_phase_drone.py` (`ZP_WORK=zp_blind` and `ZP_WORK=zp_hh`).
Development = 11 recordings at seed 42; held-out = MH_04, MH_05, V1_03, V2_03, room5, room6.

## Blind window after the drone smoothing

The blind rows of the entry are the causal trajectory's own. Starting them instead from the smoothed position at the
failure (held there, or coasting with the un-pulled motion, or a blend) improves blind on the development recordings
(hold: 1.622 -> 1.528 m) but makes it worse on held-out recordings at every seed (hold, seed 42: 1.690 -> 1.965 m).
Not kept: the smoothed position at the failure is not reliable on unseen recordings.

## Offline zero-phase smoothing on handheld mounts

Same smoothing on TUM-VI with short time constants (2-20 s). On rooms 1-4 (seen by the velocity network in training)
checkpoints drop from 0.539 to 0.375 m; on the unseen rooms 5-6 they get worse (0.90 -> 0.96 m at 6 s, -> 1.09 m at
10 s). Not kept.
