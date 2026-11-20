# Soft gate on the network velocity, handheld mounts (20 Nov 2026): not kept

Idea from VIO-Edge (MIT licence; `src/main.py` `update_velocity`, `nn_gate = 16`; see
[`competitor_training_notes.md`](competitor_training_notes.md) V1):

- if the network's velocity disagrees with the filter's by more than both uncertainties allow, inflate the
  network's standard deviation by √(d²/gate) instead of trusting it fully;
- d² = Σᵢ (v_net,i − v_i)² / (σ_net,i² + σ_f²) over the three world axes;
- our filter keeps no covariance, so its own velocity std σ_f is a fixed option.

Applied in `learned_update` only when rotor drag is off (handheld mounts), so the drone recordings are untouched.
Tested with `--model-gate` (gate, 3-DoF chi-squared) and `--model-gate-sf` (σ_f). The code was not kept.

`python tools/exp_option.py --name soft_gate sweep`, organisers' script; raw rows in [`soft_gate.json`](soft_gate.json):


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | gate 16, sf 0.3 | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | gate 9, sf 0.3 | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | gate 16, sf 0.1 | 1.430 | 1.296 | 1.622 | 1.272 | 0.361 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | gate 4, sf 0.1 | 1.430 | 1.297 | 1.622 | 1.272 | 0.362 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | gate 1, sf 0.1 | 1.430 | 1.298 | 1.622 | 1.273 | 0.362 | 2.73 / 3.20 | 1.30 / 1.42 | 0.59 / 0.63 | 11/11 |

**Not kept.** No setting moves the score, not even gate 1 with σ_f 0.1 m/s, which inflates the network's std on
almost every update: the TUM-VI rows stay at 0.59 / 0.63 m. This matches VIO-Edge's own finding (devlog D-051: "the
innovation gate never fires on these sets") and our earlier one that the score hardly depends on how much the filter
trusts the network ([`network_calibration.md`](network_calibration.md): `--model-vref` 0.3-5 changes ATE by
< 1 %). Because development showed no effect at all, the frozen check (all 17, four seeds) was not run.
