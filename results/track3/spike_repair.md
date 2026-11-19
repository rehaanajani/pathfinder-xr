# Spike repair with a one-row look-ahead: tested, **not kept**

**Idea** (VIO-Edge repairs isolated spikes with one row of look-ahead; ZeroDrift uses a Hampel filter): our health
check drops only non-finite, all-zero, out-of-range and long frozen rows, so the organisers' spikes (up to ~9 m/s² and
1.5 rad/s on one row, all six channels) went straight into the integration. Implementation (commit "Added optional
spike repair ...", reverted in the next commit): a row is a spike when at least two of its six channels jump away from
both neighbours by more than T = max(k × running mean |row-to-row change|, floor) while the neighbours agree within
T/2; it is replaced by the neighbours' mean. On MH_04 / room5 / V1_01 it repairs 15 / 24 / 28 rows at k = 5
(the organisers injected 18 / 33 / 30, some on rows they later dropped).

**Result: no measurable effect, not kept.** On the 11 development recordings every threshold leaves ATE,
checkpoints, dropouts and drift unchanged to three decimals and moves blind by 0.004 m (2.222 → 2.218). One-row
spikes are too short to matter after the filter's own smoothing (tilt low-pass, gravity correction, the network's 1 s
window). "entry" below is `main` with dropout bridging (`gap_bridge.md`). Not checked on held-out data, since nothing
was gained on development data.


### Development sweep (11 development recordings, seed 42; mean metres)

| inputs | setting | ATE | checkpoints | blind 5 s | dropouts | drift 1 s | Machine Hall ATE / blind | Vicon ATE / blind | TUM-VI ATE / blind | rows ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, seed 42 | entry | 1.894 | 1.692 | 2.222 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.14 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | standing still | 2.295 | 2.139 | 2.323 | 2.095 | 0.740 | 4.19 / 4.68 | 2.01 / 2.21 | 1.16 / 0.66 | 11/11 |
| dev, seed 42 | despike 5 | 1.894 | 1.692 | 2.218 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.13 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | despike 8 | 1.894 | 1.692 | 2.218 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.13 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | despike 12 | 1.894 | 1.692 | 2.219 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.13 | 0.59 / 0.63 | 11/11 |
| dev, seed 42 | despike 5, floors /2 | 1.894 | 1.692 | 2.218 | 1.696 | 0.430 | 3.87 / 4.45 | 1.72 / 2.13 | 0.59 / 0.63 | 11/11 |

