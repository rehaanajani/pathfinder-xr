# Why our IMU-only tracking trails TLIO (diagnosis, 14 Nov 2026)

Full TLIO test split, IMU only, ATE without alignment (mean / median over sequences):

| Setup | Mean | Median |
|---|---:|---:|
| Published TLIO | 1.72 | - |
| Published EqNIO O(2) | 1.43 | - |
| Our network + our filter (36 seqs) | 6.27 | 5.13 |
| Same with stochastic cloning (36 seqs) | 6.99 | 4.45 |
| Our network + filter (first 18 seqs) | 4.75 | 3.51 |
| **True displacement + white noise 0.136 m** (same RMSE as our net) | **1.01** | 0.98 |
| True displacement + white noise 0.05 m | 0.38 | 0.37 |
| True displacement, no noise | 0.19 | 0.18 |
| **Our network fed the TRUE orientation** | **2.52** | **1.65** |

Conclusions
1. The filter is not the bottleneck: with displacement errors of our network's size but random, it beats TLIO.
2. Our network's errors are systematic (correlated), so they accumulate instead of averaging out.
3. About half the gap comes from orientation feedback: the filter's small tilt/yaw errors rotate the
   network input. TLIO's measurement is expressed in a yaw-only gravity-aligned frame with a rotation
   Jacobian, which corrects this; we use world-frame position-only updates.
4. The rest is training: ours is 10 epochs / 12 min on a laptop GPU; TLIO trained about 4 h on a DGX.

Scripts: `oracle_displacement.py`, `true_rotation.py`, `cloning_compare.py` (run from the repo root).
