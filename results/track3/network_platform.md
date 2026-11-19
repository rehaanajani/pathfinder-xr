# Network by platform: gravity-aligned network for handheld TUM-VI (20 Nov 2026): not kept

Idea: keep the body-frame network for the drone mount and give every other mount, handheld TUM-VI included, the
gravity-aligned network (`velnet_weights_world.bin`), which `docs/AIRIO.md` found better on room5 at window level.
Test: TUM-VI rooms 1-5 with the mount guard forced to fire (`--mount-guard-deg 0.001`); the drone recordings are
unaffected by this choice. The fallback coupling (`FALLBACK_COUPLING`) equals the TUM-VI defaults (pull 3 s, no
position coupling), so there is no separate coupling to test.

Organisers' `degrade_imu.py` (their seed 42 unless stated), accel-first input, C++ entry (`tools/eval_organiser.py`;
same output as `main.py` within 1e-6 m), metres. Choices on training and validation recordings only; the held-out
recordings (MH_05, V2_03, room6) and the fresh seeds 1, 7, 2026 were not used. Raw results:
[`network/`](network/); tables generated with `python tools/network_tables.py`.

| | ATE | Checkpoints | Blind 5 s | Dropouts |
|---|---:|---:|---:|---:|
| body-frame (current) | 0.723 | 0.666 | 0.639 | 0.690 |
| gravity-aligned (forced) | 0.816 | 0.762 | 0.673 | 0.786 |

| Recording | body-frame (current) | gravity-aligned (forced) |
|---|---:|---:|
| tumvi:room1 | 0.57 / 0.52 | 0.77 / 0.65 |
| tumvi:room2 | 0.66 / 0.61 | 0.70 / 0.68 |
| tumvi:room3 | 0.72 / 0.61 | 0.79 / 0.69 |
| tumvi:room4 | 0.75 / 0.73 | 0.96 / 0.97 |
| tumvi:room5 | 0.91 / 0.86 | 0.86 / 0.82 |

Per-recording cells: ATE / checkpoints.

**Not kept.** Rooms 1-4 are training data for both networks, so they favour whichever fits its training data best
and are not a fair test. On room5, the only room neither network trained on, the gravity-aligned network is 0.05 m
better on ATE but 0.06 m worse on blind 5 s (0.68 vs 0.62), which fails "blind does not get worse". A fair test
needs more unseen handheld rooms: see the leave-one-room-out gate in [`network_handheld.md`](network_handheld.md).
