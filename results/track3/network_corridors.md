# More handheld training data: TUM-VI corridors (20 Nov 2026): not kept

Idea: the body-frame network does not generalise across handheld rooms ([`network_handheld.md`](network_handheld.md));
more handheld data might fix that. TUM-VI corridors 1-5 have motion capture only at their start and end (51.9, 71.0,
60.0, 52.6, 39.7 s; 275 s in total), so only windows fully inside a motion-capture stretch are usable.

**Cache fix first** (`tools/train_airio.py`): the ground-truth mask used to drop only windows ending just after a gap
closed; a window ending inside a gap kept targets interpolated across it, which for a corridor is most of the
recording. Now every window whose span touches a ground-truth gap is dropped. Check: corridor1 keeps 9,938 1-s
windows for 51.9 s of motion capture in two stretches (expected about 9,980). `prep --extra tumvi:corridor1 ...`
adds recordings to the training split. Existing cache files were not rebuilt (EuRoC and rooms 1-6 have no long
ground-truth gaps).

**Gate** (`tools/loro_handheld.sh` with the corridors added to the training list), window-level end-velocity error,
m/s, against the body-frame runs without corridors ([`network/loro_window_corridors.json`](network/loro_window_corridors.json),
[`network/loro_window.json`](network/loro_window.json)):

| Unseen room (run) | body-frame, seeds 0 / 1 | body-frame + corridors, seeds 0 / 1 |
|---|---:|---:|
| room1 (room1 left out) | 0.83 / 0.87 | 0.79 / 0.78 |
| room5 (room1 left out) | 0.84 / 0.83 | 0.75 / 0.75 |
| room2 (room2 left out) | 0.75 / 0.79 | 0.76 / 0.77 |
| room5 (room2 left out) | 0.78 / 0.74 | 0.71 / 0.74 |
| room3 (room3 left out) | 0.92 / 1.00 | 0.80 / 0.85 |
| room5 (room3 left out) | 0.79 / 0.78 | 0.72 / 0.74 |
| room4 (room4 left out) | 0.66 / 0.62 | 0.73 / 0.62 |
| room5 (room4 left out) | 0.79 / 0.75 | 0.71 / 0.74 |

Pass rule fixed before the results (as for the handheld network): better on at least 4 of the 5 unseen rooms by more
than the seed spread. Better on room1, room3 and room5; a tie on room2; a tie or worse on room4. **3 of 5: not
kept**, and not taken further to end-to-end runs. The corridors do help on most rooms; a later attempt could add the
magistrale sequences (7-10 GB each) for more motion-capture time.
