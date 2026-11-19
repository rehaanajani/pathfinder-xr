# A handheld network: gravity-aligned input on TUM-VI (20 Nov 2026): not kept

Idea: `docs/AIRIO.md` found the older gravity-aligned input better than the body-frame one on handheld room5. If that
holds, keep the body-frame network for the drone and give handheld mounts a gravity-aligned network retrained with
the current pipeline. Organisers' `degrade_imu.py`, C++ entry, metres; raw results in [`network/`](network/). The
held-out recordings (MH_05, V2_03, room6) and the fresh seeds 1, 7, 2026 were not used.

## The trap: training rooms favour the body-frame network

Rooms 1-4 are training data, and the 17-recording table is mostly training recordings. Organisers' seed 42, ATE:

| Room | body-frame, room in training | body-frame, room left out | gravity-aligned, room in training | gravity-aligned, room left out |
|---|---:|---:|---:|---:|
| room1 | 0.57 | 0.96 | 0.73 | 0.86 |
| room2 | 0.66 | 0.91 | 0.68 | 0.70 |
| room3 | 0.72 | 1.04 | 0.78 | 0.86 |
| room4 | 0.75 | 0.98 | 0.93 | 1.08 |
| **mean** | **0.68** | **0.97** | **0.78** | **0.88** |

Seen rooms: body-frame 0.68 m, gravity-aligned 0.78 m. Left out: body-frame 0.97 m, gravity-aligned 0.88 m. The
body-frame network memorises the rooms it trained on (its stated std is calibrated there and 2-11x over-confident on
unseen recordings, [`network_calibration.md`](network_calibration.md)); the gravity-aligned one generalises better.

## Gate: leave one room out (`tools/loro_handheld.sh`)

Each of rooms 1-4 left out in turn, both inputs trained without it (two seeds each; each input with its own shipped
configuration: body-frame 42,716 parameters, gravity-aligned 27,020; 20 epochs). Each run picks its best epoch on a
pool of the left-out room, room5, MH_04 and V1_03 (the same for both inputs, slightly optimistic for both).
Window-level end-velocity error, m/s ([`network/loro_window.json`](network/loro_window.json)):

| Left-out room | body-frame, seeds 0 / 1 | gravity-aligned, seeds 0 / 1 | predict zero |
|---|---:|---:|---:|
| room1 | 0.83 / 0.87 | 0.70 / 0.70 | 1.10 |
| room5 (in run 1) | 0.84 / 0.83 | 0.70 / 0.69 | 1.04 |
| room2 | 0.75 / 0.79 | 0.42 / 0.42 | 1.02 |
| room5 (in run 2) | 0.78 / 0.74 | 0.64 / 0.64 | 1.04 |
| room3 | 0.92 / 1.00 | 0.45 / 0.42 | 1.02 |
| room5 (in run 3) | 0.79 / 0.78 | 0.67 / 0.66 | 1.04 |
| room4 | 0.66 / 0.62 | 0.62 / 0.60 | 0.65 |
| room5 (in run 4) | 0.79 / 0.75 | 0.63 / 0.64 | 1.04 |

Pass rule fixed before the results: gravity-aligned better on at least 4 of the 5 unseen rooms by more than the seed
spread. **Passed at window level** (rooms 1, 2, 3, 5; room4 a tie).

## End to end on unseen rooms

Rooms 1-4 with the left-out networks (2 training seeds x organisers' seeds 42, 101, 102); room5 with the shipped
body-frame network vs the new full-split gravity-aligned network (both seeds) over seeds 42 and 101-105:

| Unseen room | Body-frame: ATE / checkpoints / blind 5 s / dropouts | Gravity-aligned: ATE / checkpoints / blind 5 s / dropouts |
|---|---|---|
| room1 | 0.95 / 0.81 / 0.54 / 0.94 | 0.86 / 0.74 / 0.53 / 0.83 |
| room2 | 0.90 / 0.84 / 0.68 / 0.87 | 0.69 / 0.69 / 0.77 / 0.65 |
| room3 | 1.04 / 0.93 / 0.72 / 1.02 | 0.86 / 0.76 / 0.71 / 0.85 |
| room4 | 0.98 / 0.97 / 0.69 / 0.96 | 1.08 / 1.07 / 0.72 / 1.05 |
| room5 (shipped body-frame vs new gravity-aligned, 6 degradation seeds) | 0.90 / 0.85 / 0.65 / 0.83 | 0.83 / 0.81 / 0.76 / 0.79 |
| **mean of the five** | **0.95 / 0.88 / 0.66 / 0.92** | **0.87 / 0.81 / 0.70 / 0.83** |

The old `velnet_weights_world.bin` on room5 over the same seeds (ATE / checkpoints / blind 5 s / dropouts): 0.83 / 0.79 / 0.69 / 0.80.

ATE, checkpoints and dropouts are about 10 % better on four of the five unseen rooms; room4 (slow motion, velocity RMS
0.65 m/s) is worse on everything. Blind 5 s is worse on three of five (room5 by about 0.11 m with both seeds
averaged). The coupling cannot buy the blind back:

| Setting (new gravity-aligned seed 1, room5, 6 degradation seeds) | ATE | Blind 5 s |
|---|---:|---:|
| current coupling | 0.810 | 0.697 |
| --pos-tau 6 | 0.826 | 0.740 |
| --pos-tau 10 | 0.886 | 0.775 |
| --model-blind-tau 0.3 | 0.810 | 0.693 |
| --pos-tau-blind 1.0 | 0.810 | 0.695 |
| --pos-tau-blind 0.5 | 0.810 | 0.694 |

## Verdict: not kept

As a swap for handheld mounts it fails the acceptance rules on numbers we already have:

- on the 17-recording table the rooms 1-4 entries are training rooms, where the swap makes ATE, checkpoints and
  dropouts worse (rooms 1-5 at seed 42: ATE 0.723 -> 0.791);
- blind 5 s gets worse on unseen rooms;
- room6 (held out) moves slowly, like room4 (velocity RMS 0.54 m/s), the one unseen room where the gravity-aligned
  input lost everything, so the held-out look was not spent on it.

What it does show: on handheld motion the network does not generalise across rooms with the body-frame input, and a
gravity-aligned input does markedly better at window level (0.42-0.70 vs 0.75-1.00 m/s). If the scored input is a
recording we never trained on, that matters; if it is one of the public recordings, memorisation helps. That question
also bears on retraining on all 17 recordings.

## More handheld data (not started)

TUM-VI corridors 1-5 have motion capture only at their start and end: 51.9, 71.0, 60.0, 52.6 and 39.7 s (275 s in
total, about half of rooms 1-4 together). Using them needs a stricter mask in the window cache first (windows
overlapping a ground-truth gap must be dropped, not interpolated across), then the same gate.
