# Retraining with VIO-Edge's regularisation, judged by leave-one-room-out (21 Nov 2026): not kept

Idea (VIO-Edge, MIT licence; `src/research/train_vel.py`; [`competitor_training_notes.md`](competitor_training_notes.md)
V3, V4, V6): our body-frame network memorises its training rooms, so train it the way VIO-Edge does:

- dropout 0.3 on the pooled features and the hidden layer;
- weight decay 0.02;
- 15 epochs.

The body-frame network also gets a 3-epoch MSE warm-up before the NLL, and per-window synthetic noise and bias. Both
inputs go through the same leave-one-room-out gate as [`network_handheld.md`](network_handheld.md):
`tools/train_airio.py` / `tools/train_velocity.py` with `--dropout 0.3 --wd 0.02 --epochs 15` (body-frame also
`--mse-epochs 3 --noise-aug 1`).

Window-level end-velocity error (m/s, training seeds 0 / 1) against the unregularised runs of that page (raw:
[`network/loro_window_regularised.json`](network/loro_window_regularised.json)):

| Unseen room (run) | body-frame | body-frame, regularised | gravity-aligned | gravity-aligned, regularised |
|---|---:|---:|---:|---:|
| room1 (room1 left out) | 0.83 / 0.87 | 0.90 / failed | 0.70 / 0.70 | 0.73 / 0.73 |
| room5 (room1 left out) | 0.84 / 0.83 | 0.88 / failed | 0.70 / 0.69 | 0.72 / 0.72 |
| room2 (room2 left out) | 0.75 / 0.79 | 0.88 / 0.84 | 0.42 / 0.42 | 0.45 / 0.47 |
| room5 (room2 left out) | 0.78 / 0.74 | 0.78 / 0.81 | 0.64 / 0.64 | 0.67 / 0.68 |
| room3 (room3 left out) | 0.92 / 1.00 | failed / 1.02 | 0.45 / 0.42 | 0.44 / 0.46 |
| room5 (room3 left out) | 0.79 / 0.78 | failed / 0.81 | 0.67 / 0.66 | 0.68 / 0.67 |
| room4 (room4 left out) | 0.66 / 0.62 | 0.63 / 0.61 | 0.62 / 0.60 | 0.59 / failed |
| room5 (room4 left out) | 0.79 / 0.75 | 0.82 / 0.76 | 0.63 / 0.64 | 0.68 / failed |

"failed": three runs died (two with CUDA out of memory while sharing the GPU), and were not rerun because the
outcome was already clear.

**Not kept.** Regularisation makes both inputs slightly worse on unseen rooms: body-frame by up to 0.13 m/s,
gravity-aligned by 0.01-0.05 m/s, with room4 level. VIO-Edge's recipe suits its MLP; for our convolutional networks the
limit on unseen rooms is not overfitting that dropout or weight decay can remove.

The trainer options stay (off by default) so the result can be reproduced.
