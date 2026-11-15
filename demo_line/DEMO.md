# Running the Path Finder demo

Open Terminal, go to the project folder, and use `./demo.sh` with one word:

```bash
cd ~/Desktop/pathfinder-xr
./demo.sh check
```

| Command | What it shows | Time |
| --- | --- | --- |
| `./demo.sh check` | What's installed: data, models, phone | 2 s |
| `./demo.sh tests` | All tests pass | 20 s |
| `./demo.sh quick` | 3 real headset recordings, 16 failure situations: normal tracker vs ours. Opens a report | 1 min |
| `./demo.sh page` | The interactive page: pick a failure, watch the paths | instant |
| `./demo.sh headset` | Same as quick, on 12 recordings | 4 min |
| `./demo.sh camera` | Real cameras + real IMU (TUM-VI): dark, blur, camera off | 3 min |
| `./demo.sh phone` | Speed test on a USB-connected Android phone | 3 min |

## What to say while it runs

1. **The problem.** A headset tracks your head with cameras and motion sensors. When one goes bad (dark
   room, blur, a glitching sensor), normal tracking drifts or jumps.
2. **What Path Finder does.** It checks every sensor many times a second, ignores the one that's lying,
   and keeps tracking with what's left, saying how unsure it is.
3. **Show `quick`.** Point at three rows:
   - **camera off:** normal 0.33 m off, ours 0.11 m
   - **camera off + biased sensor:** 2.35 m vs 0.12 m
   - **camera off + saturated sensors:** 5.58 m vs 0.47 m

   Lower is better (metres away from the true position).
4. **Be upfront about the cost.** With nothing wrong, ours is about 1 cm behind. That's the price of caution.
5. **Show `camera`.** This is real stereo camera footage with real darkness and blur applied to the
   images. Ours leads in 5 of 7 situations.
6. **Show `phone`** (if a phone is connected): it runs on a real Samsung phone chip, with the
   milliseconds measured.

## What's real and what isn't

- **Real:**
  - Meta's TLIO headset recordings (real IMU, real head motion)
  - TUM-VI (real stereo cameras, real IMU, motion-capture truth)
  - the phone timing
- **Simulated, and labelled as such:**
  - the extra IMUs in the multi-IMU tests (copies of the one real IMU with their own noise and bias)
  - the camera position fixes in the headset benchmark (TLIO has no images)
- **Don't demo on simulated walks.** The learned model was trained on real head motion and gets
  simulated walks wrong, and they are also faster than real heads.

## If something fails

- **"Python setup missing":** the script installs it itself the first time (about 2 min).
- **"MISSING Meta TLIO data" or "motion model":** run `bash reproduce.sh` once. It downloads the data
  and models.
- **Phone not found:**
  1. Settings > About phone > Software information > tap **Build number** 7 times.
  2. Developer options > **USB debugging** ON.
  3. Use a data cable and tap **Allow** on the phone.
- **Results are saved in** `runs/demo_*`, never over the committed results, so you can rerun safely.
