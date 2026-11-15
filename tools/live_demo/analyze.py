"""Score a recorded live session (runs/live/*.jsonl) offline.

    python tools/live_demo/analyze.py runs/live/<file>.jsonl

There is no motion-capture truth on a phone. Two honest checks instead:
  * Camera-cover outages: ARCore stops tracking; when it resumes, its pose is the best reference we
    have. Error of each tracker right before ARCore comes back = drift during the outage.
  * Return to start: the walker stopped on the starting spot, so distance between the first and
    last position is drift for every tracker (ARCore included).
Also reports how fast the laptop processes the stream (must be faster than real time).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tools.live_demo.server as S  # noqa: E402


def main(path):
    lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    imu_t = [m["t"] for m in lines if m["k"] in ("g", "a")]
    dur = (max(imu_t) - min(imu_t)) / 1e9
    live = S.Live()
    log = []
    t0 = time.time()
    for m in lines:
        live.line(m)
        if m["k"] == "ar" and m["t"] > 0 and live.trk and live.trk["ours"].out is not None:
            o, s = live.trk["ours"].out, live.trk["standard"].out
            log.append({"t": m["t"], "tracking": m["s"] == "TRACKING", "ar": live.ar_pos.copy() if live.ar_pos is not None else None,
                        "ours": o["position"].copy(), "std": s["position"].copy(), "ours_std": o["position_std_m"],
                        "reason": o["reason"]})
    proc = time.time() - t0
    print(f"session {dur:.0f} s of IMU; processed in {proc:.1f} s ({dur / proc:.1f}x real time)")
    # Outages: runs of not-tracking frames; compare at the first tracking frame after.
    outs, i = [], 0
    while i < len(log):
        if not log[i]["tracking"]:
            j = i
            while j < len(log) and not log[j]["tracking"]:
                j += 1
            if j < len(log) and j - i >= 30:                 # >= ~1 s of lost tracking (30 Hz frames)
                ref = log[j]["ar"]
                before = log[j - 1]
                outs.append({"frames": j - i, "seconds": (log[j]["t"] - log[i]["t"]) / 1e9,
                             "standard_m": float(np.linalg.norm(before["std"] - ref)),
                             "ours_m": float(np.linalg.norm(before["ours"] - ref)),
                             "ours_std_m": before["ours_std"]})
            i = j
        else:
            i += 1
    print(f"\ncamera-cover outages (>= 1 s): {len(outs)}")
    for k, o in enumerate(outs, 1):
        print(f"  {k}. {o['seconds']:4.1f} s  error when ARCore returned: standard {o['standard_m']:.2f} m, "
              f"ours {o['ours_m']:.2f} m (ours claimed +-{o['ours_std_m']:.2f} m, valid={o['ours_std_m'] < 1.0})")
    tr = [x for x in log if x["tracking"]]
    if tr:
        a0, a1 = tr[0]["ar"], tr[-1]["ar"]
        print(f"\nreturn to start (first vs last tracked pose): ARCore {np.linalg.norm(a1 - a0):.2f} m, "
              f"ours {np.linalg.norm(tr[-1]['ours'] - tr[0]['ours']):.2f} m, standard {np.linalg.norm(tr[-1]['std'] - tr[0]['std']):.2f} m")
        walked = sum(np.linalg.norm(np.diff(np.array([x['ar'] for x in tr]), axis=0), axis=1))
        print(f"walked (ARCore path length): {walked:.1f} m")
    out = Path(path_out := Path(path).with_suffix(".summary.json"))
    out.write_text(json.dumps({"session_s": dur, "realtime_factor": dur / proc, "outages": outs}, indent=1))


if __name__ == "__main__":
    import argparse
    from pathfinder.filter import FilterConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    for it in a.set:
        k, v = it.split("=", 1)
        S.OURS_SET[k] = type(getattr(FilterConfig(), k))(v)
    main(a.session)
