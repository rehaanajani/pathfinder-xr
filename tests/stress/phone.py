"""Replay (perturbed) real phone recordings through tools/live_demo/server.Live, the way analyze.py does."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tools.live_demo.server as S

RUNS = Path(__file__).resolve().parents[2] / "runs" / "live"
SESSIONS = {"A": RUNS / "phone_session_A_32s_and_192s_outages.jsonl",
            "B": RUNS / "phone_session_B_61s_outage.jsonl"}


def have_sessions():
    return all(p.exists() for p in SESSIONS.values())


def load_lines(name):
    out = [json.loads(l) for l in SESSIONS[name].read_text().splitlines() if l.strip()]
    for i, m in enumerate(out):
        m["_id"] = i                       # stable id so perturbed replays can be matched to the baseline
    return out


def raw_text(name):
    return SESSIONS[name].read_text()


def replay(lines, set_=None, crash_ok=False):
    """Feed message dicts (or raw strings, parsed like serve_phone/analyze do) into a fresh Live.
    Returns dict(log=[per camera frame], crash=exception or None, proc_s, dur_s)."""
    S.OURS_SET.clear()
    S.OURS_SET.update(set_ or {})
    for v in S.STATE["paths"].values():
        v.clear()
    live = S.Live()
    log, crash = [], None
    t0 = time.time()
    n_imu_t = []
    try:
        for m in lines:
            if isinstance(m, str):
                m = json.loads(m)
            live.line(m)
            if m.get("k") == "ar" and live.trk and live.trk["ours"].out is not None:
                o, s = live.trk["ours"].out, live.trk["standard"].out
                log.append({"id": m.get("_id"), "t": m["t"], "tracking": m.get("s") == "TRACKING",
                            "ar": None if live.ar_pos is None else live.ar_pos.copy(),
                            "ours": o["position"].copy(), "std": s["position"].copy(),
                            "ours_std": o["position_std_m"], "valid": o["valid"], "reason": o["reason"],
                            "tracked": o["tracked"]})
            if m.get("k") in ("g", "a"):
                n_imu_t.append(m.get("t", 0))
    except Exception as e:                      # noqa: BLE001
        if not crash_ok:
            raise
        crash = e
    proc = time.time() - t0
    S.OURS_SET.clear()
    return {"log": log, "crash": crash, "proc_s": proc, "live": live}


def outages(log, min_frames=30):
    """Runs of not-tracking ARCore frames -> error of ours/standard vs ARCore when it returns."""
    out, i = [], 0
    while i < len(log):
        if not log[i]["tracking"]:
            j = i
            while j < len(log) and not log[j]["tracking"]:
                j += 1
            if j < len(log) and j - i >= min_frames:
                ref, before = log[j]["ar"], log[j - 1]
                out.append({"seconds": (log[j]["t"] - log[i]["t"]) / 1e9,
                            "ours_m": float(np.linalg.norm(before["ours"] - ref)),
                            "std_m": float(np.linalg.norm(before["std"] - ref)),
                            "claimed": float(before["ours_std"]), "valid": bool(before["valid"]),
                            "reason": before["reason"]})
            i = j
        else:
            i += 1
    return out


# ---- perturbations (all seeded) -----------------------------------------------------------------
def p_none(L, seed):
    return L


def p_shuffle_local(L, seed, span=400):
    """Out-of-order delivery: lines swapped within +-span lines (like a jittery Wi-Fi link)."""
    rng = np.random.default_rng(seed)
    key = np.arange(len(L)) + rng.uniform(-span, span, len(L))
    return [L[i] for i in np.argsort(key)]


def p_shuffle_global(L, seed):
    rng = np.random.default_rng(seed)
    return [L[i] for i in rng.permutation(len(L))]


def p_duplicates(L, seed, frac=0.2):
    rng = np.random.default_rng(seed)
    out = []
    for m in L:
        out.append(m)
        if rng.random() < frac:
            out.append(m)
    return out


def p_drop_imu(L, seed, frac=0.3):
    rng = np.random.default_rng(seed)
    return [m for m in L if not (m["k"] in ("g", "a") and rng.random() < frac)]


def p_drop_arcore_20s(L, seed, start_s=None):
    """Remove every ARCore line in a 20 s window (default: starting 40 s into the session)."""
    t0 = min(m["t"] for m in L if m["k"] == "g")
    s0 = t0 + int((40 if start_s is None else start_s) * 1e9)
    return [m for m in L if not (m["k"] == "ar" and s0 <= m["t"] + 0 < s0 + int(20e9))
            and not (m["k"] == "ar" and m["t"] > 0 and _ar_in(m, L, s0))]


def _ar_in(m, L, s0):
    return False


def p_drop_arcore_by_arrival(L, seed, start_s=40.0, dur_s=20.0):
    """Drop ARCore lines in a 20 s stretch of *arrival* (position in the stream, by the IMU clock)."""
    out, t_last = [], None
    t0 = None
    for m in L:
        if m["k"] == "g":
            t0 = m["t"] if t0 is None else t0
            t_last = m["t"]
        if m["k"] == "ar" and t_last is not None and start_s * 1e9 <= t_last - t0 < (start_s + dur_s) * 1e9:
            continue
        out.append(m)
    return out


def p_clock_offset_cam(L, seed, off_s=10.0):
    return [dict(m, t=m["t"] + int(off_s * 1e9)) if m["k"] == "ar" else m for m in L]


def p_clock_offset_imu(L, seed, off_s=10.0):
    return [dict(m, t=m["t"] + int(off_s * 1e9)) if m["k"] in ("g", "a") else m for m in L]


def p_clock_offset_accel_only(L, seed, off_s=10.0):
    """Gyro and accelerometer on different clocks (the real phone has them ~2 ms apart)."""
    return [dict(m, t=m["t"] + int(off_s * 1e9)) if m["k"] == "a" else m for m in L]


PERTURBATIONS = {"none": p_none, "shuffle_local": p_shuffle_local, "shuffle_global": p_shuffle_global,
                 "duplicates": p_duplicates, "drop_imu_30pct": p_drop_imu,
                 "drop_arcore_20s": p_drop_arcore_by_arrival, "cam_clock_+10s": p_clock_offset_cam,
                 "imu_clock_+10s": p_clock_offset_imu, "accel_clock_+10s": p_clock_offset_accel_only}


def garbage_frames(log, err_m=2.0):
    """Frames where ARCore is tracking, ours claims valid, yet ours is further than err_m from ARCore."""
    n = bad = 0
    for x in log:
        if x["tracking"] and x["ar"] is not None:
            n += 1
            if x["valid"] and np.linalg.norm(x["ours"] - x["ar"]) > err_m:
                bad += 1
    return bad, n


def garbage_vs_baseline(base_log, log, err_m=2.0):
    """Silent garbage = ARCore frames where the baseline replay agreed with ARCore (< 1 m), ARCore did not
    itself jump, yet the perturbed replay reports valid=True and sits more than err_m from ARCore."""
    b = {x["id"]: x for x in base_log if x["id"] is not None}
    prev, n, bad = None, 0, 0
    for x in log:
        y = b.get(x["id"])
        if x["tracking"] and x["ar"] is not None and y is not None and y["ar"] is not None and y["tracking"]:
            agree = np.linalg.norm(y["ours"] - y["ar"]) < 1.0
            smooth = prev is None or np.linalg.norm(x["ar"] - prev) < 0.3
            prev = x["ar"]
            if agree and smooth:
                n += 1
                if x["valid"] and np.linalg.norm(x["ours"] - x["ar"]) > err_m:
                    bad += 1
        else:
            prev = None if not x["tracking"] else x["ar"]
    return bad, n
