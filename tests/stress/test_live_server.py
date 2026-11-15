"""Real phone recordings (release phone-sessions-v1) through tools/live_demo/server.Live, perturbed.

Needs runs/live/phone_session_{A,B}_*.jsonl (gunzip the release assets). Slow tier.
"""
import json
import os
import time

import numpy as np
import pytest

import phone
import tools.live_demo.server as S

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not phone.have_sessions(), reason="phone sessions not in runs/live")]

REALISTIC = ["duplicates", "drop_imu_30pct", "drop_arcore_20s", "cam_clock_+10s", "imu_clock_+10s"]
_BASE = {}


def baseline(name):
    if name not in _BASE:
        _BASE[name] = phone.replay(phone.load_lines(name))
    return _BASE[name]


@pytest.mark.parametrize("name", ["A", "B"])
def test_baseline_reproduces_published_numbers(name):
    outs = [o for o in phone.outages(baseline(name)["log"]) if o["seconds"] < 400]
    want = {"A": [(32, 1.96), (192, 0.50)], "B": [(61, 0.60)]}[name]
    assert len(outs) == len(want)
    for o, (sec, err) in zip(outs, want):
        assert abs(o["seconds"] - sec) < 1 and abs(o["ours_m"] - err) < 0.1
        assert not o["valid"]            # claimed uncertainty is above 1 m after a long outage


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("pert", REALISTIC)
def test_realistic_link_problems_do_not_crash_or_lie(name, pert):
    base = baseline(name)
    lines = phone.PERTURBATIONS[pert](phone.load_lines(name), 1)
    r = phone.replay(lines, crash_ok=True)
    assert r["crash"] is None, repr(r["crash"])
    bad, n = phone.garbage_vs_baseline(base["log"], r["log"])
    assert bad <= 0.01 * n, f"{pert}/{name}: valid=True and >2 m from ARCore on {bad}/{n} frames"
    for o in phone.outages(r["log"]):
        if o["seconds"] < 400 and o["ours_m"] > 1.0:
            assert not o["valid"], f"{pert}/{name}: {o['ours_m']:.1f} m off but valid=True"


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.xfail(strict=False, reason="measured: 2.4-4% of frames valid and >2 m off after +-0.4 s reordering (STRESS_REPORT)")
def test_out_of_order_delivery(name):
    """Lines swapped within ~+-0.4 s (jittery Wi-Fi)."""
    base = baseline(name)
    r = phone.replay(phone.p_shuffle_local(phone.load_lines(name), 1), crash_ok=True)
    assert r["crash"] is None
    bad, n = phone.garbage_vs_baseline(base["log"], r["log"])
    assert bad <= 0.01 * n, f"{bad}/{n} frames valid=True and >2 m from ARCore after local reordering"


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.xfail(strict=False, reason="measured: 5-8% of frames valid and >2 m off with a 10 s gyro/accel skew (STRESS_REPORT)")
def test_accelerometer_clock_offset(name):
    """Gyro and accelerometer on clocks 10 s apart (real phone: ~2 ms)."""
    base = baseline(name)
    r = phone.replay(phone.p_clock_offset_accel_only(phone.load_lines(name), 1), crash_ok=True)
    assert r["crash"] is None
    bad, n = phone.garbage_vs_baseline(base["log"], r["log"])
    assert bad <= 0.01 * n, f"{bad}/{n} frames valid=True and >2 m from ARCore"


def test_global_shuffle_does_not_crash():
    r = phone.replay(phone.p_shuffle_global(phone.load_lines("B"), 1), crash_ok=True)
    assert r["crash"] is None


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#18")
def test_truncated_last_line_in_replay_and_analyze(tmp_path):
    txt = phone.raw_text("B")
    cut = txt[: len(txt) - 40]                       # last line cut mid-way
    f = tmp_path / "cut.jsonl"
    f.write_text(cut)
    import tools.live_demo.analyze as A
    A.main(str(f))                                   # must not raise JSONDecodeError
    S.replay(S.Live(), str(f), speed=1e9)


@pytest.mark.xfail(strict=True, reason="rehaanajani/pathfinder-xr#18")
def test_malformed_lines_do_not_kill_the_server():
    """One bad line from the phone (missing field / wrong type) must not stop the tracker."""
    lines = phone.load_lines("B")[:6000]
    lines.insert(3000, {"k": "g"})                   # no timestamp
    lines.insert(3001, {"k": "a", "t": 1, "v": [1, 2]})   # wrong vector length
    lines.insert(3002, {"k": "ar", "t": 5, "s": "TRACKING"})  # missing pose
    lines.insert(3003, {"k": "zz", "t": 5})          # unknown kind
    lines.insert(3004, {"k": "g", "t": "x", "v": [0, 0, 0]})
    r = phone.replay(lines, crash_ok=True)
    assert r["crash"] is None, repr(r["crash"])


@pytest.mark.xfail(strict=True, reason="measured: IMU lines buffered without bound until the first TRACKING frame (60001 samples after 60 s); low severity, STRESS_REPORT")
def test_arcore_never_tracking_does_not_grow_without_bound():
    """If ARCore never reports TRACKING the trackers never start; IMU lines must not pile up forever."""
    L = [m for m in phone.load_lines("B") if m["k"] != "ar"][:120_000]     # ~60 s of IMU
    S.OURS_SET.clear()
    live = S.Live()
    t0 = time.time()
    for m in L:
        live.line(m)
    dt = time.time() - t0
    assert len(live.g) < 5000 and len(live.a) < 5000, f"buffered {len(live.g)} gyro samples with no tracker"
    assert dt < 30, f"{dt:.0f} s to buffer 60 s of IMU data"


def test_imu_gap_is_not_bridged_silently():
    """Session A contains a 61 s hole in the IMU timestamps (Wi-Fi drop). The tracker must say it
    is unsure across it rather than hold the last sample for 61 s and keep claiming validity."""
    L = phone.load_lines("A")
    r = phone.replay(L)
    ids = np.array([x["id"] for x in r["log"]])
    g = [(i, m["t"]) for i, m in enumerate(L) if m["k"] == "g"]
    j = int(np.argmax(np.diff([t for _, t in g])))
    first_after = int(np.argmax(ids > g[j + 1][0]))
    x = r["log"][first_after]
    err = float(np.linalg.norm(x["ours"] - x["ar"]))
    assert err < 1.0 or not x["valid"], f"first frame after the 61 s IMU hole: {err:.1f} m off, valid=True"
