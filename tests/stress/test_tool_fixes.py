"""Regression tests for tool defects found in the PR #12 review (16 Nov 2026)."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T0 = 1403636579758555392


def _imu(d, n=400):
    (d / "imu0").mkdir(parents=True)
    with open(d / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(n):
            f.write(f"{T0 + i * 5_000_000},0.01,0,0,0,0,9.81\n")


def _module(folder, name):
    sys.path.insert(0, str(ROOT / folder))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def test_fallback_trainer_records_world_frame_slots(tmp_path):
    # tools/train_velocity.py trains the 7-channel world-frame fallback. With the default (version 3,
    # body-frame) weights present it must not record 10-channel slots.
    _imu(tmp_path)
    entry = _module("src", "main")
    src = (ROOT / "tools" / "train_velocity.py").read_text(encoding="utf-8")
    assert '"--model", "none", "--model-input", "world"' in src
    rec = []
    entry.run(str(tmp_path), str(tmp_path / "est.txt"),
              entry.args([str(tmp_path), "--preset", "learned", "--model", "none", "--model-input", "world",
                          "--model-window", "200", "--model-dec", "1"]), rec=rec)
    widths = {len(r[2]) for r in rec if r[2] is not None}
    assert widths == {7}


def test_eval_tables_reads_both_drift_key_names():
    tables = _module("tools", "eval_tables")
    assert tables.drift({"score_track3": {"position_drift_1s_m": 1.5}}) == 1.5
    assert tables.drift({"score_track3": {"rpe_1s_m": 2.5}}) == 2.5


def test_oracle_ceiling_uses_the_current_drift_key():
    src = (ROOT / "tools" / "oracle_ceiling.py").read_text(encoding="utf-8")
    assert '"rpe_1s_m"' not in src
    assert '"position_drift_1s_m"' in src
    assert 'res["position_drift_1s_m"]' in (ROOT / "tools" / "score_track3.py").read_text(encoding="utf-8")
