"""Stress-test configuration.

Slow tests are opt-in so that ``pytest tests`` stays fast:
    STRESS_SLOW=1 python -m pytest -q tests/stress
Fuzz size: STRESS_FUZZ_N (default 40 in the fast tier, 600 with STRESS_SLOW=1).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(ROOT)           # ours_cfg() loads models/camera_trust.json by a relative path


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long stress test, run with STRESS_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("STRESS_SLOW"):
        return
    skip = pytest.mark.skip(reason="slow stress test: set STRESS_SLOW=1")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)
