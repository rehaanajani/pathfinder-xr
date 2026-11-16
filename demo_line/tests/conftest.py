"""The demo line ships its own `pathfinder` package, which differs from the one at the repository root.

`pytest demo_line/tests/...` from the repository root would otherwise import the root package, so some tests run
against the wrong filter and still pass. Here demo_line/ is put first on sys.path, and a run that mixes demo-line
tests with other paths (for example `pytest tests demo_line/tests`) is stopped, because one process can only hold
one `pathfinder`.
"""
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1]


def _inside_demo(arg, base):
    p = Path(str(arg).split("::", 1)[0])
    p = (p if p.is_absolute() else base / p).resolve()
    return p == DEMO or DEMO in p.parents


def pytest_configure(config):
    base = Path(config.invocation_params.dir)
    outside = [a for a in config.args if not _inside_demo(a, base)]
    if outside:
        pytest.exit(f"demo_line tests need demo_line/pathfinder; run them on their own, not with {outside}: "
                    "cd demo_line && python -m pytest", returncode=4)
    pf = sys.modules.get("pathfinder")
    if pf is not None and DEMO not in Path(getattr(pf, "__file__", None) or ".").resolve().parents:
        pytest.exit(f"another `pathfinder` package is already imported ({pf.__file__}); "
                    "run the demo-line tests from inside demo_line/", returncode=4)
    if str(DEMO) in sys.path:
        sys.path.remove(str(DEMO))
    sys.path.insert(0, str(DEMO))
