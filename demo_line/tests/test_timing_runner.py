import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("demo_timing_runner", Path(__file__).resolve().parents[1] / "tools/run_isolated_timing.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_dependency_path_is_preserved_after_demo_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/dependency-target")
    assert runner._env()["PYTHONPATH"].split(os.pathsep) == [str(runner.DEMO), "/dependency-target"]


@pytest.mark.parametrize("code,measurement", [(1, {}), (0, {"rt_factor": .5, "invariant_failures": []}),
                                             (0, {"rt_factor": 2, "invariant_failures": ["bad covariance"]})])
def test_measurement_failure_is_not_hidden_by_passing_pytest(tmp_path, monkeypatch, code, measurement):
    results = iter([SimpleNamespace(returncode=code, stdout=json.dumps(measurement), stderr=""),
                    SimpleNamespace(returncode=0, stdout="passed", stderr="")])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: next(results))
    out = tmp_path / "timing.json"
    assert runner.main(["--out", str(out)]) != 0
    assert json.loads(out.read_text())["returncode"] != 0


def test_existing_evidence_is_not_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "timing.json"
    out.write_text("original")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("must not run"))
    with pytest.raises(FileExistsError):
        runner.main(["--out", str(out)])
    assert out.read_text() == "original"
