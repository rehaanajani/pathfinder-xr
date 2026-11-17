"""Run the demo real-time check in its own process.

Acceptance evidence for ``test_other_imu_rates`` is this script, or the ``demo-timing``
GitHub Actions job. A number taken while other tests or agents share the machine is
exploratory and is not a pass or a fail of the 1x requirement.

    cd demo_line
    python tools/run_isolated_timing.py

The script sets one BLAS thread and starts no other benchmark. It cannot see unrelated
host load; run it when the machine is otherwise idle, and say so in the written note.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
TEST = "tests/stress/test_nasty_inputs.py::test_other_imu_rates"


def _env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(DEMO), env.get("PYTHONPATH", ""))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


MEASURE = r"""
import json, time
from pathfinder.faults import Fault
from tests.stress.harness import make_case, run_pair, check_invariants
c = make_case(3, 12, 1000, 3, [Fault("camera_blackout", 5, 3)])
r = run_pair(c, check_every=25)
print(json.dumps({
    "rate_hz": 1000,
    "duration_s": 12.0,
    "wall_s": r["wall"],
    "rt_factor": r["rt_factor"],
    "max_stream_run_diff_m": r["max_diff"],
    "invariant_failures": check_invariants(r),
}))
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--junit", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-demo-timing-" + uuid.uuid4().hex[:8]
        args.out = ROOT / "results" / "runs" / run_id / "timing.json"
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite timing evidence: {args.out}")
    env = _env()
    measured = subprocess.run([sys.executable, "-c", MEASURE], cwd=DEMO, env=env, capture_output=True, text=True)
    pytest_cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", TEST]
    junit = args.junit
    if junit is not None:
        pytest_cmd.append(f"--junitxml={junit}")
    tested = subprocess.run(pytest_cmd, cwd=DEMO, env=env, capture_output=True, text=True)
    payload = {
        "schema": "pathfinder.timing-run/1",
        "test": f"demo_line/{TEST}",
        "isolated_process": True,
        "blas_threads": 1,
        "returncode": int(tested.returncode),
        "measure_returncode": int(measured.returncode),
        "claim": "acceptance evidence only for this process; suite-parallel timings are exploratory",
    }
    if measured.returncode == 0 and measured.stdout.strip():
        payload["measured_1000hz"] = json.loads(measured.stdout.strip().splitlines()[-1])
    else:
        payload["measure_stderr"] = measured.stderr[-2000:]
    payload["pytest_stdout"] = tested.stdout[-2000:]
    payload["pytest_stderr"] = tested.stderr[-2000:]
    measurement = payload.get("measured_1000hz", {})
    measurement_ok = (measured.returncode == 0 and measurement.get("rt_factor", 0) > 1
                      and measurement.get("invariant_failures") == [])
    payload["returncode"] = int(tested.returncode or (0 if measurement_ok else 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(args.out)
    print(payload["pytest_stdout"])
    return payload["returncode"]


if __name__ == "__main__":
    sys.exit(main())
