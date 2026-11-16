"""Deterministic smoke evaluation on the generated synthetic fixture (issue #3 D).

    python tools/smoke_eval.py --out ci-runs

Runs clean and a 1 s image blackout through vo_only, imu_only, conventional and pathfinder into a new
immutable run directory, then applies loose regression gates (finite filter outputs and error bounds).
Outage and validity flags are checked separately by tests/test_smoke_pipeline.py. Synthetic plumbing evidence only: never quote these numbers as accuracy.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pathfinder import data_tumvi_raw as D, scoring  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.provenance import run_bundle  # noqa: E402
from pathfinder.smoke import make_fixture  # noqa: E402
from pathfinder.vision_eval import Frozen, imu_noise_from_config, run_system, system_configs  # noqa: E402
from pathfinder.vision_frontend import run_frontend  # noqa: E402

GATES = {("clean", "pathfinder"): 0.05, ("blackout", "pathfinder"): 0.20, ("clean", "conventional"): 0.10,
         ("clean", "vo_only"): 0.10}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    kw = {"root": Path(a.out)} if a.out else {}
    failures = []
    with tempfile.TemporaryDirectory() as tmp, run_bundle("smoke-synthetic", vars(a), protocol="smoke", seeds=[0],
                                                          **kw) as run:
        make_fixture(tmp, duration=6.0)
        s = D.load_streams("smoke", Path(tmp))
        ref = D.load_reference("smoke", Path(tmp))
        rig = RigCalibration.from_kalibr(s.calib_path)
        fz = Frozen(imu_noise=imu_noise_from_config(s.dir / "dso/imu_config.yaml"))
        cfgs = system_configs(fz)
        rows = []
        for scn, sched in (("clean", []), ("blackout", [{"kind": "blackout", "start": 3.5, "end": 4.5, "severity": 1.0}])):
            fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], schedule=sched, cache=False)
            for system in ("vo_only", "imu_only", "conventional", "pathfinder"):
                cfg, cam, mdl = (None, True, False) if system == "vo_only" else cfgs[system]
                out, diag = run_system(system, cfg, cam, mdl, s.imu_t, s.gyro, s.acc, fe["events"])
                res, _ = scoring.score(out, ref, [{**f, "kind": "image:" + f["kind"]} for f in fe["schedule"]])
                ate = res.get("start_se3", {}).get("ate_all", {}).get("rmse")
                started = out.get("track_state", np.ones(len(out["t"]))) != 0
                finite = bool(np.isfinite(out["p"][started]).all())
                rows.append({"scenario": scn, "system": system, "ate_rmse_m": ate, "finite": finite,
                             "valid_fraction": res.get("valid_fraction"), "frontend": fe["stats"]})
                gate = GATES.get((scn, system))
                if system != "vo_only" and not finite:
                    failures.append(f"{scn}/{system}: non-finite output")
                if gate is not None and (ate is None or ate > gate):
                    failures.append(f"{scn}/{system}: ATE {ate} > gate {gate}")
                print(f"{scn:9s} {system:13s} ATE {ate if ate is None else round(ate, 4)}")
        run.write_json("smoke.json", {"label": "synthetic smoke fixture; not accuracy evidence", "rows": rows,
                                      "gates": {f"{k[0]}/{k[1]}": v for k, v in GATES.items()}, "failures": failures})
        print("run:", run.path)
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
