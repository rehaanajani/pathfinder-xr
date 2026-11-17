"""Score an established external VIO trajectory as its own baseline.

The external system reads the real images and the real IMU itself. Its poses are
not a measurement for Path Finder: fusing them would count that IMU twice, once
inside the external estimator and again in this filter. This module imports
scoring and the reference loader only. It has no fusion entry point.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder import scoring  # noqa: E402
from pathfinder.provenance import run_bundle, sha256_file  # noqa: E402


def refuse_fusion(*_args, **_kwargs):
    """Kept so callers and tests have one explicit rejection."""
    raise RuntimeError("external VIO poses are not a filter input; the IMU would be counted twice")


def load_pose_table(path, quat_order="wxyz", time_unit="auto"):
    """Whitespace or comma separated poses: t, tx, ty, tz, and a quaternion.

    ``quat_order`` is ``wxyz`` (EuRoC / TUM) or ``xyzw`` (scipy). ``time_unit``
    is ``s``, ``ns``, or ``auto`` (values above 1e14 are nanoseconds).
    """
    if quat_order not in ("wxyz", "xyzw"):
        raise ValueError("quat_order must be wxyz or xyzw")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 8:
            raise ValueError(f"pose row needs 8 fields, got {len(parts)}")
        rows.append(parts[:8])
    if not rows:
        raise ValueError("trajectory is empty")
    stamps = [int(r[0]) if _integer_token(r[0]) else float(r[0]) for r in rows]
    if time_unit == "auto":
        time_unit = "ns" if float(np.median(np.asarray(stamps, float))) > 1e14 else "s"
    if time_unit == "ns":
        t = np.array([int(v) * 1e-9 if _integer_token(str(v)) else float(v) * 1e-9 for v in stamps], float)
    elif time_unit == "s":
        t = np.asarray(stamps, float)
    else:
        raise ValueError("time_unit must be s, ns, or auto")
    pose = np.array([[float(v) for v in r[1:]] for r in rows], float)
    if np.any(np.diff(t) <= 0) or not np.isfinite(t).all() or not np.isfinite(pose).all():
        raise ValueError("trajectory timestamps must be finite and strictly increasing")
    q = pose[:, 3:7]
    if quat_order == "wxyz":
        q = q[:, [1, 2, 3, 0]]
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm <= 0):
        raise ValueError("trajectory quaternions must be nonzero")
    return t, pose[:, :3], q / norm[:, None], time_unit


def _integer_token(token: str) -> bool:
    token = token[1:] if token.startswith("-") else token
    return token.isdigit()


def as_estimator_output(t, p, q):
    """Poses only. Validity stays false: this covariance is not the filter's."""
    n = len(t)
    return {"t": np.asarray(t, float), "p": np.asarray(p, float), "q": np.asarray(q, float),
            "valid": np.zeros(n, bool), "tracked": np.ones(n, bool),
            "P_pos": np.full((n, 3, 3), np.nan)}


def score_output(out, ref, faults=()):
    return scoring.score(out, ref, faults)


def score_file(seq, trajectory, data_root=None, quat_order="wxyz", time_unit="auto", faults=()):
    root = Path(data_root) if data_root else D.ROOT
    t, p, q, unit = load_pose_table(trajectory, quat_order, time_unit)
    ref = D.load_reference(seq, root)
    res, series = score_output(as_estimator_output(t, p, q), ref, faults)
    return res, series, unit


def module_imports():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--quat-order", default="wxyz", choices=("wxyz", "xyzw"))
    ap.add_argument("--time-unit", default="auto", choices=("auto", "s", "ns"))
    ap.add_argument("--system", required=True, help="external system name, not a Path Finder config")
    ap.add_argument("--pinned-commit", required=True)
    ap.add_argument("--config", required=True, help="external config file used for the run")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    if a.system.startswith("pathfinder"):
        raise SystemExit("refusing to label an external trajectory as a Path Finder system")
    traj = Path(a.trajectory)
    cfg = Path(a.config)
    inputs = {"trajectory": traj, "external_config": cfg,
              "reference_scoring_only": D.seq_dir(a.seq, Path(a.data_root) if a.data_root else D.ROOT) / "mav0/mocap0/data.csv"}
    with run_bundle(f"external-vio-{a.system}-{a.seq}", vars(a), inputs, protocol="external-vio-baseline") as run:
        res, _series, unit = score_file(a.seq, traj, a.data_root, a.quat_order, a.time_unit)
        payload = {"system": a.system, "sequence": a.seq, "pinned_commit": a.pinned_commit,
                   "quat_order": a.quat_order, "time_unit_resolved": unit,
                   "trajectory_sha256": sha256_file(traj), "config_sha256": sha256_file(cfg),
                   "fused_into_pathfinder": False, "vo_covariance_validated": False,
                   "notes": a.notes, "metrics": res}
        run.write_json("external_vio.json", payload)
        print(a.system, a.seq, "ate", (res.get("start_se3") or {}).get("ate_all"), "run", run.path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
