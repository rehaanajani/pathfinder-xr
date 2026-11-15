"""Immutable, provenance-rich run bundles (issue #3 C).

Adapted from ``pathfinder/manifest.py`` in PR #5 (branch ``issue-2-real-vo``) with: full SHA-256
hashes (no truncation, no silent size limit), refusal to overwrite an existing run directory, a
status file that survives failures, and hashes of every input file the run declares.

A run directory ``results/runs/<utc>-<name>-<sha7>[-dirty]`` holds
    manifest.json   code commit, dirty tree, diff hash, interpreter/package versions, hardware,
                    command line, config, seeds, protocol version, input/model/calibration hashes
    status.json     "running" -> "complete" | "failed" (with traceback) | "interrupted"
    ...             the runner's own outputs
Nothing is ever deleted or rewritten here; promotion into committed artifacts is an explicit,
separate step (``tools/promote.py``) that records the source run id.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
SCHEMA = "pathfinder.run/1"


def sha256_file(path, chunk=1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except Exception:
        return None


def code_state() -> dict:
    diff = _git("diff", "HEAD")
    return {"commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD")}


def environment() -> dict:
    vers = {}
    for mod in ("numpy", "scipy", "cv2", "torch", "yaml", "onnxruntime", "pandas", "matplotlib"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    threads = os.environ.get("OMP_NUM_THREADS")
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "omp_num_threads": threads, "packages": vers}


class Run:
    def __init__(self, path: Path, manifest: dict):
        self.path, self.manifest = path, manifest

    def write_json(self, name, obj):
        p = self.path / name
        if p.exists():
            raise FileExistsError(f"{p} already written; run bundles are immutable")
        p.write_text(json.dumps(obj, indent=1, default=_default))
        return p

    def file(self, name) -> Path:
        p = self.path / name
        if p.exists():
            raise FileExistsError(f"{p} already exists; run bundles are immutable")
        return p

    def status(self, state, **extra):
        (self.path / "status.json").write_text(json.dumps({"state": state, "t": time.time(), **extra},
                                                          indent=1, default=_default))


def _default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


@contextmanager
def run_bundle(name: str, config: dict, inputs: dict | None = None, protocol: str | None = None,
               seeds=None, root: Path = RUNS):
    """Create a new immutable run directory and record status even if the body raises."""
    code = code_state()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sha = (code["commit"] or "nogit")[:7] + ("-dirty" if code["dirty"] else "")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    path = Path(root) / f"{stamp}-{safe}-{sha}"
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "name": name, "created_utc": stamp, "command": sys.argv,
                "code": code, "environment": environment(), "config": config, "seeds": seeds,
                "protocol": protocol,
                "inputs": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in (inputs or {}).items()}}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_default))
    run = Run(path, manifest)
    run.status("running")
    try:
        yield run
    except KeyboardInterrupt:
        run.status("interrupted")
        raise
    except BaseException as exc:
        run.status("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    else:
        run.status("complete")
