"""Run manifest (issue #3 C): which code, settings, data and models produced a result file."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def sha256(path, limit_mb=64):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def manifest(args=None, models=(), extra=None):
    import numpy, scipy
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = None
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": " ".join(sys.argv),
        "args": vars(args) if args is not None and hasattr(args, "__dict__") else args,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": tv,
        "models": {str(m): sha256(m) for m in models},
        **(extra or {}),
    }


def write(path, **kw):
    Path(path).write_text(json.dumps(manifest(**kw), indent=1))
