"""Repository checks for CI and release (issue #3 C/D).

    python tools/check_repo.py

1. every tracked Python file compiles;
2. versioned JSON artifacts carry a known ``schema`` and required fields (models manifest, VO
   calibration, promoted result bundles);
3. no tracked file contains credential-like strings (GitHub/AWS/private-key patterns);
4. no tracked file exceeds 10 MB (datasets/checkpoints belong in releases, not git); the submission video in
   docs/media/ may be up to 50 MB, because the organisers require it in the repository.
Exits non-zero with a list of problems.
"""

import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET = re.compile(rb"(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|gho_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
                    rb"|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|xox[bap]-[A-Za-z0-9-]{20,})")
SCHEMAS = {
    "pathfinder.models/1": ("release", "models"),
    "pathfinder.vo-calibration/1": ("sequence", "segment_s", "selected_vo_cov_scale", "selected_kf_every",
                                    "frontend_src_sha256", "imu_noise", "split_role"),
    "pathfinder.result-bundle/1": ("source_run", "protocol", "files"),
    "pathfinder.exposure/1": ("recordings",),
}


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


def main():
    problems = []
    files = tracked()
    for p in files:
        if not p.is_file():
            continue
        if p.suffix == ".py":
            try:
                py_compile.compile(str(p), doraise=True, cfile=None)
            except py_compile.PyCompileError as exc:
                problems.append(f"compile: {p.relative_to(ROOT)}: {exc.msg.splitlines()[-1]}")
        limit = 50 if p.relative_to(ROOT).parts[:2] == ("docs", "media") and p.suffix == ".mp4" else 10
        if p.stat().st_size > limit * 2 ** 20:
            problems.append(f"size: {p.relative_to(ROOT)} is {p.stat().st_size / 2 ** 20:.1f} MB")
        if p.suffix in (".png", ".jpg", ".mp4", ".pdf", ".pptx", ".onnx", ".pt", ".zip"):
            continue
        if SECRET.search(p.read_bytes()):
            problems.append(f"secret-like string in {p.relative_to(ROOT)}")
        if p.suffix == ".json" and p.stat().st_size < 5 * 2 ** 20:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                problems.append(f"json: {p.relative_to(ROOT)}: {exc}")
                continue
            schema = obj.get("schema") if isinstance(obj, dict) else None
            if isinstance(schema, str) and schema.startswith("pathfinder."):
                need = SCHEMAS.get(schema)
                if need is None and not schema.startswith("pathfinder.run/"):
                    problems.append(f"schema: unknown {schema} in {p.relative_to(ROOT)}")
                for k in need or ():
                    if k not in obj:
                        problems.append(f"schema: {p.relative_to(ROOT)} ({schema}) lacks {k}")
    for msg in problems:
        print(msg)
    print(f"checked {len(files)} tracked files: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
