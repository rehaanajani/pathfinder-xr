"""Download and verify the released model checkpoints (issue #3 B).

    python tools/fetch_models.py [--only dispnet_faults] [--include-optional] [--verify-only]

Authenticated access without exposing credentials: uses the GitHub CLI (``gh release download``,
which reads its own keyring/GH_TOKEN) when available, otherwise HTTPS with a token taken from
``GH_TOKEN``/``GITHUB_TOKEN`` in an Authorization header. Tokens are never printed, logged or written.
Every file is written to a temporary name, checked against ``models/MODELS.json`` (size and SHA-256)
and only then moved into place. Any mismatch or download failure exits non-zero. There is no fallback
to training: retraining is an explicit, separate ``train.py`` run that produces a new model identity.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "MODELS.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, spec: dict) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size != spec["bytes"]:
        return f"size {path.stat().st_size} != {spec['bytes']}"
    got = sha256(path)
    return None if got == spec["sha256"] else f"sha256 {got} != {spec['sha256']}"


def _download_gh(repo, tag, asset, dest_dir) -> Path:
    subprocess.run(["gh", "release", "download", tag, "--repo", repo, "--pattern", asset, "--dir", str(dest_dir),
                    "--clobber"], check=True, capture_output=True, timeout=1800)
    return Path(dest_dir) / asset


def _download_https(repo, tag, asset, dest_dir) -> Path:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/octet-stream", "User-Agent": "pathfinder-fetch-models"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(urllib.request.Request(api, headers={**headers, "Accept": "application/vnd.github+json"}),
                                timeout=30) as r:
        rel = json.load(r)
    match = [a for a in rel.get("assets", []) if a["name"] == asset]
    if not match:
        raise FileNotFoundError(f"asset {asset} not in release {tag}")
    out = Path(dest_dir) / asset
    with urllib.request.urlopen(urllib.request.Request(match[0]["url"], headers=headers), timeout=60) as r, \
            open(out, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", help="model key(s) from MODELS.json")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="check local files; download nothing")
    a = ap.parse_args(argv)
    man = json.loads(MANIFEST.read_text())
    if man.get("schema") != "pathfinder.models/1":
        sys.exit("models manifest schema mismatch")
    repo, tag = man["release"]["repo"], man["release"]["tag"]
    keys = a.only or [k for k, v in man["models"].items() if a.include_optional or not v.get("optional")]
    failures = 0
    for key in keys:
        spec = man["models"][key]
        dest = ROOT / spec["path"]
        problem = verify(dest, spec)
        if problem is None:
            print(f"ok       {key}: {spec['path']}")
            continue
        if a.verify_only:
            print(f"FAILED   {key}: {problem}")
            failures += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                got = (_download_gh if shutil.which("gh") else _download_https)(repo, tag, spec["asset"], tmp)
            except Exception as exc:                      # message only; never the request headers
                print(f"FAILED   {key}: download error {type(exc).__name__}")
                failures += 1
                continue
            problem = verify(got, spec)
            if problem:
                print(f"FAILED   {key}: downloaded file rejected ({problem})")
                failures += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), dest)
            print(f"fetched  {key}: {spec['path']}")
    if failures:
        sys.exit(f"{failures} model(s) missing or invalid; no retraining is attempted")


if __name__ == "__main__":
    main()
