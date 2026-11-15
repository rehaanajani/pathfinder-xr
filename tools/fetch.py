"""Resumable, checksummed download of large external assets (datasets, release models).

    python tools/fetch.py URL DEST [--sha256 HEX] [--connect-timeout 10] [--read-timeout 30]

* HTTP Range resume from ``DEST.part``; short connect/read timeouts and bounded retries, so a
  stalled server fails visibly instead of hanging.
* The final file only appears (atomic rename) after the size matches Content-Length and, if
  given, the SHA-256 matches. A mismatch keeps the ``.part`` file and exits non-zero.
* ``DEST.fetch.json`` records URL, size, SHA-256, start/end time and every attempt's outcome.
* Tokens are never accepted on the command line or written to logs. Private GitHub release
  assets are fetched with ``gh release download`` (see docs/REPRODUCE.md), not here.
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _total_from(resp, offset: int):
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1]
        return int(tail) if tail.isdigit() else None
    cl = resp.headers.get("Content-Length")
    return int(cl) + offset if cl and cl.isdigit() else None


def fetch(url: str, dest: Path, sha256: str | None = None, connect_timeout: float = 10.0,
          read_timeout: float = 30.0, retries: int = 50, backoff: float = 5.0, log=print) -> dict:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    record = {"url": url, "dest": str(dest), "started": time.time(), "attempts": []}
    meta = dest.with_name(dest.name + ".fetch.json")
    if dest.exists():
        digest = sha256_file(dest)
        if sha256 and digest != sha256.lower():
            raise SystemExit(f"{dest} exists but SHA-256 {digest} != expected {sha256}")
        record.update(status="already_present", size=dest.stat().st_size, sha256=digest)
        meta.write_text(json.dumps(record, indent=1))
        return record
    total = None
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if total is not None and offset >= total:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "pathfinder-fetch/1",
                                                   **({"Range": f"bytes={offset}-"} if offset else {})})
        entry = {"attempt": attempt, "offset": offset, "t": time.time()}
        try:
            host = urllib.parse.urlsplit(url).hostname
            socket.create_connection((host, 443 if url.startswith("https") else 80), connect_timeout).close()
            with urllib.request.urlopen(req, timeout=read_timeout) as resp:
                if offset and resp.status != 206:
                    offset = 0                       # server ignored Range: restart cleanly
                    part.unlink(missing_ok=True)
                total = _total_from(resp, offset)
                with open(part, "ab") as fh:
                    last = time.time()
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        if time.time() - last > 15:
                            last = time.time()
                            log(f"{dest.name}: {fh.tell() / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB")
            entry["outcome"] = "ok"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            entry["outcome"] = f"{type(exc).__name__}: {exc}"
            log(f"attempt {attempt}: {entry['outcome']}")
            record["attempts"].append(entry)
            meta.write_text(json.dumps(record, indent=1))
            time.sleep(backoff)
            continue
        record["attempts"].append(entry)
        size = part.stat().st_size
        if total is None or size >= total:
            break
    size = part.stat().st_size if part.exists() else 0
    if total is not None and size != total:
        record.update(status="incomplete", size=size, expected_size=total)
        meta.write_text(json.dumps(record, indent=1))
        raise SystemExit(f"incomplete download {size}/{total} bytes; rerun to resume")
    digest = sha256_file(part)
    record.update(size=size, sha256=digest, finished=time.time())
    if sha256 and digest != sha256.lower():
        record["status"] = "checksum_mismatch"
        meta.write_text(json.dumps(record, indent=1))
        raise SystemExit(f"SHA-256 mismatch: {digest} != {sha256}")
    os.replace(part, dest)
    record["status"] = "complete"
    meta.write_text(json.dumps(record, indent=1))
    log(f"{dest}: {size} bytes sha256={digest}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--sha256")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--read-timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=50)
    a = ap.parse_args(argv)
    fetch(a.url, a.dest, a.sha256, a.connect_timeout, a.read_timeout, a.retries,
          log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    sys.exit(main())
