"""Safe extraction and verification of a dataset tar archive (issue #3 B/C).

    python tools/extract_dataset.py ARCHIVE DEST_DIR            # extract regular files only
    python tools/extract_dataset.py ARCHIVE DEST_DIR --verify   # compare an existing extraction

Only regular files and directories are written. Absolute paths, ``..`` components, device files and
every symlink/hardlink are refused and logged (TUM-VI archives contain ``dso/cam*/images`` symlinks;
the loader reads ``mav0/cam*/data`` regular files instead). ``--verify`` re-reads the archive and
checks that every regular member exists with the same size and SHA-256; it writes nothing.
A JSON report goes to stdout and ``DEST_DIR/../<archive>.extract.json``.
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath


def _safe(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and not name.startswith(("/", "\\")) and ":" not in p.parts[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("dest")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    dest = Path(a.dest).resolve()
    rep = {"archive": str(a.archive), "dest": str(dest), "mode": "verify" if a.verify else "extract",
           "regular": 0, "dirs": 0, "skipped_links": [], "refused": [], "mismatch": [], "missing": [], "bytes": 0}
    with tarfile.open(a.archive, "r:*") as tf:
        for m in tf:
            if not _safe(m.name):
                rep["refused"].append(m.name)
                continue
            target = (dest / m.name).resolve()
            if dest not in target.parents and target != dest:
                rep["refused"].append(m.name)
                continue
            if m.issym() or m.islnk():
                rep["skipped_links"].append(f"{m.name} -> {m.linkname}")
                continue
            if m.isdir():
                rep["dirs"] += 1
                if not a.verify:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                rep["refused"].append(m.name)
                continue
            rep["regular"] += 1
            rep["bytes"] += m.size
            src = tf.extractfile(m)
            h = hashlib.sha256()
            if a.verify:
                if not target.is_file() or target.is_symlink():
                    rep["missing"].append(m.name)
                    continue
                if target.stat().st_size != m.size:
                    rep["mismatch"].append(m.name)
                    continue
                for block in iter(lambda: src.read(1 << 20), b""):
                    h.update(block)
                h2 = hashlib.sha256(target.read_bytes()).hexdigest()
                if h.hexdigest() != h2:
                    rep["mismatch"].append(m.name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                with open(tmp, "wb") as fh:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        fh.write(block)
                tmp.replace(target)
    rep["n_skipped_links"] = len(rep["skipped_links"])
    rep["skipped_links"] = rep["skipped_links"][:20]
    rep["ok"] = not (rep["refused"] or rep["mismatch"] or rep["missing"])
    text = json.dumps(rep, indent=1)
    print(text)
    (dest.parent / (Path(a.archive).name + (".verify.json" if a.verify else ".extract.json"))).write_text(text)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
