"""Release/provenance tooling: verified model fetch, safe extraction, immutable run bundles (issue #3)."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import extract_dataset  # noqa: E402
import fetch_models  # noqa: E402

from pathfinder.provenance import run_bundle  # noqa: E402


def test_model_verification_rejects_size_and_hash_mismatch(tmp_path):
    p = tmp_path / "m.pt"
    p.write_bytes(b"abc")
    spec = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert fetch_models.verify(p, spec) is None
    assert "sha256" in fetch_models.verify(p, {**spec, "sha256": "0" * 64})
    assert "size" in fetch_models.verify(p, {**spec, "bytes": 4})
    assert fetch_models.verify(tmp_path / "missing.pt", spec) == "missing"


def test_models_manifest_matches_release_schema():
    man = json.loads((ROOT / "models" / "MODELS.json").read_text())
    assert man["schema"] == "pathfinder.models/1"
    for spec in man["models"].values():
        assert len(spec["sha256"]) == 64 and spec["bytes"] > 0 and spec["path"].startswith("models/")


def _tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, kind, data in members:
            ti = tarfile.TarInfo(name)
            if kind == "file":
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            elif kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, data
                tf.addfile(ti)


def test_extraction_refuses_traversal_and_links_and_verify_detects_tampering(tmp_path):
    arc = tmp_path / "a.tar"
    _tar(arc, [("ds/mav0/x.txt", "file", b"hello"), ("../evil.txt", "file", b"x"),
               ("ds/dso/images", "sym", "../../mav0"), ("/abs.txt", "file", b"y")])
    dest = tmp_path / "out"
    dest.mkdir()
    rc = extract_dataset.main([str(arc), str(dest)])
    assert rc == 1                                           # refused members make the run fail visibly
    assert (dest / "ds/mav0/x.txt").read_bytes() == b"hello"
    assert not (tmp_path / "evil.txt").exists() and not (dest / "ds/dso/images").exists()
    rep = json.loads((tmp_path / "a.tar.extract.json").read_text())
    assert rep["n_skipped_links"] == 1 and len(rep["refused"]) == 2
    good = tmp_path / "g.tar"
    _tar(good, [("ds/x.txt", "file", b"hello")])
    out2 = tmp_path / "o2"
    out2.mkdir()
    assert extract_dataset.main([str(good), str(out2)]) == 0
    assert extract_dataset.main([str(good), str(out2), "--verify"]) == 0
    (out2 / "ds/x.txt").write_bytes(b"hellO")
    assert extract_dataset.main([str(good), str(out2), "--verify"]) == 1


def test_run_bundles_are_immutable_and_record_failures(tmp_path):
    with run_bundle("t", {"a": 1}, root=tmp_path) as run:
        run.write_json("x.json", {"v": 1})
        with pytest.raises(FileExistsError):
            run.write_json("x.json", {"v": 2})
    assert json.loads((run.path / "status.json").read_text())["state"] == "complete"
    man = json.loads((run.path / "manifest.json").read_text())
    assert man["schema"] == "pathfinder.run/1" and "commit" in man["code"] and "packages" in man["environment"]
    with pytest.raises(RuntimeError):
        with run_bundle("f", {}, root=tmp_path) as bad:
            raise RuntimeError("boom")
    st = json.loads((bad.path / "status.json").read_text())
    assert st["state"] == "failed" and "boom" in st["error"]


def test_repo_check_passes():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_repo.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
