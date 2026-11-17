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


def test_compile_check_does_not_write_bytecode_and_survives_overlap(tmp_path):
    import threading

    import check_repo

    src = tmp_path / "mod.py"
    src.write_text("VALUE = 1\n\ndef f(x):\n    return x + 1\n", encoding="utf-8")
    errors = []

    def go():
        try:
            check_repo.compile_source(src)
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert list(tmp_path.rglob("*.pyc")) == []
    assert not (tmp_path / "__pycache__").exists()
    broken = tmp_path / "broken.py"
    broken.write_text("def (\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        check_repo.compile_source(broken)


def test_same_second_runs_are_distinct_and_text_is_lf(tmp_path, monkeypatch):
    from pathfinder import provenance
    monkeypatch.setattr(provenance.time, "strftime", lambda *args: "20260101T000000Z")
    with run_bundle("same", {}, root=tmp_path) as first:
        first.write_json("text.json", {"one": 1, "two": 2})
    with run_bundle("same", {}, root=tmp_path) as second:
        pass
    assert first.path != second.path
    for path in first.path.glob("*.json"):
        assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("name", [".", "./", "", "ds/file:secret", "ds\\file"])
def test_unsafe_archive_names_are_rejected_without_crashing(name):
    assert not extract_dataset._safe(name)


def test_model_manifest_cannot_escape_model_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "pathfinder.models/1", "release": {"repo": "owner/repo", "tag": "v1"},
        "models": {"bad": {"path": "models/../../outside", "asset": "model.pt", "bytes": 1, "sha256": "0" * 64}}}))
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_models, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="unsafe model"):
        fetch_models.main(["--verify-only"])


def test_failed_report_is_not_promoted(tmp_path):
    run, output = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    (run / "status.json").write_text('{"state":"failed"}')
    from tools.vision_report import main
    with pytest.raises(SystemExit, match="Only complete"):
        main([str(run), "--out", str(output)])
    assert not output.exists()
