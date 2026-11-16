"""Legacy evaluation must never overwrite a previous result directory."""
import subprocess
import sys
from pathlib import Path


def test_existing_output_is_refused_before_loading_models(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text('{"previous": true}', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "evaluate.py"), "--synthetic", "--out", str(tmp_path)],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == '{"previous": true}'
