"""Exercise downloader failures using a local HTTP server, never live credentials."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import urllib.request

import pytest

from tools.fetch import fetch
from tools.fetch_models import SafeRedirectHandler


@contextmanager
def server():
    class Handler(BaseHTTPRequestHandler):
        seen = []

        def log_message(self, *args):
            pass

        def do_GET(self):
            type(self).seen.append((self.path, dict(self.headers)))
            if self.path in ("/404", "/416"):
                self.send_error(int(self.path[1:]))
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/ok")
                self.end_headers()
            elif self.path == "/truncated":
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"partial")
                self.close_connection = True
            else:
                self.send_response(206 if self.path == "/bad-range" else 200)
                if self.path == "/bad-range":
                    self.send_header("Content-Range", "bytes 0-4/5")
                self.send_header("Content-Length", "5")
                self.end_headers()
                self.wfile.write(b"hello")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://localhost:{srv.server_port}", Handler.seen
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


@pytest.mark.parametrize("path", ["/404", "/416", "/truncated"])
@pytest.mark.parametrize("has_part", [False, True])
def test_failed_response_never_promotes_partial(tmp_path, path, has_part):
    dest = tmp_path / "asset"
    part = tmp_path / "asset.part"
    if has_part:
        part.write_bytes(b"stale")
    with server() as (url, seen):
        with pytest.raises(SystemExit, match="incomplete download"):
            fetch(url + path, dest, retries=2, backoff=0, log=lambda _: None)
        if path in ("/404", "/416"):
            assert len(seen) == 1
    assert not dest.exists()


def test_wrong_range_is_rejected_before_appending(tmp_path):
    part = tmp_path / "asset.part"
    part.write_bytes(b"old")
    with server() as (url, _):
        with pytest.raises(SystemExit):
            fetch(url + "/bad-range", tmp_path / "asset", retries=1, backoff=0, log=lambda _: None)
    assert part.read_bytes() == b"old"


def test_explicit_port_and_ignored_range_restart(tmp_path):
    (tmp_path / "asset.part").write_bytes(b"old")
    with server() as (url, _):
        result = fetch(url + "/ok", tmp_path / "asset", retries=1, log=lambda _: None)
    assert result["status"] == "complete"
    assert (tmp_path / "asset").read_bytes() == b"hello"


def test_cross_origin_redirect_strips_auth_and_cookie():
    with server() as (url, seen):
        opener = urllib.request.build_opener(SafeRedirectHandler())
        request = urllib.request.Request(url + "/redirect", headers={
            "Authorization": "Bearer test-only", "Cookie": "test-only=yes"})
        with opener.open(request, timeout=5) as response:
            assert response.read() == b"hello"
        assert seen[0][1].get("Authorization") == "Bearer test-only"
        assert "Authorization" not in seen[1][1]
        assert "Cookie" not in seen[1][1]
