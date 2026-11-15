"""Fetch only the IMU + ground-truth CSVs of EuRoC MAV (no images), via HTTP range requests.

    python tools/fetch_euroc.py [--out local_data/euroc]

EuRoC (Burri et al., IJRR 2016) is hosted on the ETH Research Collection (doi 10.3929/ethz-b-000690084) as large
zips of the full sequences (~25 GB with images). Range requests let us read just the zip directories and the
two small CSVs per sequence (a few MB in total). Licence: see the dataset page (research use, cite the paper).
"""
import argparse
import time
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/130 Safari/537.36"
BASE = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
PARTS = {"machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
         "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
         "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44"}
WANT = ("mav0/imu0/data.csv", "mav0/state_groundtruth_estimate0/data.csv", "mav0/imu0/sensor.yaml")


def _get(req, tries=8):
    """GET with backoff: the repository answers 429 when hit with many requests."""
    import time
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            wait = 30 * (k + 1)
            print(f"429 from server, waiting {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)


class HTTPRange(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (with a small read-ahead cache)."""

    def __init__(self, url, base=0, size=None):
        self.url, self.base, self.pos = url, base, 0
        if size is None:
            import time
            for k in range(8):
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        size = int(r.headers["Content-Range"].split("/")[1])
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    print(f"429 from server, waiting {30 * (k + 1)}s", file=sys.stderr, flush=True)
                    time.sleep(30 * (k + 1))
        self.size = size
        self.cache = (0, b"")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start, n):
        end = min(self.size, start + n) - 1
        req = urllib.request.Request(self.url, headers={"User-Agent": UA,
                                                        "Range": f"bytes={self.base + start}-{self.base + end}"})
        return _get(req)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, cb = self.cache
        if not (c0 <= self.pos and self.pos + n <= c0 + len(cb)):
            want = max(n, 4 * 1024 * 1024)
            self.cache = (self.pos, self._fetch(self.pos, want))
            c0, cb = self.cache
        out = cb[self.pos - c0:self.pos - c0 + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def inner_offset(zf, info):
    """Byte offset of a member's data inside the outer file (for stored nested zips)."""
    f = zf.fp
    f.seek(info.header_offset)
    h = f.read(30)
    n, m = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
    return info.header_offset + 30 + n + m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local_data/euroc")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    for part, bid in PARTS.items():
        url = BASE.format(bid)
        outer = zipfile.ZipFile(HTTPRange(url))
        names = outer.namelist()
        if a.list:
            print(part, [(i.filename, i.compress_type, round(i.file_size / 1e9, 2)) for i in outer.infolist()][:20])
            continue
        inner = [i for i in outer.infolist() if i.filename.endswith(".zip")]
        direct = [n for n in names if n.endswith(WANT)]
        for n in direct:                                         # CSVs directly in the outer zip
            dst = out / Path(n).relative_to(Path(n).parts[0]) if len(Path(n).parts) > 4 else out / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(outer.read(n))
            print("got", dst, flush=True)
        for info in inner:                                       # sequences packed as nested zips
            seq = Path(info.filename).stem
            if (out / seq / WANT[0]).exists() and (out / seq / WANT[1]).exists():
                print("have", seq, flush=True)
                continue
            for k in range(5):                                   # dropped connections: retry this sequence
                try:
                    stream_member(url, outer, info, out / seq)
                    break
                except Exception as e:                           # noqa: BLE001
                    print(f"{seq}: {type(e).__name__}, retry {k + 1}", file=sys.stderr, flush=True)
                    time.sleep(20 * (k + 1))


def stream_member(url, outer, info, dst_root):
    """Stream one deflated member (a nested sequence zip) once, keeping only the wanted CSVs."""
    import zlib
    from stream_unzip import stream_unzip
    start = inner_offset(outer, info)
    end = start + info.compress_size - 1
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
    import time
    for k in range(8):
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            print(f"429 from server, waiting {30 * (k + 1)}s", file=sys.stderr, flush=True)
            time.sleep(30 * (k + 1))

    def chunks():
        d = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        with resp as r:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                yield d.decompress(c) if d else c
            if d:
                yield d.flush()

    got = 0
    for name, size, data in stream_unzip(chunks()):
        name = name.decode()
        if name.endswith(WANT):
            rel = name[name.index("mav0/"):]
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"".join(data))
            print("got", dst, flush=True)
            got += 1
            if got == len(WANT):
                return                                      # everything we need: stop downloading
        else:
            for _ in data:
                pass


if __name__ == "__main__":
    main()
