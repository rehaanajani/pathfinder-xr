"""Check that a TUM-VI stereo stream is the extracted photographic archive.

This does not prove the pictures depict the motion-capture path. It rejects
missing files, links, non-PNG payloads, 8-bit stand-ins, constant images, and
identical left/right copies. The visual front end still converts 16-bit grey
to the top 8 bits through OpenCV; that limit is recorded, not hidden.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder.provenance import run_bundle  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"


def png_info(raw: bytes) -> dict:
    if len(raw) < 26 or raw[:8] != PNG or raw[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height, bit, color = struct.unpack(">IIBB", raw[16:26])
    return {"width": width, "height": height, "bit_depth": bit, "color_type": color,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def assess_png(raw: bytes) -> dict:
    info = png_info(raw)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV could not decode the PNG")
    flat = img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1)
    info.update({"dtype": str(img.dtype), "std": float(np.std(img)),
                 "unique": int(min(np.unique(flat[:, 0]).size, 10001)),
                 "shape": list(img.shape)})
    reasons = []
    if info["bit_depth"] != 16 or info["color_type"] != 0:
        reasons.append("expected 16-bit greyscale PNG")
    if img.dtype != np.uint16 or img.ndim != 2:
        reasons.append("decoded image is not 2-D uint16")
    if info["width"] != 512 or info["height"] != 512:
        reasons.append("expected 512x512")
    if info["std"] < 5 or info["unique"] < 32:
        reasons.append("image is too flat to be a camera frame")
    info["reasons"] = reasons
    info["ok"] = not reasons
    return info


def audit_sequence(name: str, root=None, samples=5) -> dict:
    root = Path(root) if root else D.ROOT
    streams = D.load_streams(name, root)
    n = len(streams.frame_paths)
    if n < 3:
        raise ValueError(f"{name} has {n} regular stereo pairs")
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})[:samples]
    frames = []
    ok = True
    for i in picks:
        left, right = streams.frame_paths[i]
        pair = {"index": i, "t_cam": float(streams.frame_t[i])}
        raws = []
        for role, path in (("left", Path(left)), ("right", Path(right))):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"not a regular image: {path}")
            raw = path.read_bytes()
            raws.append(raw)
            info = assess_png(raw)
            info["path"] = str(path)
            pair[role] = info
            ok = ok and info["ok"]
        pair["left_right_identical"] = raws[0] == raws[1]
        if pair["left_right_identical"]:
            ok = False
        frames.append(pair)
    return {"sequence": name, "frames_regular": n, "imu_samples": int(len(streams.imu_t)),
            "index_sha256": D.index_sha256(name, root), "sampled": frames, "ok": ok,
            "frontend_conversion": "cv2.IMREAD_GRAYSCALE keeps the top 8 bits of these 16-bit PNGs",
            "not_synthetic": "samples are archive PNGs addressed by mav0/cam*/data.csv; mocap is not an image input",
            "limits": "A photographic file can still be the wrong recording. This audit checks format, "
                      "dynamic range, stereo distinctness and regular-file paths."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.data_root) if a.data_root else D.ROOT
    cam0 = D.seq_dir(a.seq, root) / "mav0/cam0/data.csv"
    with run_bundle(f"verify-images-{a.seq}", vars(a), {"cam0_index": cam0}, protocol="real-image-audit") as run:
        report = audit_sequence(a.seq, root)
        run.write_json("image_audit.json", report)
        print(a.seq, "ok" if report["ok"] else "FAILED", "frames", report["frames_regular"], run.path.name)
        return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
