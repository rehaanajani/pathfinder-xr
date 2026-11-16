"""Fetch only the IMU and mocap ground-truth CSVs of the TUM-VI rooms (EuRoC format, 512x16 export), no images.

    python tools/fetch_tumvi.py [--rooms 1 2 3 4 5 6] [--out local_data/tumvi]

Each room is a ~1.7 GB uncompressed tar on cdn3.vision.in.tum.de. It is streamed and only mav0/imu0/data.csv,
mav0/mocap0/data.csv and mav0/imu0/sensor.yaml are written; nothing else touches the disk.
Dataset: Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", IROS 2018 (CC BY 4.0).
Output layout matches tools/eval_track3.py: <out>/dataset-room<N>_512_16/mav0/{imu0,mocap0}/...
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room{}_512_16.tar"
WANT = ("mav0/imu0/data.csv", "mav0/imu0/sensor.yaml", "mav0/mocap0/data.csv")


def fetch(room, out):
    root = Path(out) / f"dataset-room{room}_512_16"
    got = set()
    req = urllib.request.Request(URL.format(room), headers={"User-Agent": "pathfinder-eval"})
    with urllib.request.urlopen(req) as r, tarfile.open(fileobj=r, mode="r|") as tf:
        for m in tf:
            name = m.name.split("/", 1)[-1] if m.name.startswith("dataset") else m.name
            if name in WANT:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
                got.add(name)
                print(f"room{room}: {name} {dst.stat().st_size} B", flush=True)
            if len(got) == len(WANT):
                break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default="local_data/tumvi")
    a = ap.parse_args()
    for room in a.rooms:
        got = fetch(room, a.out)
        if len(got) != len(WANT):
            print(f"room{room}: MISSING {set(WANT) - got}")


if __name__ == "__main__":
    main()
