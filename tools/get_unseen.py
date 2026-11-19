#!/usr/bin/env python3
"""Download the unseen-environment test data (approved by the user on 7 Nov).

TUM-VI non-room sequences: IMU from the ORB_SLAM3 repository (EuRoC format, same bytes as the official export), and ground truth
(dso/gt_imu.csv, IMU frame) read from the tail of the official exported tar with one HTTP range request, so the 1-20 GB tars are
never downloaded. Blackbird: the Air-IO authors' 16.7 MB light copy.

Usage: get_unseen.py <out_dir>
"""
import os, sys, urllib.request

OUT = sys.argv[1]
SEQS = [f"corridor{i}" for i in range(1, 6)] + [f"magistrale{i}" for i in range(1, 7)] + [f"slides{i}" for i in range(1, 4)] + [f"outdoors{i}" for i in range(1, 9)]
IMU = "https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/master/Examples/Monocular-Inertial/TUM_IMU/dataset-{}_512.txt"
TAR = "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-{}_512_16.tar"
BB = "https://github.com/Air-IO/Air-IO/releases/download/datasets/Blackbird.zip"


def fetch(url, headers=None, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout).read()


def gt_from_tail(n):
    u = TAR.format(n)
    size = int(urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=60).headers["Content-Length"])
    key = f"dataset-{n}_512_16/dso/gt_imu.csv".encode()
    for win in (8_000_000, 20_000_000, 60_000_000):
        d = fetch(u, {"Range": f"bytes={size - win}-{size - 1}"})
        i = d.find(key)
        if i < 0:
            continue
        sz = int(d[i + 124:i + 136].split(b"\0")[0].strip() or b"0", 8)
        if i + 512 + sz <= len(d):
            return d[i + 512:i + 512 + sz]
    return None


def main():
    for n in SEQS:
        d = os.path.join(OUT, f"tumvi_{n}")
        os.makedirs(os.path.join(d, "imu0"), exist_ok=True)
        f_imu, f_gt = os.path.join(d, "imu0", "data.csv"), os.path.join(d, "gt_imu.csv")
        if not os.path.exists(f_imu):
            open(f_imu, "wb").write(fetch(IMU.format(n)))
        if not os.path.exists(f_gt):
            g = gt_from_tail(n)
            if g is None:
                print(n, "ground truth not found", flush=True)
                continue
            open(f_gt, "wb").write(g)
        print(n, os.path.getsize(f_imu) // 1000, "kB imu,", os.path.getsize(f_gt) // 1000, "kB gt", flush=True)
    bd = os.path.join(OUT, "blackbird")
    os.makedirs(bd, exist_ok=True)
    z = os.path.join(bd, "Blackbird.zip")
    if not os.path.exists(z):
        open(z, "wb").write(fetch(BB))
    print("blackbird", os.path.getsize(z) // 1000, "kB", flush=True)


if __name__ == "__main__":
    main()
