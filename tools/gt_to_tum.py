#!/usr/bin/env python3
"""Convert a EuRoC-format pose CSV (ns, x, y, z, qw, qx, qy, qz, ...) to TUM format for evo.

Works for both mocap0/data.csv (TUM-VI) and state_groundtruth_estimate0/data.csv (EuRoC).
Usage: gt_to_tum.py <in.csv> <out.tum>
"""
import sys


def main(src, dst):
    n = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.strip().split(",")
            ns = int(c[0])
            x, y, z, qw, qx, qy, qz = c[1:8]
            out.write(f"{ns // 10**9}.{ns % 10**9:09d} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
            n += 1
    print(f"{dst}: {n} poses")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
