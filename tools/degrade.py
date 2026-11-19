#!/usr/bin/env python3
"""Make a degraded copy of an IMU CSV, so we can test before the organisers' script arrives.

This schedule is OUR GUESS at what the organisers do, not their degradation script. Replace it
when theirs is released. Noise and bias grow over the file, 2 s gaps are cut out, and the last
5 s are removed to stand in for IMU failure.

Usage: degrade.py <in.csv> <out.csv> [--seed N] [--gap-mode delete|zero]
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gaps", type=int, default=3)
    ap.add_argument("--gap-mode", choices=["delete", "zero"], default="delete")
    ap.add_argument("--fail-seconds", type=float, default=5.0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    header, rows = [], []
    with open(a.src) as f:
        for line in f:
            if line[0] == "#":
                header.append(line)
            elif line.strip():
                c = line.strip().split(",")
                rows.append((int(c[0]), [float(x) for x in c[1:7]]))

    t0, t1 = rows[0][0], rows[-1][0] - int(a.fail_seconds * 1e9)
    span = t1 - t0
    # Gaps sit in the later 70% of the file, spaced so that they do not overlap.
    starts = sorted(t0 + int(span * (0.3 + 0.7 * (i + rnd.random() * 0.8) / a.gaps)) for i in range(a.gaps))
    # Random constant bias directions, ramping up with progress through the file.
    gyro_dir = [rnd.uniform(-1, 1) for _ in range(3)]
    acc_dir = [rnd.uniform(-1, 1) for _ in range(3)]

    kept = 0
    with open(a.dst, "w") as out:
        out.writelines(header)
        for t, v in rows:
            if t > t1:
                break
            in_gap = any(s <= t < s + 2_000_000_000 for s in starts)
            if in_gap and a.gap_mode == "delete":
                continue
            if in_gap:
                out.write(f"{t},0,0,0,0,0,0\n")
                kept += 1
                continue
            x = (t - t0) / span  # progress 0..1
            sg, sa = (0.002 + 0.008 * x), (0.02 + 0.08 * x)  # noise grows 5x
            w = [v[i] + rnd.gauss(0, sg) + 0.02 * x * gyro_dir[i] for i in range(3)]
            acc = [v[i + 3] + rnd.gauss(0, sa) + 0.2 * x * acc_dir[i] for i in range(3)]
            out.write(f"{t}," + ",".join(f"{q:.6f}" for q in w + acc) + "\n")
            kept += 1
    print(f"{a.dst}: kept {kept} of {len(rows)} rows, gaps at "
          + ", ".join(f"{(s - t0) * 1e-9:.1f}s" for s in starts))


if __name__ == "__main__":
    main()
