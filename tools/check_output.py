#!/usr/bin/env python3
"""Check an estimated_trajectory.txt against the organisers' rules (FAQ of 17 Nov 2026).

  - exactly one pose per input data row, in the input order, nothing appended (the blind 5 s window is already in
    the input as rows with unusable values)
  - each pose's timestamp equals its row's timestamp (compared in nanoseconds; duplicated input stamps stay duplicated)
  - 8 columns: timestamp tx ty tz qx qy qz qw, all finite, unit quaternions
It also reports the largest position step between rows. This is our own checker, not the organisers' evaluator.

Usage: check_output.py <imu.csv> <estimated_trajectory.txt>
"""
import math
import sys


def to_ns(s):
    """Seconds text to integer nanoseconds without going through a float."""
    neg = s.startswith("-")
    sec, _, frac = s.lstrip("+-").partition(".")
    v = int(sec or "0") * 10**9 + int((frac + "000000000")[:9])
    return -v if neg else v


def stamp_ns(h, scale_ns):
    h = h.strip()
    if h.lstrip("+-").isdigit():
        return int(h) * scale_ns
    return to_ns(h) if scale_ns == 10**9 else round(float(h) * scale_ns)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    heads = []
    with open(sys.argv[1]) as f:
        for line in f:
            c = line.strip()
            if c and c[0] != "#" and (c[0].isdigit() or c[0] in "+-."):
                heads.append(c.split(",", 1)[0])
    # the unit of the input stamps, from the usual step (ns, us, ms or s), as main.py reads them
    nums = []
    for h in heads[:4096]:
        try:
            nums.append(float(h))
        except ValueError:
            pass
    steps = sorted(b - a for a, b in zip(nums, nums[1:]) if b > a)
    step = steps[len(steps) // 2] if steps else 1e9
    scale_ns = 1 if step >= 1e5 else (1000 if step >= 1e2 else (10**6 if step >= 0.5 else 10**9))

    errors, est, max_step, max_qerr, prev = [], [], 0.0, 0.0, None
    with open(sys.argv[2]) as f:
        for i, line in enumerate(f, 1):
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if len(c) != 8:
                errors.append(f"line {i}: {len(c)} columns, need 8")
                continue
            vals = [float(x) for x in c[1:]]
            if not all(math.isfinite(v) for v in vals):
                errors.append(f"line {i}: non-finite value")
                continue
            est.append(to_ns(c[0]))
            max_qerr = max(max_qerr, abs(math.sqrt(sum(v * v for v in vals[3:])) - 1.0))
            if prev is not None:
                max_step = max(max_step, math.dist(prev, vals[:3]))
            prev = vals[:3]

    if len(est) != len(heads):
        errors.append(f"{len(est)} poses for {len(heads)} input rows")
    wrong = 0
    for h, t in zip(heads, est):
        try:
            if stamp_ns(h, scale_ns) != t:
                wrong += 1
        except ValueError:
            pass                     # an unreadable input stamp: any value is accepted for that row
    if wrong:
        errors.append(f"{wrong} poses whose timestamp differs from their input row")
    if max_qerr > 1e-6:
        errors.append(f"quaternion norm off by {max_qerr:.2e}")

    print(f"input rows {len(heads)}, poses {len(est)}, timestamp mismatches {wrong}")
    print(f"largest position step {max_step * 1000:.2f} mm, worst quaternion norm error {max_qerr:.1e}")
    if errors:
        print("FAIL")
        for e in errors[:20]:
            print(" -", e)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
