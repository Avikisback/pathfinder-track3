#!/usr/bin/env python3
"""Check an estimated_trajectory.txt against the organisers' rules (FAQ of 17 Nov 2026).

  - exactly one pose per input data row, in the input order, nothing appended (the blind 5 s window is already in
    the input as rows with unusable values)
  - each pose's timestamp equals its row's timestamp in exact decimal arithmetic: the unit comes from the header
    (`[ns]`, `[us]`, `[ms]`, `[s]`) or else from the usual step; duplicated input stamps stay duplicated; a row whose
    stamp cannot be read is expected to repeat the stamp before it (main.py's convention)
  - 8 columns: timestamp tx ty tz qx qy qz qw, all finite, unit quaternions
It also reports the largest position step between rows. This is our own checker, not the organisers' evaluator.

Usage: check_output.py <imu.csv> <estimated_trajectory.txt>
"""
import math
import sys
from decimal import Decimal, InvalidOperation

UNITS = (("ns", Decimal("1e-9")), ("us", Decimal("1e-6")), ("ms", Decimal("1e-3")), ("s", Decimal(1)))


def is_data(c):
    return bool(c) and c[0] != "#" and (c[0].isdigit() or c[0] in "+-.")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    heads, header = [], []
    with open(sys.argv[1], encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            c = line.strip()
            if is_data(c):
                heads.append(c.split(",", 1)[0].strip())
            elif c and not heads:
                header.append(c)
    unit = None
    for h in header:
        field = h.split(",", 1)[0].lower().replace("µ", "u").replace("μ", "u")
        for name, u in UNITS:
            if "[" + name + "]" in field or "(" + name + ")" in field:
                unit = unit or u
    vals = []
    for h in heads[:4096]:
        try:
            d = Decimal(h)
            if d.is_finite():
                vals.append(d)
        except InvalidOperation:
            pass
    if unit is None:
        steps = sorted(b - a for a, b in zip(vals, vals[1:]) if b > a)
        s = float(steps[len(steps) // 2]) if steps else 1e9
        unit = UNITS[0][1] if s >= 1e5 else UNITS[1][1] if s >= 1e2 else UNITS[2][1] if s >= 0.5 else UNITS[3][1]

    errors, est, max_step, max_qerr, prev = [], [], 0.0, 0.0, None
    with open(sys.argv[2]) as f:
        for i, line in enumerate(f, 1):
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if len(c) != 8:
                errors.append(f"line {i}: {len(c)} columns, need 8")
                continue
            vals8 = [float(x) for x in c[1:]]
            if not all(math.isfinite(v) for v in vals8):
                errors.append(f"line {i}: non-finite value")
                continue
            try:
                est.append(Decimal(c[0]))
            except InvalidOperation:
                errors.append(f"line {i}: unreadable timestamp {c[0]!r}")
                continue
            max_qerr = max(max_qerr, abs(math.sqrt(sum(v * v for v in vals8[3:])) - 1.0))
            if prev is not None:
                max_step = max(max_step, math.dist(prev, vals8[:3]))
            prev = vals8[:3]

    if len(est) != len(heads):
        errors.append(f"{len(est)} poses for {len(heads)} input rows")
    wrong, bad_in, last = 0, 0, Decimal(0)
    for h, t in zip(heads, est):
        try:
            d = Decimal(h)
            if d.is_finite() and abs(d) < 2 ** 63:
                last = (d * unit).quantize(Decimal("1e-9"))
            else:
                bad_in += 1
        except InvalidOperation:
            bad_in += 1
        if t != last:
            wrong += 1
    if wrong:
        errors.append(f"{wrong} poses whose timestamp differs from their input row")
    if max_qerr > 1e-6:
        errors.append(f"quaternion norm off by {max_qerr:.2e}")

    print(f"input rows {len(heads)}, poses {len(est)}, timestamp mismatches {wrong}"
          + (f", unreadable input stamps {bad_in} (expected to repeat the one before)" if bad_in else ""))
    print(f"largest position step {max_step * 1000:.2f} mm, worst quaternion norm error {max_qerr:.1e}")
    if errors:
        print("FAIL")
        for e in errors[:20]:
            print(" -", e)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
