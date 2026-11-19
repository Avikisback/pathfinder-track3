"""Runs src/main.py on the 50 inputs from the Codex review (experiments/tmp/codex) and checks, independently of
main.py's own parsing: one pose per data row (BOM-aware), each output stamp equal to its input stamp in exact decimal
arithmetic (unit from the header when it names one, else from the usual step), finite values, heap and time.
Usage: codex_cases.py [entry.py] [filter]"""
import glob
import os
import subprocess
import sys
import time
import tracemalloc
from decimal import Decimal, InvalidOperation

ENTRY = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/clipping-pipeline/xr_track3/src/main.py"
FILT = sys.argv[2] if len(sys.argv) > 2 else ""
DIR = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/codex"
OUT = "/tmp/codex_out.txt"
sys.path.insert(0, os.path.dirname(ENTRY))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("entry", ENTRY)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

UNITS = {"ns": Decimal("1e-9"), "us": Decimal("1e-6"), "ms": Decimal("1e-3"), "s": Decimal(1)}


def rows_of(p):
    head, tok = [], []
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            c = line.strip()
            if c and c[0] != "#" and (c[0].isdigit() or c[0] in "+-."):
                tok.append(c.split(",", 1)[0].strip())
            elif c and not tok:
                head.append(c)
    return head, tok


def unit_of(head, tok):
    for h in head:
        f = h.split(",", 1)[0].lower().replace("µ", "u").replace("μ", "u")
        for k, v in UNITS.items():
            if "[" + k + "]" in f or "(" + k + ")" in f:
                return v
    vals = []
    for t in tok[:4096]:
        try:
            d = Decimal(t)
            if d.is_finite():
                vals.append(d)
        except InvalidOperation:
            pass
    st = sorted(b - a for a, b in zip(vals, vals[1:]) if b > a)
    s = float(st[len(st) // 2]) if st else 1e9
    return UNITS["ns"] if s >= 1e5 else UNITS["us"] if s >= 1e2 else UNITS["ms"] if s >= 0.5 else UNITS["s"]


res = []
for p in sorted(glob.glob(DIR + "/*.csv")):
    name = os.path.basename(p)[:-4]
    if FILT not in name:
        continue
    head, tok = rows_of(p)
    u = unit_of(head, tok)
    exp, prev = [], Decimal("0")
    for t in tok:
        try:
            d = Decimal(t)
            if d.is_finite() and abs(d) < Decimal(2) ** 63:
                prev = d * u
        except InvalidOperation:
            pass
        exp.append(prev.quantize(Decimal("1e-9")))
    if os.path.exists(OUT):
        os.remove(OUT)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, ENTRY, p, OUT], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    got = [line.split() for line in open(OUT)] if os.path.exists(OUT) else []
    got = [g for g in got if g and not g[0].startswith("#")]
    bad_ts = sum(1 for e, g in zip(exp, got) if Decimal(g[0]) != e) if len(got) == len(exp) else -1
    fin = all(all(abs(float(x)) < 1e300 for x in g[1:]) and len(g) == 8 for g in got)
    sys.argv = ["main.py", p, OUT]
    tracemalloc.start()
    try:
        M.main()
    except SystemExit:
        pass
    except Exception as ex:
        print("   in-process exception", name, ex)
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    dur = float(exp[-1] - exp[0]) if len(exp) > 1 else 0.0
    ms = 1000 * wall / dur if dur > 1 else float("nan")
    ok = r.returncode == 0 and len(got) == len(exp) and bad_ts == 0 and fin and peak < 8.0
    err = (r.stderr.strip().splitlines() or [""])[-1][:70]
    print("%-4s %-22s rows %7d poses %7d stamp-mismatch %5d heap %6.2f MB  %6.2f s %8.1f ms/s %s" % (
        "ok" if ok else "FAIL", name, len(exp), len(got), bad_ts, peak, wall, ms, err), flush=True)
