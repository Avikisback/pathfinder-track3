"""Stress test of src/main.py on inputs the hidden test might hold: very long, very fast, tiny, malformed.
Usage: odd_inputs.py <a degraded room csv> <a degraded flight csv>
For each case: does it finish, one pose per data row, all values finite, whole-process time per second of IMU, heap peak."""
import os
import subprocess
import sys
import time
import tracemalloc

import numpy as np

SRC = "/mnt/e/clipping-pipeline/xr_track3/src"
sys.path.insert(0, SRC)
import main as M      # noqa: E402

room, flight = sys.argv[1], sys.argv[2]
W = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/odd"
os.makedirs(W, exist_ok=True)


def lines_of(p):
    head, rows = [], []
    for line in open(p):
        (rows if line[:1].isdigit() else head).append(line.rstrip("\n"))
    return head, rows


def write(name, lines, newline="\n"):
    p = os.path.join(W, name)
    with open(p, "w", newline="") as f:
        f.write(newline.join(lines) + newline)
    return p


cases = []
hr, rr = lines_of(room)
hf, rf = lines_of(flight)
ts = np.array([int(x.split(",", 1)[0]) for x in rr])
rest = [x.split(",", 1)[1] for x in rr]
span = ts[-1] - ts[0] + 5_000_000


def repeat(times):
    out = list(hr)
    for i in range(times):
        out += ["%d,%s" % (t + i * span, r) for t, r in zip(ts, rest)]
    return out


cases.append(("room as given", room))
cases.append(("flight as given", flight))
cases.append(("room x2 (4.7 min)", write("long2.csv", repeat(2))))
cases.append(("room x6 (14 min)", write("long6.csv", repeat(6))))
cases.append(("room x25 (59 min)", write("long25.csv", repeat(25))))
# a fast IMU: every row five times, 1 ms apart, with a little noise so rows are not frozen repeats
rng = np.random.default_rng(0)
fast = list(hr)
vals = np.array([[float(v) for v in r.split(",")[:6]] for r in rest])
for i in range(len(ts)):
    for j in range(5):
        v = vals[i] if not vals[i].any() else vals[i] + rng.normal(0, 1e-3, 6)
        fast.append("%d,%s" % (ts[i] + j * 1_000_000, ",".join("%.6f" % x for x in v)))
cases.append(("room at 1000 Hz", write("fast.csv", fast)))
tf = np.array([int(x.split(",", 1)[0]) for x in rf])
restf = [x.split(",", 1)[1] for x in rf]
spanf = tf[-1] - tf[0] + 5_000_000
cases.append(("flight x3 (9 min)", write("flight3.csv", list(hf) + ["%d,%s" % (t + i * spanf, r) for i in range(3) for t, r in zip(tf, restf)])))
cases.append(("header only", write("header.csv", hr)))
cases.append(("empty file", write("empty.csv", [])))
cases.append(("one row", write("one.csv", hr + rr[:1])))
cases.append(("100 rows", write("short.csv", hr + rr[:100])))
cases.append(("10 s", write("ten.csv", hr + rr[:2000])))
cases.append(("no header", write("nohead.csv", rr)))
cases.append(("Windows line ends", write("crlf.csv", hr + rr, "\r\n")))
cases.append(("blank lines and spaces", write("blank.csv", hr + [x for r in rr[:8000] for x in (" " + r.replace(",", " , "), "")] + rr[8000:])))
bad = list(rr)
for i in range(3000, 3400):
    bad[i] = bad[i].split(",", 1)[0] + ",nan,nan,nan,nan,nan,nan"
for i in range(9000, 9050):
    bad[i] = bad[i].split(",", 1)[0] + ",,,x,,,"
for i in range(12000, 12010):
    bad[i] = "garbage"
bad[15000] = bad[15000].split(",", 1)[0] + ",inf,0,0,1e30,0,0"
cases.append(("NaN, empty and garbage rows", write("bad.csv", hr + bad)))
cases.append(("seconds as decimals", write("secs.csv", hr + ["%.9f,%s" % (t * 1e-9, r) for t, r in zip(ts, rest)])))
cases.append(("microseconds", write("micro.csv", hr + ["%d,%s" % (t // 1000, r) for t, r in zip(ts, rest)])))
cases.append(("extra columns", write("extra.csv", hr + [r + ",1.0,2.0" for r in rr])))
cases.append(("all zeros", write("zeros.csv", hr + ["%d,0,0,0,0,0,0" % t for t in ts])))
back = list(rr)
back[5000:5200] = back[5000:5200][::-1]
cases.append(("timestamps going backwards", write("back.csv", hr + back)))
cases.append(("half the file missing", write("half.csv", hr + rr[:6000] + rr[len(rr) // 2 + 6000:])))

out = os.path.join(W, "out.txt")
fails = 0
for name, p in cases:
    rows = [line for line in open(p) if M.is_data(line.strip())]
    dur = 0.0
    try:
        a, b = float(rows[0].split(",")[0]), float(rows[-1].split(",")[0])
        dur = (b - a) * M.unit(abs(b - a) / max(1, len(rows) - 1)) if len(rows) > 1 else 0.0
    except (ValueError, IndexError):
        pass
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, os.path.join(SRC, "main.py"), p, out], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    note = r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else ""
    got = [line.split() for line in open(out) if not line.startswith("#")] if os.path.exists(out) else []
    fine = r.returncode == 0 and len(got) == len(rows) and all(len(g) == 8 for g in got)
    if fine and got:
        fine = bool(np.isfinite(np.array(got, float)).all())
    sys.argv = ["main.py", p, out]
    tracemalloc.start()
    try:
        M.main()
    except SystemExit:
        pass
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    ms = 1000 * wall / dur if dur > 1 else float("nan")
    ok = fine and peak < 8.0 and not (ms > 20.0)
    fails += not ok
    print("%-4s %-28s %7d rows %7.1f s | %5.2f s = %5.1f ms per s | heap %.2f MB | poses %d %s" % (
        "ok" if ok else "FAIL", name, len(rows), dur, wall, ms, peak, len(got), note))
print("failures:", fails)
