"""Time and heap of src/main.py on a folder of degraded files.
Usage: resources.py <folder> [filter]
Per file: whole-process wall time (interpreter start, imports, solve, write) and the tracemalloc peak of the
algorithm (tracing starts after the imports, which is the heap the entry itself allocates)."""
import glob
import os
import subprocess
import sys
import time
import tracemalloc

SRC = "/mnt/e/clipping-pipeline/xr_track3/src"
sys.path.insert(0, SRC)
import main as M      # noqa: E402

folder = sys.argv[1]
flt = sys.argv[2] if len(sys.argv) > 2 else ""
out = "/tmp/res_out.txt"
worst_ms, worst_mb = 0.0, 0.0
for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    if flt not in os.path.basename(p):
        continue
    rows = sum(1 for line in open(p) if line[:1].isdigit())
    first = last = None
    for line in open(p):
        if line[:1].isdigit():
            v = int(line.split(",", 1)[0])
            first = v if first is None else first
            last = v
    dur = (last - first) * 1e-9
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, os.path.join(SRC, "main.py"), p, out], check=True, stderr=subprocess.DEVNULL)
        best = min(best, time.perf_counter() - t0)
    sys.argv = ["main.py", p, out]
    tracemalloc.start()
    M.main()
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    got = sum(1 for line in open(out) if not line.startswith("#"))
    ms = 1000 * best / dur
    worst_ms, worst_mb = max(worst_ms, ms), max(worst_mb, peak)
    print("%-22s %6d rows %6.1f s | %.2f s whole process = %5.2f ms per s | heap peak %.2f MB | poses %s" % (
        os.path.basename(p), rows, dur, best, ms, peak, "ok" if got == rows else "MISMATCH %d" % got))
print("worst: %.2f ms per s of IMU (limit 20), heap %.2f MB (limit 8)" % (worst_ms, worst_mb))
