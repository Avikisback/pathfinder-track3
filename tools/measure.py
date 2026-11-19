#!/usr/bin/env python3
"""Runtime and memory of src/main.py on one input, written as metadata.json for the submission bundle.

  python3 tools/measure.py <main.py> <imu csv | dataset folder | zip> <estimated_trajectory.txt> <metadata.json>

Memory, as the FAQ (17 Nov) asks: at the algorithm level. main.py is imported first (Python and NumPy loaded), then
tracemalloc records the peak heap that the estimation allocates while it reads, solves and writes. NumPy reports its
array buffers to tracemalloc, so this is the working memory of the algorithm. The process's peak resident size
(interpreter and libraries included) is recorded too, for reference.
Time: wall time of a separate whole run (`python3 main.py ...`: interpreter start, imports, read, solve, write), best of
three, per second of IMU data.
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    entry, data, out, meta = sys.argv[1:]
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, entry, data, out], check=True)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    sys.path.insert(0, os.path.dirname(os.path.abspath(entry)))
    import numpy
    import main as M
    sys.argv = ["main.py", data, out]
    tracemalloc.start()
    M.main()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    stamps = [line.split()[0] for line in open(out) if line.strip() and not line.startswith("#")]
    dur = float(stamps[-1]) - float(stamps[0]) if len(stamps) > 1 else 0.0
    m = dict(
        input=os.path.basename(data.rstrip("/")),
        imu_rows=len(stamps),
        imu_seconds=round(dur, 3),
        runtime_seconds=round(best, 3),
        runtime_ms_per_imu_second=round(1000 * best / dur, 3) if dur > 0 else None,
        peak_heap_mb=round(peak / 1e6, 3),
        peak_heap_note="tracemalloc peak of the estimation itself (after Python and NumPy are loaded)",
        peak_process_rss_mb=round(rss_kb * 1024 / 1e6, 1),
        threads=1,
        python=platform.python_version(),
        numpy=numpy.__version__,
        machine=f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
    )
    with open(meta, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
