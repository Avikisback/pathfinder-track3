#!/usr/bin/env python3
"""Path Finder, Samsung XR Hackathon 2026, Track 3. IMU-only 6-DoF trajectory with a blind forecast.

    python3 main.py <imu csv | dataset folder> [estimated_trajectory.txt]

Standalone: Python 3 and NumPy only. One thread, offline, no network, no learned weights.
Input : EuRoC-format IMU CSV (timestamp [ns], gyro xyz [rad/s], accel xyz [m/s^2]); the two sensor triplets are told
        apart by which one carries gravity, so either column order works.
Output: one line per input row, `timestamp tx ty tz qx qy qz qw`, timestamp in seconds. Nothing is appended.

Method (the whole file is read before any pose is written, so slow sensor errors are solved for, not filtered):
  1. Rows are put on a regular time grid. Spikes are removed and short holes filled; longer holes stay dropouts.
  2. Rotor vibration in the accelerometer tells a drone from a handheld device.
  3. Gyro calibration: a rest period in the file if there is one; a drone without one keeps the declared calibration
     only if a bias estimate from the file itself agrees with it.
  4. One least-squares solve over the file finds gyro and accel bias curves, a tilt per stretch between dropouts and a
     velocity step per dropout, such that the 5 s average velocity and the position stay small (drones: such that the
     velocity agrees with rotor drag). How fast the bias may change is chosen per file by Bayesian evidence.
  5. A self-check, then a gain per octave band of velocity.
  6. Trailing dead rows (the blind window): hold plus a damped share of the last velocity.
The constants below were fitted on training data only (EuRoC flights, TUM-VI rooms 1-5). None names a sequence.

Limits (organisers' FAQ, 17 Nov 2026): 8 MB at the heap level, 20 ms per second of IMU, one thread. Memory stays
bounded whatever the file: rows are worked through in pieces, poses are written as they are made and timestamps are
re-read at write time; a file faster or longer than ROW_BUDGET rows is averaged into cells as it is read; the one
solve has size caps (KNOTS_MAX, RES_MAX, GAPS_MAX) that only bind on files longer than about 3 minutes. Whatever
happens inside, the output has exactly one finite pose per input row.
"""
import os
import sys
from itertools import islice

# one thread, as the rules ask: keep the linear-algebra library from starting its own (must be set before NumPy loads)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# ---------------------------------------------------------------- declared platform constants
DRONE_GYRO_CAL = np.array([-0.0022018809090909091, 0.022254927272727272, 0.077799836363636368])
K_DRAG = 0.19496636404464057
LEVER = np.array([0.018644724812211431, 0.015676984800147502, 0.10267844651274931])
RATE = np.array([-0.013313909714381402, 0.014065944625227762, 0.029610013375686053])
EDGES = np.array([0, 1.0 / 256, 1.0 / 128, 1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1, 2, 1e9])
ERR_XY = np.array([2.0644044799297574e-33, 0.00083701680649340298, 0.0035667962383138601, 0.010129373443721503,
                   0.010055975141197816, 0.0056158554693356262, 0.0020367076785602193, 0.00081228236915690665,
                   0.00035951337258634665, 0.00015714926358070501, 6.1772936873354928e-05])
ERR_Z = np.array([8.8698405112690774e-35, 6.632350230792317e-05, 0.0001059201364596372, 0.00028128343903706664,
                  0.00049844387814139217, 0.00037394354081618053, 0.00017696485056683476, 0.00010295535752374746,
                  9.0362198518052027e-05, 4.8671263056616192e-05, 2.0350926889176627e-05])
GAIN_XY = np.array([5.5750279286166194e-17, 0.0056957691568474888, 0.062254196112800475, 0.22295319468886118,
                    0.1788242900745429, 0.96747099366975364, 0.96246303180511728, 0.94112730068352957,
                    0.92161894560820623, 0.86486504244358242, 0.81982851253914224])
GAIN_Z = np.array([0, 0.021256890787656885, 0, 0.073248875799799662, 0.12918869551390599, 0.43889718008461948,
                   0.90620593224091406, 0.91642940849231225, 1.0002979430773686, 0.95951238831035623, 0.84385350013881344])

HOP = int(os.environ.get("PF_HOP", 8))   # solver block: rows per velocity sample
CH = 2048            # rows per piece for per-row work, so that temporaries stay small
ROW_BUDGET = int(os.environ.get("PF_ROWS", 40000))   # most rows one solve holds; a faster stretch is averaged down
# size caps of the one solve, so that a long file fits in the same memory as a short one (none binds below about 3 min)
KNOTS_MAX = int(os.environ.get("PF_KNOTS", 24))      # knots per bias curve; a longer file gets wider knot spacing
RES_MAX = int(os.environ.get("PF_RES", 200))       # residual rows (normally one per second); a longer file gets one every few seconds
GAPS_MAX = int(os.environ.get("PF_GAPS", 20))       # dropouts that keep their own velocity step and tilt; shorter ones beyond this are bridged
DTV = 0.1            # s, velocity track for the post-filter
BRIDGE_HAND, BRIDGE_DRONE = 0.08, 0.25   # s, longest dead run filled by interpolation
RATE_LIMIT = 0.3     # m/s^2, evidence-chosen accel drift rate above which a drone solve is distrusted
DEBUG = bool(os.environ.get("PF_DEBUG"))
EXPL = bool(os.environ.get("PF_EXPL"))   # debug: build every gyro column the long way


def dbg(*a):
    if DEBUG:
        sys.stderr.write(" ".join(str(x) for x in a) + "\n")


def mark(label):
    """Debug only: peak traced heap since the last mark."""
    if DEBUG:
        import tracemalloc
        if tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            sys.stderr.write("  heap %-9s now %.2f MB, peak %.2f MB\n" % (label, cur / 1e6, peak / 1e6))
            tracemalloc.reset_peak()


# ---------------------------------------------------------------- small rotation maths (quaternions are w x y z)
def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    o = np.empty(np.broadcast(aw, bw).shape + (4,))
    o[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    o[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    o[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    o[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return o


def qexp(r):
    """Rotation vector(s) to quaternion(s)."""
    r = np.asarray(r, float)
    a = np.sqrt((r * r).sum(-1))
    small = a < 1e-12
    s = np.where(small, 0.5, np.sin(0.5 * a) / np.where(small, 1.0, a))
    q = np.concatenate([np.cos(0.5 * a)[..., None], r * s[..., None]], -1)
    return q / np.sqrt((q * q).sum(-1))[..., None]


def qlog(q):
    """Quaternion to rotation vector (single)."""
    q = q / np.sqrt(q @ q)
    if q[0] < 0:
        q = -q
    s = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if s < 1e-12:
        return 2.0 * q[1:]
    return q[1:] * (2.0 * np.arctan2(s, q[0]) / s)


def qmat(q):
    """Quaternion(s) to rotation matrix (body to world)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - w * z); m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y); m[..., 2, 1] = 2 * (y * z + w * x); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def qchain(dq, q0=None):
    """Running product q0 * dq[0] * dq[1] * ... for every row, without a Python loop over rows: products inside chunks of
    16 are vectorised across chunks, then the chunk totals are chained."""
    n = len(dq)
    L = 16
    nc = (n + L - 1) // L
    pad = np.zeros((nc * L, 4))
    pad[:, 0] = 1.0
    pad[:n] = dq
    Q = pad.reshape(nc, L, 4)
    for j in range(1, L):
        Q[:, j] = qmul(Q[:, j - 1], Q[:, j])
    pre = np.empty((nc, 4))
    aw, ax, ay, az = (1.0, 0.0, 0.0, 0.0) if q0 is None else [float(c) for c in q0]
    tot = Q[:, L - 1].tolist()
    for c in range(nc):                       # plain floats: a few hundred steps, no array overhead
        pre[c, 0], pre[c, 1], pre[c, 2], pre[c, 3] = aw, ax, ay, az
        bw, bx, by, bz = tot[c]
        aw, ax, ay, az = (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                          aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)
        nn = (aw * aw + ax * ax + ay * ay + az * az) ** 0.5
        aw, ax, ay, az = aw / nn, ax / nn, ay / nn, az / nn
    out = qmul(pre[:, None, :], Q).reshape(nc * L, 4)[:n]
    return out / np.sqrt((out * out).sum(1))[:, None]


def chain_chunks(n, t, rate, q0=None):
    """Attitude after every row, in pieces of CH rows. rate(a, b) gives the body rate of rows a..b-1.
    Yields (a, b, q) with q of shape (b - a, 4)."""
    q = np.array([1.0, 0, 0, 0]) if q0 is None else np.asarray(q0, float)
    for a in range(0, n, CH):
        b = min(n, a + CH)
        dt = np.empty(b - a)
        dt[1:] = t[a + 1:b] - t[a:b - 1]
        dt[0] = t[a] - t[a - 1] if a > 0 else 0.0
        qc = qchain(qexp(rate(a, b) * dt[:, None]), q)
        q = qc[-1]
        yield a, b, qc


def two_vec(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(a @ b)
    if d < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    c = np.cross(a, b)
    q = np.array([1 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


def box(x, size):
    """Running mean along axis 0 with the edges held (scipy uniform_filter1d, mode nearest)."""
    n = len(x)
    lo = size // 2
    hi = size - lo - 1
    shp = (1,) * (x.ndim - 1)
    p = np.concatenate([np.repeat(x[:1], lo, 0), x, np.repeat(x[-1:], hi, 0)], 0)
    c = np.cumsum(p, axis=0)
    out = c[size - 1:].copy()
    out[1:] -= c[:n - 1]
    del shp
    return out / size


def box_rows(x, rows, size):
    """box(x, size)[rows] from prefix sums, for rows that are at least size // 2 from the start."""
    n = len(x)
    c = np.cumsum(x, axis=0)
    lo = rows - size // 2
    hi = lo + size
    over = np.maximum(hi - n, 0)
    top = c[np.minimum(hi, n) - 1]
    out = top.copy()
    m = lo > 0
    out[m] -= c[lo[m] - 1]
    if over.any():
        out += over.reshape((-1,) + (1,) * (x.ndim - 1)) * x[-1]
    return out / size


def lowpass(x, fc, fs):
    """Zero-phase low-pass with the response of a second-order Butterworth run forwards and backwards."""
    n = len(x)
    pad = min(n - 1, int(3 * fs / fc))
    p = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
    X = np.fft.rfft(p, axis=0)
    w = np.fft.rfftfreq(len(p), 1.0 / fs)
    wc = np.tan(np.pi * fc / fs)
    ww = np.tan(np.pi * np.minimum(w, 0.4999 * fs) / fs)
    g = 1.0 / (1.0 + (ww / wc) ** 4)
    return np.fft.irfft(X * g.reshape((-1,) + (1,) * (x.ndim - 1)), len(p), axis=0)[pad:pad + n]


def spline_basis(tt, knots):
    B = np.zeros((len(tt), len(knots)))
    idx = np.clip(np.searchsorted(knots, tt) - 1, 0, len(knots) - 2)
    x = np.clip((tt - knots[idx]) / (knots[idx + 1] - knots[idx]), 0, 1)
    B[np.arange(len(tt)), idx] = 1 - x
    B[np.arange(len(tt)), idx + 1] = x
    return B


def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


# ---------------------------------------------------------------- input
def find_csv(path):
    """The IMU CSV: the file itself, imu0/data.csv under a dataset folder, or the same inside a zip (unpacked to a
    temporary file next to nothing else; it is removed when the run ends)."""
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        import atexit
        import shutil
        import tempfile
        import zipfile
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.replace("\\", "/").endswith("imu0/data.csv")]
            if not names:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                sys.exit("main.py: no IMU csv in " + path)
            tmp = tempfile.mkdtemp(prefix="pathfinder_")
            atexit.register(shutil.rmtree, tmp, True)
            dst = os.path.join(tmp, "data.csv")
            with z.open(min(names, key=len)) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        return dst
    if os.path.isdir(path):
        best = None
        for d, _, files in os.walk(path):
            if "data.csv" in files and os.path.basename(d) == "imu0":
                p = os.path.join(d, "data.csv")
                if best is None or len(p) < len(best):
                    best = p
        if best is None:
            sys.exit("main.py: no imu0/data.csv under " + path)
        return best
    return path


def is_data(c):
    """A CSV line that holds a row (not empty, not a comment, not a header)."""
    return bool(c) and c[0] != "#" and (c[0].isdigit() or c[0] in "+-.")


def values(chunk):
    """Timestamps (float64, approximate: for binning only) and the six sensor values (float32) of a list of CSV lines;
    what cannot be read is NaN."""
    try:
        a = np.loadtxt(chunk, delimiter=",", dtype=np.float64, ndmin=2)
        if a.shape[1] < 7:
            raise ValueError
        return a[:, 0].copy(), a[:, 1:7].astype(np.float32)
    except Exception:
        t = np.full(len(chunk), np.nan)
        out = np.full((len(chunk), 6), np.nan, np.float32)
        for i, c in enumerate(chunk):
            q = c.split(",")
            v = num(q[0].strip())
            if v is not None:
                t[i] = v
            for j in range(6):
                try:
                    out[i, j] = float(q[1 + j])
                except (ValueError, IndexError):
                    pass
        return t, out


def unit(step):
    """Seconds per timestamp unit, from the usual step between rows (ns, us, ms or s)."""
    return 1e-9 if step >= 1e5 else (1e-6 if step >= 1e2 else (1e-3 if step >= 0.5 else 1.0))


def data_lines(path, lo=0, hi=None):
    """The data lines of the file (stripped), rows lo..hi-1 counted in data rows."""
    i = -1
    with open(path, "r") as f:
        for line in f:
            c = line.strip()
            if not is_data(c):
                continue
            i += 1
            if i < lo:
                continue
            if hi is not None and i >= hi:
                return
            yield c


def num(h):
    """A timestamp's value in file units (int when it is a whole number), or None."""
    try:
        v = int(h) if h.lstrip("+-").isdigit() else float(h)
        return v if v == v and abs(v) != float("inf") else None
    except ValueError:
        return None


def read_cells(lines, first, last, width, scale):
    """Reader for files beyond the row budget: rows are averaged into cells of `width` timestamp units while they are
    read, so the file is never held at its full rate. Rows that cannot be used (zeros, unreadable, a frozen repeat, a
    spike) are left out of the average. A cell whose rows are all unusable stays a zero row; a cell without rows is a
    dropout. A row's cell never moves backwards (an unreadable or out-of-range stamp stays in the cell before it).
    Returns [time in s, six values] of the cells that have rows, and how many input rows each of them holds."""
    top = int((last - first) / width + 0.5) + 1
    acc = np.zeros((top + 1, 6), np.float32)
    used = np.zeros(top + 1, np.uint32)
    rows = np.zeros(top + 1, np.uint32)
    state = dict(prev=0, row=np.full(6, np.nan, np.float32))
    chunk = []
    f0 = float(first)

    def flush():
        if not chunk:
            return
        t, v = values(chunk)
        c = np.floor((t - f0) / width + 0.5)
        c[~np.isfinite(c) | (c > top)] = -1
        c = np.maximum.accumulate(np.concatenate([[state["prev"]], c]))[1:].astype(np.int64)
        state["prev"] = int(c[-1])
        np.add.at(rows, c, 1)
        good = np.isfinite(v).all(1) & (v != 0).any(1) & (np.abs(v[:, :3]).max(1) < 100.0) & (np.abs(v[:, 3:]).max(1) < 1000.0)
        rep = np.zeros(len(v), bool)
        rep[0] = bool((v[0] == state["row"]).all())
        rep[1:] = (v[1:] == v[:-1]).all(1)                # a frozen sensor repeats all six values exactly
        state["row"] = v[-1].copy()
        good &= ~rep
        if good.sum() > 16 and len(v) > 8:                # spikes: far from the median of five rows, as in repair()
            n1 = float(np.median(np.sqrt((v[good, :3].astype(np.float64) ** 2).sum(1))))
            n2 = float(np.median(np.sqrt((v[good, 3:].astype(np.float64) ** 2).sum(1))))
            gyro_first = abs(n2 - 9.81) <= abs(n1 - 9.81)
            inner = good[2:-2].copy()
            flag = np.zeros(len(v) - 4, bool)
            for j in range(6):
                x = v[:, j]
                dev = np.abs(x[2:-2] - np.median(np.stack([x[:-4], x[1:-3], x[2:-2], x[3:-1], x[4:]]), axis=0))
                if inner.any():
                    flag |= dev > 10.0 * 1.4826 * float(np.median(dev[inner])) + (0.02 if gyro_first == (j < 3) else 0.3)
            good[2:-2] &= ~flag
        np.add.at(acc, c[good], v[good])
        np.add.at(used, c[good], 1)
        chunk.clear()

    for c in lines:
        chunk.append(c)
        if len(chunk) >= 1024:
            flush()
    flush()
    acc /= np.maximum(used, 1)[:, None]
    del used
    seen = rows > 0
    t = np.nonzero(seen)[0] * (width * scale)
    v = acc[seen]
    del acc
    return [t, v], rows[seen].astype(np.int64)


def spread(chunks, counts):
    """Cell poses to row poses: every input row takes the pose of its cell. Pieces of at most about CH rows."""
    base = 0
    for P, Q in chunks:
        k = len(P)
        rep = counts[base:base + k]
        base += k
        cs = np.cumsum(rep)
        a = 0
        while a < k:
            before = int(cs[a - 1]) if a else 0
            b = max(a + 1, int(np.searchsorted(cs, before + CH, side="right")))
            yield np.repeat(P[a:b], rep[a:b], axis=0), np.repeat(Q[a:b], rep[a:b], axis=0)
            a = b


def read_rows(lines, scale):
    """Returns [time in seconds from the first row, six sensor values] of the given data lines.
    The timestamps themselves are not kept; they are read again when the poses are written (stamps)."""
    rel, vals = [], []
    state = dict(first=None, ints=True)
    if True:
        chunk = []

        def flush():
            if not chunk:
                return
            heads = [c.split(",", 1)[0].strip() for c in chunk]
            vals.append(values(chunk)[1])
            r = np.zeros(len(heads))
            for i, h in enumerate(heads):
                try:
                    if state["ints"] and h.lstrip("+-").isdigit():
                        v = int(h)
                    else:
                        v = float(h)
                        state["ints"] = False
                    if state["first"] is None:
                        state["first"] = v
                    r[i] = v - state["first"]             # whole numbers subtract exactly before the float conversion
                except ValueError:
                    state["ints"] = False
                    r[i] = np.nan
            rel.append(r)
            chunk.clear()

        for c in lines:
            chunk.append(c)
            if len(chunk) >= 1024:
                flush()
        flush()
    if not vals:
        return [np.zeros(0), np.zeros((0, 6), np.float32)]
    v = np.concatenate(vals) if len(vals) > 1 else vals[0]
    del vals
    t = np.concatenate(rel) if len(rel) > 1 else rel[0]
    del rel
    bad = ~np.isfinite(t)
    if bad.any():                             # unreadable stamps: carry the last good one forward
        t = t[np.maximum.accumulate(np.where(bad, 0, np.arange(len(t))))]
        t[~np.isfinite(t)] = 0.0
    np.maximum.accumulate(t, out=t)           # a stamp that goes backwards counts as a repeat of the one before
    t *= scale
    return [t, v]


def stamps(path, exact_ns, scale):
    """The timestamp of every data row, in file order, as the text to write (seconds)."""
    with open(path, "r") as f:
        for line in f:
            c = line.strip()
            if not is_data(c):
                continue
            h = c.split(",", 1)[0].strip()
            d = h.lstrip("+-")
            if exact_ns and d.isdigit():
                d = d.zfill(10)
                yield ("-" if h[0] == "-" else "") + d[:-9] + "." + d[-9:]
            else:
                try:
                    yield "%.9f" % (float(h) * scale)
                except ValueError:
                    yield "0.000000000"


# ---------------------------------------------------------------- grid, validity and repair
class Rows:
    pass


def make_grid(t_in, v):
    """Regular grid at the median sample period. Missing rows are re-inserted as dead rows; a repeated timestamp maps to
    the same grid row. Returns the grid and, for every input row, its grid row."""
    n_in = len(t_in)
    d = np.diff(t_in)
    med = float(np.median(d[d > 0])) if (d > 0).any() else 0.005
    step = np.maximum(np.rint(d / med), 0).astype(np.int64)
    gi = np.concatenate([[0], np.cumsum(step)])
    n = int(gi[-1]) + 1
    R = Rows()
    R.med = med
    R.fs = int(round(1.0 / med))
    good = np.isfinite(v).all(1) & (v != 0).any(1) & (np.abs(v[:, :3]).max(1) < 100.0) & (np.abs(v[:, 3:]).max(1) < 1000.0)
    rep = np.zeros(n_in, bool)
    rep[1:] = (v[1:] == v[:-1]).all(1)        # a frozen sensor repeats all six values exactly
    good &= ~rep
    imu = np.zeros((n, 6), np.float32)
    ok = np.zeros(n, bool)
    first = np.ones(n_in, bool)
    first[1:] = step > 0                      # a repeated timestamp keeps the first row's values
    sel = first & good
    imu[gi[sel]] = v[sel]
    ok[gi[sel]] = True
    R.imu, R.ok, R.n = imu, ok, n
    R.t = t_in[0] * 0 + np.arange(n) * med
    # keep true times where rows exist (jitter), so integration uses the real intervals
    R.t[gi[first]] = t_in[first] - t_in[0]
    R.t = np.maximum.accumulate(R.t)
    return R, gi


def repair(R, bridge_s, spikes=True):
    ok, imu = R.ok, R.imu
    live = np.nonzero(ok)[0]
    if len(live) < 16:
        return
    last = int(live[-1])
    if spikes:
        n5 = ok[:-4] & ok[1:-3] & ok[2:-2] & ok[3:-1] & ok[4:]
        has = np.zeros(R.n, bool)
        has[2:-2] = n5
        has[last - 1:] = False
        flag = np.zeros(R.n, bool)
        blk = max(10, 2 * R.fs)
        for c in range(6):
            x = imu[:, c]
            m = np.median(np.stack([x[:-4], x[1:-3], x[2:-2], x[3:-1], x[4:]]), axis=0)
            dev = np.zeros(R.n, np.float32)
            dev[2:-2] = np.abs(x[2:-2] - m)
            floor = 0.02 if R.gyro_first == (c < 3) else 0.3
            for b0 in range(0, last + 1, blk):
                h = has[b0:b0 + blk]
                if h.any():
                    dd = dev[b0:b0 + blk]
                    sc = 1.4826 * float(np.median(dd[h]))
                    flag[b0:b0 + blk] |= h & (dd > 10.0 * sc + floor)
        ok[flag] = False
    bridge(R, bridge_s)


def bridge(R, seconds, most=None):
    """Interpolates dead runs inside the file (not the trailing ones) of at most `seconds`; with `most`, instead every
    run beyond the `most` longest (ties broken towards later runs)."""
    ok, imu = R.ok, R.imu
    live = np.nonzero(ok)[0]
    if len(live) < 2:
        return
    last = int(live[-1])
    s, e = runs_of(~ok[:last + 1])
    if most is None:
        keep = (s > 0) & (e - s <= max(1, int(seconds * R.fs)))
    else:
        inner = np.nonzero(s > 0)[0]
        keep = np.zeros(len(s), bool)
        if len(inner) > most:
            order = inner[np.lexsort((inner, e[inner] - s[inner]))]      # shortest first, earlier first on ties
            keep[order[:len(inner) - most]] = True
    if not keep.any():
        return
    fill = np.zeros(R.n, bool)
    for a, b in zip(s[keep], e[keep]):
        fill[a:b] = True
    idx = np.nonzero(fill)[0]
    for c in range(6):
        imu[idx, c] = np.interp(R.t[idx], R.t[live], imu[live, c])
    ok[idx] = True


def cap_gaps(R, most):
    """Keeps at most `most` dropouts (the longest); shorter ones are bridged like short holes. Only long files have more."""
    bridge(R, 0.0, most)
    R.live = np.nonzero(R.ok)[0].astype(np.int32)


def filled(R, cols, a=0, b=None):
    """Sensor columns (a slice of three) for rows a..b-1, with dead rows interpolated from their neighbours."""
    b = R.n if b is None else b
    live = R.live
    idx = np.arange(a, b)
    lo = max(0, int(np.searchsorted(live, a)) - 1)
    hi = min(len(live), int(np.searchsorted(live, b)) + 1)
    lv = live[lo:hi]
    return np.stack([np.interp(idx, lv, R.imu[lv, c]) for c in range(cols.start, cols.stop)], 1)


# ---------------------------------------------------------------- platform and gyro calibration
def rotor_windows(R):
    """Per second: is rotor vibration present in the accelerometer? (size of its third difference)"""
    win = max(1, R.fs)
    nw = (R.n + win - 1) // win
    on = np.zeros(nw, bool)
    for a in range(0, R.n, CH):
        b = min(R.n, a + CH + 3)
        d3 = np.abs(np.diff(filled(R, R.ac, a, b), 3, axis=0))
        for w in range(a // win, min(nw, (min(R.n, a + CH) + win - 1) // win)):
            i = w * win
            if i < a:
                continue
            ln = min(win, R.n - i)
            if ln > 5 and i - a + ln - 3 <= len(d3):
                on[w] = np.median(d3[i - a:i - a + ln - 3]) / (0.6745 * np.sqrt(20.0)) > 0.35
    return on, win


def rest_bias(R, g):
    """Median gyro over blocks that are truly at rest (rotors off, gyro as quiet as its own noise), or None."""
    win = int(0.5 * R.fs)
    if win < 8:
        return None
    out = []
    n3 = lambda x: np.median(np.abs(np.diff(x, 3, axis=0)), axis=0) / (0.6745 * np.sqrt(20.0))
    for s in range(0, R.n - win, win):
        if not R.ok[s:s + win].all():
            continue
        x = R.imu[s:s + win].astype(np.float64)
        w, a = x[:, R.gy], x[:, R.ac]
        sd = w.std(0)
        if not (abs(np.linalg.norm(a.mean(0)) - g) < 0.2 and sd.max() < 0.03):
            continue
        ratio = float((sd / np.maximum(n3(w), 1e-4)).max())
        aN3 = float(n3(a).max())
        if (ratio < 1.8 or sd.max() < 0.006) and a.std(0).max() / max(aN3, 1e-4) < 3.0 and aN3 < 0.3:
            out.append(w.mean(0))
    if len(out) < 2:
        return None
    return np.median(np.array(out), 0)


def window_bias(R):
    """Gyro bias across the up direction from 4 s windows (see check in main). Returns bias (up part removed), up, n."""
    fs = R.fs
    wn, en = int(round(4.0 * fs)), max(1, int(round(0.5 * fs)))
    live = np.nonzero(R.ok)[0]
    last = int(live[-1])
    acc_all = R.imu[:, R.ac]
    up = acc_all[R.ok].astype(np.float64).sum(0)
    up /= np.linalg.norm(up)
    bad = np.concatenate([[0], np.cumsum(~R.ok[:last + 1])])
    wins = [k for k in range(0, last + 1 - wn, wn) if bad[k + wn] - bad[k] == 0]
    if not wins:
        return np.zeros(3), up, 0
    dt_all = np.diff(R.t, append=R.t[-1] + 1.0 / fs)
    b = np.zeros(3)
    used = 0
    for _ in range(3):
        rr, SS = [], []
        for k0 in wins:
            k1 = k0 + wn
            m0 = acc_all[k0:k0 + en].astype(np.float64).mean(0)
            m1 = acc_all[k1 - en:k1].astype(np.float64).mean(0)
            n0, n1 = np.linalg.norm(m0), np.linalg.norm(m1)
            if n0 < 1e-6 or n1 < 1e-6:
                continue
            m0, m1 = m0 / n0, m1 / n1
            a, e = k0 + en // 2, k1 - en // 2
            dt = dt_all[a:e]
            q = qchain(qexp((R.imu[a:e, R.gy].astype(np.float64) - b) * dt[:, None]))
            Cb = qmat(np.concatenate([[[1.0, 0, 0, 0]], q[:-1]]))
            IC = (Cb * dt[:, None, None]).sum(0)
            C = qmat(q[-1])
            K = np.array([[0, -m0[2], m0[1]], [m0[2], 0, -m0[0]], [-m0[1], m0[0], 0.0]])
            rr.append(m1 - C.T @ m0)
            SS.append(-C.T @ K @ IC)
        if not rr:
            return np.zeros(3), up, 0
        used = len(rr)
        res = np.array([np.linalg.norm(r) for r in rr])
        sc = 1.4826 * np.median(res) + 1e-6
        wt = np.minimum(1.0, 1.5 * sc / np.maximum(res, 1e-12))
        N = sum(w * S.T @ S for w, S in zip(wt, SS))
        gg = sum(w * S.T @ r for w, S, r in zip(wt, SS, rr))
        lam = 1e-2 * np.trace(N) / 3.0
        b = np.linalg.solve(N + lam * np.eye(3), gg + N @ b)
    return b - (b @ up) * up, up, used


# ---------------------------------------------------------------- the solver
class Solver:
    def __init__(self, R, b0, knot=8.0):
        self.R = R
        n, t, ok = R.n, R.t, R.ok
        self.n, self.t, self.ok = n, t, ok
        self.b0 = np.asarray(b0, float)
        live = np.nonzero(ok)[0]
        self.last = last = int(live[-1])
        self.t_end = t[last]
        acc = R.imu[:, R.ac]
        self.g = float(np.median(np.sqrt((acc[ok].astype(np.float64) ** 2).sum(1))))
        dur = self.t_end - t[0]
        self.Ka = self.Kg = max(2, int(round(dur / knot)) + 1)
        self.kn = np.linspace(t[0], self.t_end + 1e-3, self.Ka)
        gs, ge = runs_of(~ok)
        self.gaps_all = list(zip(gs.tolist(), ge.tolist()))
        self.gaps = [(s, e) for s, e in self.gaps_all if e < n and s > 0]
        self.G = len(self.gaps)
        edges = [0] + [e for s, e in self.gaps]
        ends = [s for s, e in self.gaps] + [last + 1]
        self.segs = list(zip(edges, ends))
        self.K = len(self.segs)
        i = np.arange(0, last + 1 - HOP + 1, HOP)
        self.idx, self.mid = i, i + HOP // 2
        self.Nb = len(i)
        self.tb = t[self.mid]
        self.okb = ok[self.mid].astype(np.float64)
        m = i[-1] + HOP
        okf = ok[:m].astype(np.float64)
        self.fb = np.add.reduceat(acc[:m].astype(np.float64) * okf[:, None], i, axis=0) / np.maximum(np.add.reduceat(okf, i), 1)[:, None]
        self.dtb = np.diff(t[np.append(i, min(i[-1] + HOP, last))])
        tc = np.minimum(self.tb, self.t_end)
        # linear-spline weights of every block: knot bi with weight 1 - bx, knot bi + 1 with weight bx (both bias curves)
        self.bi = np.clip(np.searchsorted(self.kn, tc) - 1, 0, self.Ka - 2)
        self.bx = np.clip((tc - self.kn[self.bi]) / (self.kn[self.bi + 1] - self.kn[self.bi]), 0, 1)
        self.seg_of = np.full(self.Nb, -1)
        for k, (a, b) in enumerate(self.segs):
            self.seg_of[(self.mid >= a) & (self.mid < b)] = k
        self.x = dict(v0=np.zeros(3), S=np.zeros((self.K, 3)), ca=np.zeros((self.Ka, 3)), cg=np.zeros((self.Kg, 3)), dv=np.zeros((self.G, 3)))
        self.q0 = self.quiet_init()
        self.x["cg"] = self.gravity_fit(knot)
        self.seed_tilts(20.0)
        self.vb_t = np.zeros((self.Nb, 3))
        self.on_b = np.zeros(self.Nb, bool)
        self.drone = False

    # attitude from the gyro with a bias curve; dead rows coast on a rate carried across the gap
    def rates(self, cg, knots=None):
        """Returns rate(a, b): body rate of rows a..b-1, gyro minus the bias curve, with dead rows coasting."""
        R, t, ok, n = self.R, self.t, self.ok, self.n
        kn = self.kn if knots is None else knots
        gy = R.imu[:, R.gy]

        def raw(a, b):
            tc = np.minimum(t[a:b], self.t_end)
            w = gy[a:b].astype(np.float64) - self.b0
            for i in range(3):
                w[:, i] -= np.interp(tc, kn, cg[:, i])
            w[~ok[a:b]] = 0.0
            return w

        side = max(1, int(0.25 * R.fs))
        coast = []
        for s, e in self.gaps_all:
            a0 = max(0, s - side)
            wb = raw(a0, s).mean(0) if s > a0 else np.zeros(3)
            if e < n:
                wa = raw(e, min(n, e + side)).mean(0)
                x = (np.arange(e - s) + 1.0) / (e - s + 1.0)
                coast.append(wb[None] * (1 - x)[:, None] + wa[None] * x[:, None])
            else:
                t0 = t[s - 1] if s > 0 else t[s]
                coast.append(wb[None] * np.exp(-(t[s:e] - t0) / 0.5)[:, None])

        def rate(a, b):
            w = raw(a, b)
            for (s, e), cw in zip(self.gaps_all, coast):
                if e > a and s < b:
                    lo, hi = max(s, a), min(e, b)
                    w[lo - a:hi - a] = cw[lo - s:hi - s]
            return w
        return rate

    def gyro_at(self, cg, want, q0=None, knots=None):
        """Attitude at the rows `want` (ascending)."""
        out = np.empty((len(want), 4))
        for a, b, qc in chain_chunks(self.n, self.t, self.rates(cg, knots), self.q0 if q0 is None else q0):
            lo, hi = np.searchsorted(want, [a, b])
            if hi > lo:
                out[lo:hi] = qc[want[lo:hi] - a]
        return out

    def quiet_init(self):
        """Start attitude from the quietest second (its mean accel is up), carried back to the first row by the gyro."""
        R, ok, fs = self.R, self.ok, self.R.fs
        best, bk, ba = 1e300, 0, None
        w_all = R.imu[:, R.gy]
        a_all = R.imu[:, R.ac]
        for s in range(0, self.n - fs, fs):
            m = ok[s:s + fs]
            if m.sum() < 0.8 * fs:
                continue
            w = w_all[s:s + fs][m].astype(np.float64) - self.b0
            a = a_all[s:s + fs][m].astype(np.float64)
            off = np.linalg.norm(a.mean(0)) - 9.81
            score = (w ** 2).sum(1).mean() - (w.mean(0) ** 2).sum() + 0.01 * ((a ** 2).sum(1).mean() - (a.mean(0) ** 2).sum()) + 0.01 * off * off
            if score < best:
                best, bk, ba = score, s + fs // 2, a.mean(0)
        if ba is None:
            return np.array([1.0, 0, 0, 0])
        q_up = two_vec(ba, np.array([0.0, 0.0, 1.0]))
        if bk < 2:
            return q_up

        def rate(a, b):
            w = w_all[a:b].astype(np.float64) - self.b0
            w[~ok[a:b]] = 0.0
            return w
        qc = None
        for a, b, q in chain_chunks(bk, self.t, rate):
            qc = q[-1]
        qq = qmul(q_up, qc * np.array([1.0, -1, -1, -1]))
        return qq / np.linalg.norm(qq)

    def gravity_fit(self, knot):
        """First guess of the gyro-bias curve: the averaged accel, turned into the world, should point up."""
        R, t, ok, n = self.R, self.t, self.ok, self.n
        fs = R.fs
        J = self.Kg
        knots = np.linspace(t[0], self.t_end, J)
        knots[-1] += 1e-3
        c = np.zeros((J, 3))
        acc = R.imu[:, R.ac]
        wn = max(1, fs, int(np.ceil(self.last / 600.0)))     # 1 s windows, at most 600 of them (long files: wider)
        starts = np.arange(0, self.last - wn, wn)
        if len(starts) < 3:
            return c
        okf = ok.astype(np.float64)
        cnt = np.array([okf[s:s + wn].sum() for s in starts])
        am = np.array([(acc[s:s + wn].astype(np.float64) * okf[s:s + wn, None]).sum(0) for s in starts])
        keep = cnt > 0.5 * wn
        starts, cnt, am = starts[keep], cnt[keep], am[keep] / np.maximum(cnt[keep], 1)[:, None]
        an = np.linalg.norm(am, axis=1)
        centre = starts + wn // 2
        w0 = (cnt / wn) * np.exp(-((an - self.g) / 0.5) ** 2)
        u = am / an[:, None]
        hop = 10
        bi = np.arange(hop // 2, n, hop)
        dtb = t[np.minimum(bi + hop // 2, n - 1)] - t[bi - hop // 2]
        okb = ok[bi].astype(np.float64)
        tcb = np.minimum(t[bi], self.t_end)
        gi_ = np.clip(np.searchsorted(knots, tcb) - 1, 0, J - 2)
        gx_ = np.clip((tcb - knots[gi_]) / (knots[gi_ + 1] - knots[gi_]), 0, 1)
        kb = np.minimum(np.searchsorted(bi, centre), len(bi) - 1)
        Kc = len(centre)
        s_r, sigma_b = 0.3, 0.003
        q0 = self.q0
        self.kn_fit = knots
        for it in range(3):
            both = np.union1d(bi, centre)
            q = self.gyro_at(c, both, q0, knots)
            Rm = qmat(q[np.searchsorted(both, bi)])
            m = np.einsum("nij,nj->ni", qmat(q[np.searchsorted(both, centre)]), u)
            del q
            A = np.zeros((2 * Kc, 2 + 3 * J))
            A[:Kc, 0] = 1.0
            A[Kc:, 1] = 1.0
            wgt = dtb * okb
            for j in range(J):
                a0 = int(np.searchsorted(gi_, j - 1, "left"))
                r1 = int(np.searchsorted(gi_, j, "right"))
                if r1 <= a0:
                    continue
                bl = np.where(gi_[a0:r1] == j, 1 - gx_[a0:r1], gx_[a0:r1])
                loc = np.cumsum(-Rm[a0:r1, :2, :] * (bl * wgt[a0:r1])[:, None, None], axis=0)
                cum = np.zeros((Kc, 2, 3))
                ins = (kb >= a0) & (kb < r1)
                cum[ins] = loc[kb[ins] - a0]
                cum[kb >= r1] = loc[-1]
                A[:Kc, 2 + 3 * j:5 + 3 * j] = cum[:, 0, :]
                A[Kc:, 2 + 3 * j:5 + 3 * j] = cum[:, 1, :]
            y = np.concatenate([m[:, 1], -m[:, 0]])
            w = np.sqrt(np.concatenate([w0, w0])) / s_r
            A *= w[:, None]                                  # weighted in place: no second copy of A
            y *= w
            ATA = A.T @ A
            ATy = A.T @ y
            del A
            pri = np.zeros(2 + 3 * J)
            pri[2:] = 1.0 / sigma_b ** 2
            x0 = np.concatenate([[0, 0], c.ravel()])
            dx = np.linalg.solve(ATA + np.diag(pri + 1e-9), ATy - pri * x0)
            q0 = qmul(qexp(np.array([dx[0], dx[1], 0.0])), q0)
            q0 /= np.linalg.norm(q0)
            c = c + dx[2:].reshape(J, 3)
        # the solver's knots end 1 ms later; re-express the curve on them
        return spline_basis(np.linspace(t[0], self.t_end, self.Kg), knots) @ c

    def seed_tilts(self, W):
        """Up direction per segment from gravity over the segment's first W seconds. Exact, so any size of tilt is fine."""
        R = self.R
        acc = R.imu[:, R.ac]
        pick = []
        for a, b in self.segs:
            rr = np.arange(a, min(a + int(W * R.fs), b))
            pick.append(rr[self.ok[rr]])
        allr = np.unique(np.concatenate(pick)) if pick else np.zeros(0, int)
        if not len(allr):
            return
        qg = self.gyro_at(self.x["cg"], allr)
        for k, rr in enumerate(pick):
            if len(rr) < 50:
                continue
            u = np.einsum("nij,nj->ni", qmat(qg[np.searchsorted(allr, rr)]), acc[rr].astype(np.float64)).mean(0)
            u /= np.linalg.norm(u)
            ax = np.cross(u, [0.0, 0.0, 1.0])
            sn = np.linalg.norm(ax)
            if sn > 1e-9:
                self.x["S"][k] = ax / sn * np.arctan2(sn, u[2])

    def ramp(self, s, e):
        return np.clip((self.idx - s) / max(1, e - s), 0.0, 1.0)

    def curve(self, c):
        """Bias curve with knot values c, at every block."""
        return c[self.bi] * (1 - self.bx)[:, None] + c[self.bi + 1] * self.bx[:, None]

    def support(self, j):
        """Blocks a0..r1-1 that knot j influences, and its weight there."""
        a0 = int(np.searchsorted(self.bi, j - 1, "left"))
        r1 = int(np.searchsorted(self.bi, j, "right"))
        bl = np.where(self.bi[a0:r1] == j, 1 - self.bx[a0:r1], self.bx[a0:r1])
        return a0, r1, bl

    def forward(self, x):
        """Velocity per block for the unknowns x. Returns v, Rg (gyro attitude; None when there is one segment), Rm
        (attitude with the segment tilt), fw (world specific force) and the segment tilts as matrices."""
        qg = self.gyro_at(x["cg"], self.mid)
        Sq = qexp(x["S"])
        Sm = qmat(Sq)
        if self.K == 1 and not EXPL:
            Rm = qmat(qmul(Sq[0], qg))                # one tilt for the whole file: fold it into the quaternion
            Rg = None
        else:
            Rg = qmat(qg)
            Rm = np.matmul(Sm[np.maximum(self.seg_of, 0)], Rg)
        del qg
        fw = np.einsum("nij,nj->ni", Rm, self.fb) * self.okb[:, None]
        aw = fw - np.einsum("nij,nj->ni", Rm, self.curve(x["ca"])) * self.okb[:, None]
        aw[:, 2] -= self.g * self.okb
        aw *= self.dtb[:, None]
        v = np.cumsum(aw, axis=0)
        v += x["v0"]
        for j, (s, e) in enumerate(self.gaps):
            v += x["dv"][j][None] * self.ramp(s, e)[:, None]
        return v, Rg, Rm, fw, Sm

    def apply(self, x, dx):
        Ka, Kg, G, K = self.Ka, self.Kg, self.G, self.K
        xn = {k: np.array(v, float) for k, v in x.items()}
        p = 0
        xn["v0"] += dx[p:p + 3]
        p += 3
        for k in range(K):
            xn["S"][k] = qlog(qmul(qexp(np.array([dx[p], dx[p + 1], 0.0])), qexp(xn["S"][k])))
            p += 2
        xn["ca"] += dx[p:p + 3 * Ka].reshape(Ka, 3)
        p += 3 * Ka
        xn["cg"] += dx[p:p + 3 * Kg].reshape(Kg, 3)
        p += 3 * Kg
        if G:
            xn["dv"] += dx[p:p + 3 * G].reshape(G, 3)
        return xn

    def core(self, shift, iters, sig_a, sig_g, sig_v, sig_p, sig_dv=1.0):
        """Levenberg-Marquardt on the slow sensor errors, with the drift stiffness re-chosen by evidence.

        Residual rows: the 5 s average velocity once per second, and the centred position at the same blocks.
        The sensitivity of velocity to one unknown is M (blocks, 3, c). A bias knot only acts inside its two knot
        intervals; after them its effect on velocity is constant (accel bias) or follows the summed specific force
        (gyro bias), so the rows are assembled from that short stretch plus a closed-form tail (put_local)."""
        dtb, Nb = self.dtb, self.Nb
        fs_b = 1.0 / max(float(np.median(dtb[dtb > 0])) if (dtb > 0).any() else 0.04, 1e-4)
        smooth = max(1, int(5.0 * fs_b))
        stride = max(1, int(max(1.0, float(dtb.sum()) / RES_MAX) * fs_b))
        sw = 1.0 / stride
        x = {k: np.array(v, float) for k, v in self.x.items()}
        Ka, Kg, G, K = self.Ka, self.Kg, self.G, self.K
        npar = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
        rows = np.arange(smooth // 2, Nb, stride)
        m = len(rows)
        sp, sv = 1.0 / sig_p, 1.0 / sig_v
        ia = 3 + 2 * K
        ig = ia + 3 * Ka
        lo1 = rows - smooth // 2 - 1                     # window of a row: blocks lo1 + 1 .. hi1, edge held
        hi = lo1 + 1 + smooth
        over = np.maximum(hi - Nb, 0)
        hi1 = np.minimum(hi, Nb) - 1
        T = np.cumsum(dtb)
        ST = np.cumsum(T)
        wdt = dtb * self.okb
        seg = np.maximum(self.seg_of, 0)

        def diff_pen(nk):
            D = np.zeros((nk - 1, nk))
            D[np.arange(nk - 1), np.arange(nk - 1)] = -1.0
            D[np.arange(nk - 1), np.arange(1, nk)] = 1.0
            return np.kron(D.T @ D, np.eye(3))

        S_a, S_g = diff_pen(Ka), diff_pen(Kg)
        alpha = dict(a=1.0 / sig_a ** 2, g=1.0 / sig_g ** 2)
        amp_a, amp_g = 1.0, 100.0

        def prior():
            P = np.zeros((npar, npar))
            P[:3, :3] = np.eye(3) / 25.0
            P[np.arange(3, ia), np.arange(3, ia)] = 1e-3     # keeps a tilt with no data of its own (a very short segment) solvable
            P[ia:ia + 3 * Ka, ia:ia + 3 * Ka] = alpha["a"] * S_a + amp_a * np.eye(3 * Ka)
            P[ig:ig + 3 * Kg, ig:ig + 3 * Kg] = alpha["g"] * S_g + amp_g * np.eye(3 * Kg)
            if G:
                P[ig + 3 * Kg:, ig + 3 * Kg:] = np.eye(3 * G) / sig_dv ** 2
            return P

        def vec(x):
            return np.concatenate([x["v0"], np.zeros(2 * K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()])

        def obj(v, x):
            P = prior()
            c = vec(x)
            d = box(v - shift, smooth)
            p = np.cumsum(v * dtb[:, None], axis=0)
            p -= p.mean(0)
            return (d * d).sum() * sv * sv * sw + (p * p).sum() * sp * sp * sw + c @ P @ c - (x["v0"] ** 2).sum() / 25.0

        A = None             # the Jacobian: made at the start of each iteration and freed once the normal equations hold it

        def put_scalar(col, r):
            """Unknown whose velocity sensitivity is r (per block) times the identity."""
            vr = sv * box_rows(r, rows, smooth)
            pr = np.cumsum(r * dtb)
            pr -= pr.mean()
            pr = sp * pr[rows]
            for c in range(3):
                A[c:3 * m:3, col + c] = vr
                A[3 * m + c::3, col + c] = pr

        def put_explicit(col, M):
            """Sensitivity given for every block. M is overwritten."""
            c = M.shape[2]
            A[:3 * m, col:col + c] = sv * box_rows(M, rows, smooth).reshape(3 * m, c)
            M *= dtb[:, None, None]
            np.cumsum(M, axis=0, out=M)
            M -= M.mean(0, keepdims=True)
            A[3 * m:, col:col + c] = sp * M[rows].reshape(3 * m, c)

        def put_local(col, a0, r1, E, Cm, F, SF, Gs, SG):
            """Sensitivity that is zero before block a0, E on blocks a0..r1-1, and from r1 on E[-1] + skew(F_n - F[r1-1]) Cm
            (Cm None: constant E[-1]). Sums over windows and over time come from prefix sums evaluated where needed."""
            c = E.shape[2]
            SE = np.cumsum(E, axis=0)
            E *= dtb[a0:r1, None, None]
            PE = np.cumsum(E, axis=0)
            B0 = SE[-1] - (SE[-2] if r1 - a0 > 1 else 0.0)        # E[-1] before the scaling above

            def cr(V):                                           # skew(V) @ Cm for V (k, 3)
                o = np.empty((len(V), 3, c))
                o[:, 0, :] = V[:, 1:2] * Cm[2] - V[:, 2:3] * Cm[1]
                o[:, 1, :] = V[:, 2:3] * Cm[0] - V[:, 0:1] * Cm[2]
                o[:, 2, :] = V[:, 0:1] * Cm[1] - V[:, 1:2] * Cm[0]
                return o

            Mlast = B0
            if Cm is not None:
                B0 = B0 - cr(F[r1 - 1:r1])[0]
                Mlast = B0 + cr(F[Nb - 1:Nb])[0]
            if r1 > Nb - 1:
                Mlast = SE[-1] - (SE[-2] if r1 - a0 > 1 else 0.0)

            def at(q, loc, const, lin, base):
                """Prefix sum at blocks q: `loc` inside the support, then const + lin-weighted B0 + cr(base difference)."""
                o = np.zeros((len(q), 3, c))
                inside = (q >= a0) & (q < r1)
                o[inside] = loc[q[inside] - a0]
                tl = q >= r1
                if tl.any():
                    qt = q[tl]
                    val = const + lin(qt)[:, None, None] * B0
                    if Cm is not None:
                        val = val + cr(base[qt] - base[r1 - 1])
                    o[tl] = val
                return o

            vel = at(hi1, SE, SE[-1], lambda q: (q - r1 + 1.0), SF) - at(lo1, SE, SE[-1], lambda q: (q - r1 + 1.0), SF)
            vel += over[:, None, None] * Mlast
            A[:3 * m, col:col + c] = ((sv / smooth) * vel).reshape(3 * m, c)
            pos = at(rows, PE, PE[-1], lambda q: T[q] - T[r1 - 1], Gs)
            tot = PE.sum(0)
            cnt = Nb - r1
            if cnt > 0:
                tot = tot + cnt * PE[-1] + (ST[Nb - 1] - ST[r1 - 1] - cnt * T[r1 - 1]) * B0
                if Cm is not None:
                    tot = tot + cr((SG[Nb - 1] - SG[r1 - 1] - cnt * Gs[r1 - 1])[None])[0]
            A[3 * m:, col:col + c] = (sp * (pos - tot / Nb)).reshape(3 * m, c)

        v, Rg, Rm, fw, Sm = self.forward(x)
        best = obj(v, x)
        lam = 1e-3
        ones = np.ones(Nb)
        for it in range(iters):
            A = np.zeros((6 * m, npar), np.float32)
            F = np.cumsum(fw * dtb[:, None], axis=0)             # summed specific force: what a tilt error multiplies
            SF = np.cumsum(F, axis=0)
            Gs = np.cumsum(F * dtb[:, None], axis=0)
            SG = np.cumsum(Gs, axis=0)
            put_scalar(0, ones)
            col = 3
            for k in range(K):
                M = np.zeros((Nb, 3, 2))                         # -skew(fw)[:, :, :2] dt inside segment k, summed
                w = dtb * (self.seg_of == k)
                M[:, 1, 0], M[:, 2, 0] = -fw[:, 2] * w, fw[:, 1] * w
                M[:, 0, 1], M[:, 2, 1] = fw[:, 2] * w, -fw[:, 0] * w
                np.cumsum(M, axis=0, out=M)
                put_explicit(col, M)
                del M, w
                col += 2
            for j in range(Ka):
                a0, r1, bl = self.support(j)
                if r1 > a0:
                    E = np.cumsum(Rm[a0:r1] * (-bl * wdt[a0:r1])[:, None, None], axis=0)
                    put_local(col, a0, r1, E, None, F, SF, Gs, SG)
                else:
                    A[:, col:col + 3] = 0.0
                col += 3
            # a body-frame gyro bias change turns the frame of all later rows; seen through each segment's own tilt
            for j in range(Kg):
                a0, r1, bl = self.support(j)
                if r1 <= a0:
                    A[:, col:col + 3] = 0.0
                elif K == 1 and not EXPL:
                    Phi = np.cumsum(Rm[a0:r1] * (bl * dtb[a0:r1])[:, None, None], axis=0)
                    E = np.empty_like(Phi)
                    f1, f2, f3 = fw[a0:r1, 0:1], fw[a0:r1, 1:2], fw[a0:r1, 2:3]
                    E[:, 0, :] = f2 * Phi[:, 2, :] - f3 * Phi[:, 1, :]
                    E[:, 1, :] = f3 * Phi[:, 0, :] - f1 * Phi[:, 2, :]
                    E[:, 2, :] = f1 * Phi[:, 1, :] - f2 * Phi[:, 0, :]
                    E *= dtb[a0:r1, None, None]
                    np.cumsum(E, axis=0, out=E)
                    put_local(col, a0, r1, E, Phi[-1].copy(), F, SF, Gs, SG)
                else:
                    b = np.zeros(Nb - a0)
                    b[:r1 - a0] = bl
                    C = np.cumsum(Rg[a0:] * (b * dtb[a0:])[:, None, None], axis=0)
                    sg = seg[a0:]
                    for k in np.unique(sg):
                        mk = sg == k
                        C[mk] = np.matmul(Sm[k], C[mk])
                    M = np.zeros((Nb, 3, 3))
                    f1, f2, f3 = fw[a0:, 0:1], fw[a0:, 1:2], fw[a0:, 2:3]
                    M[a0:, 0, :] = f2 * C[:, 2, :] - f3 * C[:, 1, :]
                    M[a0:, 1, :] = f3 * C[:, 0, :] - f1 * C[:, 2, :]
                    M[a0:, 2, :] = f1 * C[:, 1, :] - f2 * C[:, 0, :]
                    del C
                    M[a0:] *= dtb[a0:, None, None]
                    np.cumsum(M, axis=0, out=M)
                    put_explicit(col, M)
                    del M
                col += 3
            for j, (s, e) in enumerate(self.gaps):
                put_scalar(col, self.ramp(s, e))
                col += 3
            mark("jac")
            pc = np.cumsum(v * dtb[:, None], axis=0)
            pc -= pc.mean(0)
            y = -np.concatenate([sv * box_rows(v - shift, rows, smooth).ravel(), sp * pc[rows].ravel()])
            H = prior()                                          # normal equations: prior plus A'A, in double precision
            rhs = -(H @ vec(x))
            AtA = np.empty_like(H)
            for r0 in range(0, 6 * m, 96):                       # a slab of rows at a time
                a = A[r0:r0 + 96].astype(np.float64)
                np.matmul(a.T, a, out=AtA)
                H += AtA
                rhs += a.T @ y[r0:r0 + 96]
            del AtA
            del a, y, pc, A, F, SF, Gs, SG                      # only H and rhs are needed from here
            mark("ata")
            dH = np.maximum(np.diag(H), 1e-9)
            ok = False
            for _ in range(6):
                try:
                    dx = np.linalg.solve(H + lam * np.diag(dH), rhs)
                except np.linalg.LinAlgError:
                    lam *= 8.0
                    continue
                xn = self.apply(x, dx)
                r = self.forward(xn)
                o = obj(r[0], xn)
                if o < best:
                    x, best = xn, o
                    v, Rg, Rm, fw, Sm = r
                    lam = max(lam / 3.0, 1e-6)
                    ok = True
                    break
                del r
                lam *= 8.0
            mark("trial")
            if it >= 2 and it % 2 == 0:
                # evidence update of the drift stiffness (Fellner-Schall): a calm file gets a stiff curve, a jumpy one a loose one
                try:
                    Hi = np.linalg.inv(H + 1e-9 * np.diag(np.diag(H)))
                except np.linalg.LinAlgError:
                    Hi = None
                cv = vec(x)
                for nm, S_, i0, n_ in ((("a", S_a, ia, 3 * Ka), ("g", S_g, ig, 3 * Kg)) if Hi is not None else ()):
                    c = cv[i0:i0 + n_]
                    gam = (n_ - 3) - alpha[nm] * np.trace(Hi[i0:i0 + n_, i0:i0 + n_] @ S_)
                    new = max(gam, 0.5) / max(c @ S_ @ c, 1e-12)
                    alpha[nm] = float(np.clip(new, alpha[nm] / 8.0, alpha[nm] * 8.0))
                alpha["a"] = float(np.clip(alpha["a"], 1 / 2.0 ** 2, 1 / 0.01 ** 2))
                alpha["g"] = float(np.clip(alpha["g"], 1 / 0.3 ** 2, 1 / 0.001 ** 2))
                del Hi
                best = obj(v, x)
                ok = True
            del H
            if not ok:
                break
        self.x = x
        self.v = v
        return 1 / np.sqrt(alpha["a"]), 1 / np.sqrt(alpha["g"])

    def set_target(self, on, win):
        """Drones: body-frame velocity from rotor drag. In-plane accel, minus what the lever arm and the rotor rate add,
        low-passed at 0.2 Hz and divided by the drag constant. Rows are worked through in pieces with a margin."""
        R, n = self.R, self.n
        fs = R.fs
        onr = on[np.arange(n) // win]
        sel = onr[:self.last + 1]
        if not sel.any():
            return
        ta = np.zeros(3)
        for a in range(0, self.last + 1, CH):
            b = min(self.last + 1, a + CH)
            ta += filled(R, R.ac, a, b)[sel[a:b]].sum(0)
        ta /= np.linalg.norm(ta)
        e1 = np.cross(ta, [0, 1, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ta, e1)
        E = np.stack([e1, e2], 1)
        nb = max(1, int(round(0.05 * fs)))
        MG = 8 * nb
        fp = np.empty((n, 2), np.float32)
        for k0 in range(0, n, CH):
            k1 = min(n, k0 + CH)
            a0, a1 = max(0, k0 - MG), min(n, k1 + MG)
            w = filled(R, R.gy, a0, a1) - self.b0
            wl = box(box(w, nb), nb)
            wd = box(box(np.gradient(box(w, 2 * nb), 1.0 / fs, axis=0), nb), nb)
            a = filled(R, R.ac, a0, a1)
            a -= np.cross(wd, LEVER) + np.cross(wl, np.cross(wl, LEVER)) + np.cross(wl, RATE)
            fp[k0:k1] = (a @ E)[k0 - a0:k1 - a0]
        # 0.2 Hz low-pass on block means (the blocks are far faster than the cut-off)
        fpb = np.add.reduceat(fp[:self.idx[-1] + HOP].astype(np.float64), self.idx, axis=0) / HOP
        del fp
        fpb = lowpass(fpb, 0.2, fs / HOP)
        self.on_b = onr[self.mid]
        off = fpb[self.on_b].mean(0) if self.on_b.any() else np.zeros(2)
        self.vb_t = (-(fpb - off) / K_DRAG) @ E.T

    def solve(self, drone):
        self.drone = drone
        sig_a, sig_g = 0.5, 0.05
        if drone:
            sig_p = 5.0
            for outer in range(4):
                v, Rg, Rm, fw, Sm = self.forward(self.x)
                shift = np.einsum("nij,nj->ni", Rm, self.vb_t) * self.on_b[:, None]
                if outer > 0:
                    pos = np.cumsum(v * self.dtb[:, None], axis=0)
                    sig_p = float(np.clip(1.5 * np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()), 1.0, 20.0))
                    del pos
                del v, Rg, Rm, fw, Sm
                sig_a, sig_g = self.core(shift, 8, sig_a, sig_g, 0.5, sig_p)
        else:
            sig_a, sig_g = self.core(np.zeros((self.Nb, 3)), 12, sig_a, sig_g, 3.0, 5.0)
        self.rate = sig_a

    def attitudes(self):
        """Attitude of the grid rows, piece by piece: gyro with the solved bias, times the tilt of its segment (spread
        across dropouts). Yields (a, b, q) for rows a..b-1."""
        S = self.x["S"]
        Sq = qexp(S)
        for a, b, qg in chain_chunks(self.n, self.t, self.rates(self.x["cg"]), self.q0):
            rq = np.tile(Sq[-1], (b - a, 1))
            for k, (s, e) in enumerate(self.segs):
                if e > a and s < b:
                    rq[max(s, a) - a:min(e, b) - a] = Sq[k]
            for j, (s, e) in enumerate(self.gaps):
                if e > a and s < b:
                    lo, hi = max(s, a), min(e, b)
                    f = (np.arange(lo, hi) - s + 1.0) / (e - s + 1.0)
                    d = qlog(qmul(Sq[j + 1], Sq[j] * np.array([1.0, -1, -1, -1])))
                    rq[lo - a:hi - a] = qmul(qexp(f[:, None] * d[None]), Sq[j][None])
            q = qmul(rq, qg)
            q /= np.sqrt((q * q).sum(1))[:, None]
            yield a, b, q


# ---------------------------------------------------------------- velocity post-filter and the blind tail
def slow_rms(v):
    """RMS of the 5 s running mean of velocity (one axis at a time, to keep temporaries small)."""
    k = int(5.0 / DTV)
    return float(np.sqrt(sum((box(v[:, i], k) ** 2).sum() for i in range(v.shape[1])) / len(v)))


def post_filter(v, mode, tau=0.0):
    """mode 0: high-pass with time constant tau. 1: Wiener gain per octave from this file's own band power (drones).
    2: fixed gain per octave (handheld). Works on the mirrored spectrum of the mean-free velocity."""
    n = len(v)
    N = 2 * n
    mean = v.mean(0)

    def spectrum(i):
        """Real FFT of axis i, mean-free and mirrored (an even signal, so its spectrum is real-symmetric)."""
        x = v[:, i] - mean[i]
        return np.fft.rfft(np.concatenate([x, x[::-1]]))           # n + 1 bins

    f = np.arange(n + 1) / (N * DTV)
    if mode == 0:
        a = 2 * np.pi * f * tau
        gxy = gz = a * a / (1 + a * a)
        del a
    else:
        if mode == 1:
            b = np.searchsorted(EDGES, f, side="right") - 1
            okb = (b >= 0) & (b <= 10)
            P3 = []
            for i in range(3):                                      # band power per axis, one axis at a time
                P = np.abs(spectrum(i)) ** 2
                P3.append(np.bincount(b[okb], P[okb], minlength=11)[:11] / (N * N))
                del P
            del b, okb
            g1 = np.clip(1 - ERR_XY / np.maximum(P3[0] + P3[1], 1e-12), 0.0, 1.0)
            g2 = np.clip(1 - ERR_Z / np.maximum(P3[2], 1e-12), 0.0, 1.0)
        else:
            g1, g2 = GAIN_XY, GAIN_Z
        lc = np.log(np.sqrt(np.maximum(EDGES[:11], 1.0 / 512) * np.minimum(EDGES[1:], 4.0)))
        lf = np.log(np.maximum(f, 1e-6))
        gxy, gz = np.interp(lf, lc, g1), np.interp(lf, lc, g2)
        del lf
        gxy[0] = gz[0] = 0.0
    del f
    out = np.empty((n, 3))
    for i in range(3):                                              # real gains on a real-symmetric spectrum
        X = spectrum(i)
        X *= gz if i == 2 else gxy
        out[:, i] = np.fft.irfft(X, N)[:n]
        del X
    return out


def fit_tail(xk):
    """Handheld: gain (0 to 1) and time constant with which the velocity of the last 0.5 s best predicted this path's own
    next 5 s, over cuts at every earlier second. A file that wanders gets about zero (a hold)."""
    cuts = np.arange(100, len(xk) - 50, max(10, len(xk) // 600))     # every second; at most 600 cuts on long files
    if len(cuts) < 5:
        return 0.0, 0.35
    sj = np.arange(5, 51, 5)
    v = (xk[cuts] - xk[cuts - 5]) * 2.0
    d = np.stack([xk[cuts + j] - xk[cuts] for j in sj], 1)
    best = (np.inf, 0.0, 0.35)
    for tau in (0.35, 0.7, 1.5, 3.0):
        a = tau * (1 - np.exp(-(sj * 0.1) / tau))
        num = (a[None, :] * (d * v[:, None, :]).sum(2)).sum()
        den = (a[None, :] ** 2 * (v ** 2).sum(1)[:, None]).sum()
        G = float(np.clip(num / max(den, 1e-12), 0.0, 1.0))
        err = ((d - G * a[None, :, None] * v[:, None, :]) ** 2).sum()
        if err < best[0]:
            best = (err, G, tau)
    return best[1], best[2]


# ---------------------------------------------------------------- the pipeline for one file
def estimate(data, averaged=False):
    """data: [time in s, six values]; emptied here so the input copy can be freed early.
    Yields (positions (k, 3), quaternions x y z w (k, 4)) for consecutive runs of input rows, in file order."""
    t_in, v_in = data[0], data[1]
    data.clear()
    n_in = len(t_in)

    def still():
        for a in range(0, n_in, CH):
            k = min(n_in, a + CH) - a
            yield np.zeros((k, 3)), np.tile([0.0, 0, 0, 1.0], (k, 1))
    if n_in < 2:
        yield from still()
        return
    # which triplet is the accelerometer: the one whose size is near gravity
    nz = (v_in != 0).any(1) & np.isfinite(v_in).all(1)
    gyro_first = True
    if nz.sum() > 10:
        n1 = float(np.median(np.sqrt((v_in[nz, :3].astype(np.float64) ** 2).sum(1))))
        n2 = float(np.median(np.sqrt((v_in[nz, 3:].astype(np.float64) ** 2).sum(1))))
        gyro_first = abs(n2 - 9.81) <= abs(n1 - 9.81)
    del nz
    mark("read")
    R, gi = make_grid(t_in, v_in)
    gi = gi.astype(np.int32)
    R.gyro_first = gyro_first
    R.gy, R.ac = (slice(0, 3), slice(3, 6)) if gyro_first else (slice(3, 6), slice(0, 3))
    del v_in, t_in
    live = np.nonzero(R.ok)[0]
    if len(live) < 16 or R.t[live[-1]] < 20.0:
        yield from still()                            # too little data to solve: a constant pose
        return
    del live
    mark("grid")
    repair(R, BRIDGE_HAND, spikes=not averaged)       # averaged rows had their spikes removed when they were read
    mark("repair")
    R.live = np.nonzero(R.ok)[0].astype(np.int32)
    on, win = rotor_windows(R)
    last = int(R.live[-1])
    drone = bool(on[np.arange(last + 1) // win].mean() > 0.3)
    if drone:
        bridge(R, BRIDGE_DRONE)
        R.live = np.nonzero(R.ok)[0].astype(np.int32)
    cap_gaps(R, GAPS_MAX)
    g = float(np.median(np.sqrt((R.imu[:, R.ac][R.ok].astype(np.float64) ** 2).sum(1))))
    cal = DRONE_GYRO_CAL if drone else np.zeros(3)
    b0 = cal
    rest = rest_bias(R, g)
    if rest is not None:
        b0 = cal if np.linalg.norm(rest - cal) < 0.02 else rest
    elif drone:
        # no rest period: keep the declared calibration only if the file agrees with it across the up direction
        bw, up, nw = window_bias(R)
        if nw >= 5 and np.linalg.norm(bw - (cal - (cal @ up) * up)) >= 0.03:
            b0 = bw
    mark("platform")
    S = Solver(R, b0, max(8.0, float(R.t[last]) / KNOTS_MAX))
    mark("init")
    if drone:
        S.set_target(on, win)
    mark("target")
    S.solve(drone)
    mark("solve")
    dbg("drone", drone, "b0", np.round(b0, 5), "K", S.K, "G", S.G, "Ka", S.Ka, "rate", round(S.rate, 4))
    # velocity track at 10 Hz, self-check, band gains
    tk = np.arange(S.tb[0], S.tb[-1], DTV)
    vk = np.stack([np.interp(tk, S.tb, S.v[:, i]) for i in range(3)], 1)
    del S.v
    check = slow_rms(vk)
    c0 = 1.0 if drone else 0.5
    fell = check > c0 or (drone and S.rate > RATE_LIMIT and check > 0.6 * c0)
    if fell:
        vf = post_filter(vk, 0, 8.0 if drone else 1.0)
    else:
        vf = post_filter(vk, 1 if drone else 2)
    del vk
    dbg("check", round(check, 3), "fell", fell)
    v0 = vf[-5:].mean(0)
    xk = np.cumsum(vf, axis=0, out=vf)        # positions, in the velocity's own array
    xk *= DTV
    del vf
    # the blind tail: hold plus a damped share of the last velocity
    t_last, x_end = tk[-1], xk[-1].copy()
    if drone:
        gain, tau = (0.0, 0.35) if fell else (0.5, 0.35)
        a0 = max(0, last - int(2.0 * R.fs))
        descend = bool(on[np.arange(a0, last + 1) // win].mean() > 0.5)       # rotors still on: logged flights end by landing
    else:
        gain, tau = fit_tail(xk)
        descend = False
    mark("filter")
    for a, b, q in S.attitudes():
        t = R.t[a:b]
        pos = np.stack([np.interp(t, tk, xk[:, i]) for i in range(3)], 1)
        tl = t > t_last
        if tl.any():
            sdt = t[tl] - t_last
            pt = x_end + gain * tau * (1 - np.exp(-sdt / tau))[:, None] * v0[None]
            if descend:
                pt[:, 2] -= 0.25 * np.minimum(sdt / 3.0, 1.0)
            th = 0.5 * np.pi * np.minimum(sdt / 5.0, 1.0)      # 5 mm curve: a straight tail breaks SE(3) alignment
            pt[:, 0] += 0.005 * (np.cos(th) - 1.0)
            pt[:, 1] += 0.005 * np.sin(th)
            pos[tl] = pt
        lo, hi = np.searchsorted(gi, [a, b])
        if hi > lo:
            idx = gi[lo:hi] - a
            yield pos[idx], q[idx][:, [1, 2, 3, 0]]
    mark("estimate")


def probe(path):
    """What the reader needs to know before it holds any row, without a pass over the whole file: the first 4096 rows
    (first stamp, usual step, whether stamps are whole numbers, bytes per row), the file size, and the last stamps
    (read from the end of the file). Returns (about how many rows, first stamp, last stamp, step, whole numbers)."""
    heads, nbytes, ints, n = [], 0, True, 0
    for c in data_lines(path):
        n += 1
        nbytes += len(c) + 1
        h = c.split(",", 1)[0].strip()
        if ints and not h.lstrip("+-").isdigit():
            ints = False
        v = num(h)
        if v is not None:
            heads.append(v)
        if n >= 4096:
            break
    if len(heads) < 3:
        return n, None, None, 0.0, ints
    size = os.path.getsize(path)
    rows = n if n < 4096 else int(size / max(nbytes / n, 1.0)) + 1
    last = heads[-1]
    if n >= 4096:
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", "replace").splitlines()[1:]
        for c in tail:
            c = c.strip()
            if is_data(c):
                v = num(c.split(",", 1)[0].strip())
                if v is not None:
                    last = max(last, v)
    d = np.diff(np.array([v - heads[0] for v in heads], np.float64))
    step = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    return rows, heads[0], last, step, ints


def poses_of(path, scale, known):
    """Poses of every data row, in pieces. A file faster or longer than the row budget is averaged as it is read, and
    every row takes the pose of its averaging cell."""
    n, first, last, step, _ = known
    k = 1
    if first is not None and step > 0 and last > first:
        k = int(np.ceil(max(n, (last - first) / step + 1) / ROW_BUDGET))
    dbg("about", n, "rows, averaged by", k)
    if k > 1:
        data, counts = read_cells(data_lines(path), first, last, k * step, scale)
        yield from spread(estimate(data, True), counts)
    else:
        yield from estimate(read_rows(data_lines(path), scale))


FMT = "%s %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"


def text(ss, P, Q):
    """Output lines for stamps ss and poses P (k, 3), Q (k, 4), formatted in one go."""
    k = len(ss)
    cells = [None] * (8 * k)
    cells[0::8] = ss
    for j in range(3):
        cells[1 + j::8] = P[:k, j].tolist()
    for j in range(4):
        cells[4 + j::8] = Q[:k, j].tolist()
    return (FMT * k) % tuple(cells)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    path = find_csv(args[0])
    out = args[1] if len(args) > 1 else "estimated_trajectory.txt"
    known = probe(path)
    _, first, last, step, ints = known
    scale = unit(step) if step > 0 else 1e-9
    dbg("seconds", round((last - first) * scale, 1) if first is not None else 0.0)
    lastp, lastq = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    st = stamps(path, bool(ints and scale == 1e-9), scale)    # the input's own rows: one output line for each
    with open(out, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        try:
            for P, Q in poses_of(path, scale, known):
                if not len(P):
                    continue
                bad = ~(np.isfinite(P).all(1) & np.isfinite(Q).all(1))
                if bad.any():
                    P[bad] = lastp
                    Q[bad] = lastq
                ss = list(islice(st, len(P)))
                if ss:
                    f.write(text(ss, P, Q))
                    lastp, lastq = P[len(ss) - 1].copy(), Q[len(ss) - 1].copy()
                if len(ss) < len(P):
                    break
        except Exception as ex:                       # never leave the file short: every input row gets a pose
            sys.stderr.write("main.py: solver stopped (%s: %s); the remaining rows hold the last pose\n" % (type(ex).__name__, ex))
        # exactly one pose per input row, whatever happened above: rows not yet written hold the last pose
        P, Q = np.tile(lastp, (CH, 1)), np.tile(lastq, (CH, 1))
        while True:
            ss = list(islice(st, CH))
            if not ss:
                break
            f.write(text(ss, P, Q))


if __name__ == "__main__":
    main()
