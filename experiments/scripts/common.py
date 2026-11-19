"""Shared harness for the Track 3 experiments.

Everything here reads only IMU rows at estimate time. Ground truth is used only by label code and
by the scoring functions (names say so).
"""
import os
import random
import subprocess
import tempfile

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

ROOT = "/mnt/e/clipping-pipeline/xr_track3"
EXP = ROOT + "/experiments"
CACHE = EXP + "/cache"
TMP = EXP + "/tmp"
DEV_ROOMS = [1, 2, 3, 4, 5]
FINAL_ROOM = 6  # locked away until the end. Chosen before any of its data was looked at.
ALL_ROOMS = DEV_ROOMS + [FINAL_ROOM]


# ------------------------------------------------------------------ loading
def _read_csv_ns(path):
    ts, rows = [], []
    with open(path) as f:
        for line in f:
            if line[0] == "#" or not line.strip():
                continue
            c = line.strip().split(",")
            ts.append(int(c[0]))
            rows.append([float(x) for x in c[1:]])
    return np.array(ts, dtype=np.int64), np.array(rows, dtype=np.float64)


EUROC = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult",
         "MH05": "MH_05_difficult", "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult",
         "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
ENV = {"MH": ["MH01", "MH02", "MH03", "MH04", "MH05"], "V1": ["V101", "V102", "V103"], "V2": ["V201", "V202", "V203"]}


def euroc_dir(name):
    return f"{ROOT}/data/euroc_airio/EuRoC-Dataset/{EUROC[name]}/mav0"


def load_room(n):
    """n: int TUM-VI room or EuRoC short name. Returns t_ns, imu (gyro, accel), gt_t_ns, gt_p, gt_q (x y z w)."""
    os.makedirs(CACHE, exist_ok=True)
    if isinstance(n, str):
        c = f"{CACHE}/{n}.npz"
        if os.path.exists(c):
            d = np.load(c)
            return {k: d[k] for k in d.files}
        t, imu = _read_csv_ns(euroc_dir(n) + "/imu0/data.csv")
        gt_t, gt = _read_csv_ns(euroc_dir(n) + "/state_groundtruth_estimate0/data.csv")
        out = dict(t_ns=t, imu=imu, gt_t_ns=gt_t, gt_p=gt[:, :3], gt_q=gt[:, [4, 5, 6, 3]], gt_bw=gt[:, 10:13])
        np.savez(c, **out)
        return out
    c = f"{CACHE}/room{n}.npz"
    if os.path.exists(c):
        d = np.load(c)
        return {k: d[k] for k in d.files}
    t, imu = _read_csv_ns(f"{ROOT}/data/tumvi_room{n}/imu0/data.csv")
    gt_t, gt = _read_csv_ns(f"{ROOT}/data/tumvi_room{n}/mocap0/data.csv")
    out = dict(t_ns=t, imu=imu, gt_t_ns=gt_t, gt_p=gt[:, :3], gt_q=gt[:, [4, 5, 6, 3]])
    np.savez(c, **out)
    return out


# ------------------------------------------------------------------ degradation
SCHEDULES = {
    # Same numbers as tools/degrade.py.
    "full": dict(sg=(0.002, 0.008), sa=(0.02, 0.08), bg=0.02, ba=0.2, gaps=3, gap_s=2.0),
    # Our guess at a gentler organiser script: noise x2, bias at a quarter, gaps kept.
    "mild": dict(sg=(0.002, 0.002), sa=(0.02, 0.02), bg=0.005, ba=0.05, gaps=3, gap_s=2.0),
    # Nothing but the 5 s failure.
    "clean": dict(sg=(0.0, 0.0), sa=(0.0, 0.0), bg=0.0, ba=0.0, gaps=0, gap_s=0.0),
}


def degrade(room, seed, mode="delete", schedule="full", fail_seconds=5.0):
    """Exact in-memory copy of tools/degrade.py at schedule='full' (check_degrade verifies this).

    Returns t_ns, imu, gap_starts_ns. CSV rounding to 6 decimals is reproduced.
    Zero mode: gap rows stay, all zero. Delete mode: gap rows removed.
    """
    s = SCHEDULES[schedule]
    rnd = random.Random(seed)
    t, imu = room["t_ns"], room["imu"]
    t0, t1 = int(t[0]), int(t[-1]) - int(fail_seconds * 1e9)
    span = t1 - t0
    n_gaps = s["gaps"]
    starts = sorted(t0 + int(span * (0.3 + 0.7 * (i + rnd.random() * 0.8) / n_gaps)) for i in range(n_gaps)) if n_gaps else []
    gyro_dir = [rnd.uniform(-1, 1) for _ in range(3)]
    acc_dir = [rnd.uniform(-1, 1) for _ in range(3)]
    out_t, out_v = [], []
    gap_ns = int(s["gap_s"] * 1e9)
    for k in range(len(t)):
        tk = int(t[k])
        if tk > t1:
            break
        in_gap = any(st <= tk < st + gap_ns for st in starts)
        if in_gap and mode == "delete":
            continue
        if in_gap:
            out_t.append(tk)
            out_v.append([0.0] * 6)
            continue
        x = (tk - t0) / span
        sg = s["sg"][0] + s["sg"][1] * x
        sa = s["sa"][0] + s["sa"][1] * x
        v = imu[k]
        if sg > 0:
            w = [v[i] + rnd.gauss(0, sg) + s["bg"] * x * gyro_dir[i] for i in range(3)]
            acc = [v[i + 3] + rnd.gauss(0, sa) + s["ba"] * x * acc_dir[i] for i in range(3)]
        else:
            w, acc = list(v[:3]), list(v[3:6])
        out_t.append(tk)
        out_v.append([round(q, 6) for q in w + acc])
    return np.array(out_t, dtype=np.int64), np.array(out_v), np.array(starts, dtype=np.int64)


def write_csv(path, t, imu):
    with open(path, "w") as f:
        f.write("#timestamp [ns],wx,wy,wz,ax,ay,az\n")
        for tk, v in zip(t, imu):
            f.write(f"{int(tk)}," + ",".join(f"{q:.6f}" for q in v) + "\n")


def check_degrade(n=1, seed=2):
    """Compares degrade() with the real tool, value by value."""
    src = f"{ROOT}/data/tumvi_room{n}/imu0/data.csv"
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        for mode in ("delete", "zero"):
            p = f"{td}/o.csv"
            subprocess.run(["python3", f"{ROOT}/tools/degrade.py", src, p, "--seed", str(seed), "--gap-mode", mode],
                           check=True, stdout=subprocess.DEVNULL)
            t2, v2 = _read_csv_ns(p)
            t1, v1, _ = degrade(load_room(n), seed, mode)
            assert np.array_equal(t1, t2), "timestamps differ"
            d = np.abs(v1 - v2).max()
            print(f"degrade check {mode}: rows {len(t1)} max abs diff {d:.2e}")


# ------------------------------------------------------------------ regular grid
def to_grid(t_ns, imu):
    """Puts rows on a grid that re-inserts deleted rows. Returns t (s from first row), imu, valid, row_id.

    row_id[k] is the index of the input row for grid row k, or -1 for an inserted row.
    """
    dt = np.diff(t_ns)
    med = int(np.median(dt))
    ts, vs, ok, ids = [], [], [], []
    for k in range(len(t_ns)):
        if k > 0 and dt[k - 1] > med * 3 // 2:
            n = int(round(dt[k - 1] / med)) - 1
            for i in range(1, n + 1):
                ts.append(t_ns[k - 1] + (dt[k - 1] * i) // (n + 1))
                vs.append(np.zeros(6))
                ok.append(False)
                ids.append(-1)
        ts.append(t_ns[k])
        # a row is unusable if it is all zero, not finite, or an exact repeat of the row before (a frozen sensor);
        # real rows never repeat all six values exactly
        good = bool(np.isfinite(imu[k]).all()) and not (imu[k] == 0).all() and not (k > 0 and (imu[k] == imu[k - 1]).all())
        vs.append(imu[k] if good else np.zeros(6))
        ok.append(good)
        ids.append(k)
    return np.array(ts, dtype=np.int64), np.array(vs), np.array(ok), np.array(ids)


# ------------------------------------------------------------------ small quaternion helpers (w x y z in numba)
@njit(cache=True)
def _qmul(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


@njit(cache=True)
def _qexp(r):
    a = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if a < 1e-12:
        q = np.array([1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]])
        return q / np.sqrt(np.sum(q * q))
    s = np.sin(a * 0.5) / a
    return np.array([np.cos(a * 0.5), r[0] * s, r[1] * s, r[2] * s])


@njit(cache=True)
def _up_in_body(q):
    # third row of R(q): world z seen in the body frame
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


@njit(cache=True)
def tilt_filter(t, gyro, accel, valid, q0, bias, kp, boost_gain, g, tau_rate):
    """Baseline attitude (src/main.cpp step_valid/step_coast) as a forward pass. Returns q (w x y z) per row."""
    n = len(t)
    out = np.zeros((n, 4))
    q = q0.copy()
    out[0] = q
    rate = np.zeros(3)
    boost = 0.0
    for k in range(1, n):
        dt = (t[k] - t[k - 1])
        if valid[k]:
            w = gyro[k] - bias
            a = accel[k]
            am = np.sqrt(np.sum(a * a))
            corr = np.zeros(3)
            if abs(am - g) < 1.0:
                up = _up_in_body(q)
                u = a / am
                corr = np.array([u[1] * up[2] - u[2] * up[1], u[2] * up[0] - u[0] * up[2], u[0] * up[1] - u[1] * up[0]]) * (kp + boost)
            boost *= np.exp(-dt / 1.0)
            q = _qmul(q, _qexp((w + corr) * dt))
            q /= np.sqrt(np.sum(q * q))
            rate += (w - rate) * (dt / (0.1 + dt))
        else:
            boost = boost_gain
            rate *= np.exp(-dt / tau_rate)
            q = _qmul(q, _qexp(rate * dt))
            q /= np.sqrt(np.sum(q * q))
        out[k] = q
    return out


@njit(cache=True)
def _integrate_conj(t, gyro, valid, end):
    """Integrates the gyro over rows 0..end and returns the conjugate of the result (w x y z)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(1, end):
        if valid[k]:
            q = _qmul(q, _qexp(gyro[k] * (t[k] - t[k - 1])))
    q /= np.sqrt(np.sum(q * q))
    return np.array([q[0], -q[1], -q[2], -q[3]])


def from_two_vectors_wxyz(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = a.dot(b)
    if d < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    c = np.cross(a, b)
    q = np.array([1 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


def quiet_init(t, imu, valid):
    """Same idea as survey()+initial_attitude(): quietest second gives up direction and start attitude."""
    n = len(t)
    fs = int(round(1.0 / np.median(np.diff(t))))
    best, bk, ba = 1e300, -1, None
    for s in range(0, n - fs, fs):
        e = s + fs
        m = valid[s:e]
        if m.sum() < 0.8 * fs:
            continue
        w, a = imu[s:e, :3][m], imu[s:e, 3:][m]
        var_w = (w ** 2).sum(1).mean() - (w.mean(0) ** 2).sum()
        var_a = (a ** 2).sum(1).mean() - (a.mean(0) ** 2).sum()
        off = np.linalg.norm(a.mean(0)) - 9.81
        score = var_w + 0.01 * var_a + 0.01 * off * off
        if score < best:
            best, bk, ba = score, s + fs // 2, a.mean(0)
    if ba is None:
        ba = np.array([0, 0, 9.81])
        bk = 0
    q_up = from_two_vectors_wxyz(ba, np.array([0.0, 0.0, 1.0]))
    q_c = _integrate_conj(t, imu[:, :3], valid, bk)
    qq = _qmul(q_up, q_c)
    return qq / np.linalg.norm(qq), float(np.linalg.norm(ba))


def baseline_attitude(t, imu, valid, kp=0.2, boost=2.0, tau_rate=0.5):
    q0, g = quiet_init(t, imu, valid)
    return tilt_filter(t, imu[:, :3], imu[:, 3:], valid, q0, np.zeros(3), kp, boost, g, tau_rate)


# ------------------------------------------------------------------ scoring (ground truth only here)
def associate(gt_t, est_t, max_diff=0.01):
    """evo-style: nearest estimate for each ground-truth stamp, within max_diff seconds."""
    idx = np.searchsorted(est_t, gt_t)
    idx = np.clip(idx, 1, len(est_t) - 1)
    left = est_t[idx - 1]
    right = est_t[idx]
    idx = idx - (gt_t - left < right - gt_t)
    keep = np.abs(est_t[idx] - gt_t) < max_diff
    return np.nonzero(keep)[0], idx[keep]


def umeyama(est, gt):
    me, mg = est.mean(0), gt.mean(0)
    H = (gt - mg).T @ (est - me) / len(est)
    u, _, vt = np.linalg.svd(H)
    s = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ s @ vt
    return r, mg - r @ me


def rpe(gt_p, gt_R, est_p, est_R, delta):
    """evo RPE: E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j). Returns translation RMSE (m), rotation RMSE (deg)."""
    if len(gt_p) <= delta:
        return np.nan, np.nan
    # evo default: consecutive pairs, not all pairs
    i = np.arange(0, len(gt_p) - delta, delta)
    j = i + delta
    Qrel = gt_R[i].inv() * gt_R[j]
    Prel = est_R[i].inv() * est_R[j]
    qt = gt_R[i].inv().apply(gt_p[j] - gt_p[i])
    pt = est_R[i].inv().apply(est_p[j] - est_p[i])
    et = Qrel.inv().apply(pt - qt)
    er = Qrel.inv() * Prel
    return float(np.sqrt((et ** 2).sum(1).mean())), float(np.degrees(np.sqrt((er.magnitude() ** 2).mean())))


def score(est_t, est_p, est_q_xyzw, gt_t, gt_p, gt_q_xyzw, rpe_frames=None):
    """All numbers evo would give (checked against evo in check_evo)."""
    if rpe_frames is None:  # 1 s of ground truth, whatever its rate (V101 is 20 Hz)
        rpe_frames = int(round(1.0 / np.median(np.diff(gt_t))))
    gi, ei = associate(gt_t, est_t)
    g, e = gt_p[gi], est_p[ei]
    Rg, Re = R.from_quat(gt_q_xyzw[gi]), R.from_quat(est_q_xyzw[ei])
    # SE(3). A constant estimate has no defined rotation, the optimum is just the centroid.
    if np.ptp(e, axis=0).max() < 1e-9:
        ate_se3 = float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))
    else:
        r, tr = umeyama(e, g)
        ate_se3 = float(np.sqrt((((r @ e.T).T + tr - g) ** 2).sum(1).mean()))
    # --align_origin: T = T_gt0 * T_est0^-1
    to = Rg[0] * Re[0].inv()
    eo = to.apply(e - e[0]) + g[0]
    ate_org = float(np.sqrt(((eo - g) ** 2).sum(1).mean()))
    err_rot = Rg.inv() * (to * Re)
    att_org = float(np.degrees(np.sqrt((err_rot.magnitude() ** 2).mean())))
    # tilt part of attitude error
    ug, ue = Rg.inv().apply([0, 0, 1.0]), (to * Re).inv().apply([0, 0, 1.0])
    tilt = np.degrees(np.arccos(np.clip((ug * ue).sum(1), -1, 1)))
    rpe_t, rpe_r = rpe(g, Rg, e, Re, rpe_frames)
    rpe1_t, rpe1_r = rpe(g, Rg, e, Re, 1)   # evo's default delta
    return dict(ate_se3=ate_se3, ate_org=ate_org, att_org=att_org, tilt=float(np.sqrt((tilt ** 2).mean())),
                rpe_t=rpe_t, rpe_r=rpe_r, rpe1_t=rpe1_t, rpe1_r=rpe1_r)


def score_tail(est_t, est_p, gt_t, gt_p, t_fail):
    """Blind 5 s: (a) align on pre-failure poses, RMSE on the tail; (b) relative to the true pose at failure."""
    gi, ei = associate(gt_t, est_t)
    g, e, t = gt_p[gi], est_p[ei], gt_t[gi]
    pre, tail = t <= t_fail, t > t_fail
    if tail.sum() < 10 or pre.sum() < 10:
        return dict(blind_aligned=np.nan, blind_rel=np.nan, blind_end_rel=np.nan)
    if np.ptp(e[pre], axis=0).max() < 1e-9:
        r, tr = np.eye(3), g[pre].mean(0) - e[pre].mean(0)
    else:
        r, tr = umeyama(e[pre], g[pre])
    al = (r @ e.T).T + tr
    a = float(np.sqrt(((al[tail] - g[tail]) ** 2).sum(1).mean()))
    # relative to failure: both start at zero at the last pre-failure pose, same rotation r
    k = np.nonzero(pre)[0][-1]
    rel_e = al[tail] - al[k]
    rel_g = g[tail] - g[k]
    d = np.sqrt(((rel_e - rel_g) ** 2).sum(1))
    return dict(blind_aligned=a, blind_rel=float(np.sqrt((d ** 2).mean())), blind_end_rel=float(d[-1]))


def write_tum(path, t_ns, p, q_xyzw):
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for k in range(len(t_ns)):
            ns = int(t_ns[k])
            f.write(f"{ns // 10**9}.{ns % 10**9:09d} {p[k,0]:.6f} {p[k,1]:.6f} {p[k,2]:.6f} "
                    f"{q_xyzw[k,0]:.9f} {q_xyzw[k,1]:.9f} {q_xyzw[k,2]:.9f} {q_xyzw[k,3]:.9f}\n")


def parse_tum(path):
    t, p, q = [], [], []
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split()
            sec, frac = c[0].split(".")
            t.append(int(sec) * 10**9 + int(frac.ljust(9, "0")[:9]))
            p.append([float(x) for x in c[1:4]])
            q.append([float(x) for x in c[4:8]])
    return np.array(t, dtype=np.int64), np.array(p), np.array(q)


def run_baseline(t, imu, extra=()):
    """Runs the real baseline binary on a CSV of these rows. Returns its output."""
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        i, o = f"{td}/i.csv", f"{td}/o.txt"
        write_csv(i, t, imu)
        subprocess.run([f"{ROOT}/pathfinder", i, o, *extra], check=True, stderr=subprocess.DEVNULL)
        return parse_tum(o)


def gt_seconds(room):
    return room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"]


def fmt(d, keys=("ate_se3", "ate_org", "att_org", "tilt", "rpe_t", "rpe_r")):
    return " ".join(f"{k}={d[k]:.3f}" for k in keys if k in d)
