"""Body-frame velocity from rotation: a_b = dv_b/dt + w x v_b  (a_b = kinematic accel in body).
First, the plain steady-turn formula with the sign fixed: v_perp = -(w x a_b) / |w|^2.
Second, a sliding-window regression that needs no bias or tilt knowledge:
    f_k = c + [w_k]x v_b + noise    over a window, unknowns v_b (3) and intercept c (3).
Slow bias and any static gravity component go into c, so slope v_b is immune to them. It needs yaw rate to vary inside the window.
Variants: T = gravity removed with truth attitude (upper bound). R = raw specific force, no attitude at all (fully IMU only).
Optional extra regressors: gravity direction change from gyro inside the window (G variant): uses window-mean f as gravity direction,
propagates it with gyro, subtracts it. numpy only, read-only. TUM rooms 1-5, room 6 stays locked.
"""
import numpy as np, sys

ROOT = "E:/clipping-pipeline/xr_track3/data"
FS = 200.0
RNG = np.random.default_rng(3)


def box(x, n):
    k = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)


def quat_R(q):
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def skew(w):
    return np.stack([np.zeros(len(w)), -w[:, 2], w[:, 1], w[:, 2], np.zeros(len(w)), -w[:, 0], -w[:, 1], w[:, 0], np.zeros(len(w))], 1).reshape(-1, 3, 3)


def so3(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def analyse(name, t, w, f, R, vw, W=4.0, lp=0.25, dec=10, up_axis=None):
    """f raw specific force, w gyro. truth R, vw for scoring and for variant T."""
    g = np.array([0, 0, 9.81])
    n = int(lp * FS)
    w_l = box(w, n)
    f_l = box(f, n)
    aT = f_l - np.einsum("nji,j->ni", R, g)                  # gravity removed with truth
    vb = np.einsum("nji,nj->ni", R, box(vw, n))
    # plain steady-turn formula, sign fixed, truth gravity
    w2 = (w_l * w_l).sum(1)
    est = -np.cross(w_l, aT) / (w2[:, None] + 1e-9)
    wh = w_l / (np.sqrt(w2)[:, None] + 1e-9)
    vperp = vb - (vb * wh).sum(1, keepdims=True) * wh
    line = []
    for th in (0.5, 1.0):
        m = np.sqrt(w2)[::dec] > th
        if m.sum() > 30:
            e_, t_ = est[::dec][m], vperp[::dec][m]
            line.append(f"|w|>{th} share {m.mean():.2f} err {np.sqrt(((e_-t_)**2).sum(1).mean()):.2f}/{np.sqrt((t_**2).sum(1).mean()):.2f} rx {r(e_[:,0],t_[:,0]):.2f} ry {r(e_[:,1],t_[:,1]):.2f}")
    # sliding-window regression
    idx = np.arange(0, len(t), dec)
    Wn = int(W * FS / dec)
    rows = {k: [] for k in ("T", "R", "G")}
    truth, stds = [], []
    for s in range(0, len(idx) - Wn, int(FS / dec)):
        ii = idx[s:s + Wn]
        ww = w_l[ii]
        A = np.zeros((3 * len(ii), 6))
        A[:, :3] = skew(ww).reshape(-1, 3)
        A[:, 3:] = np.tile(np.eye(3), (len(ii), 1))
        lam = 1e-2 * len(ii) * np.diag([0.02, 0.02, 0.02, 0, 0, 0])
        ATA = A.T @ A + lam
        # variant T and R
        for key, y in (("T", aT[ii]), ("R", f_l[ii])):
            sol = np.linalg.solve(ATA, A.T @ y.reshape(-1))
            rows[key].append(sol[:3])
        # variant G: gravity direction fixed from window mean, propagated by gyro from the middle of the window (no truth)
        mid = len(ii) // 2
        dtd = dec / FS
        Rrel = [np.eye(3)]
        for k in range(mid, len(ii) - 1):
            Rrel.append(Rrel[-1] @ so3(ww[k] * dtd))
        Rb = [np.eye(3)]
        for k in range(mid - 1, -1, -1):
            Rb.append(Rb[-1] @ so3(-ww[k] * dtd))
        Rs = np.array(Rb[::-1][:-1] + Rrel)                 # R_mid_k
        fm = np.einsum("nij,nj->ni", Rs, f_l[ii]).mean(0)   # mean specific force in mid frame ~ gravity
        gm = 9.81 * fm / np.linalg.norm(fm)
        gk = np.einsum("nji,j->ni", Rs, gm)                 # gravity in body at k, IMU only
        sol = np.linalg.solve(ATA, A.T @ (f_l[ii] - gk).reshape(-1))
        rows["G"].append(sol[:3])
        truth.append(vb[ii].mean(0))
        stds.append(ww.std(0))
    truth, stds = np.array(truth), np.array(stds)
    out = [f"{name:6s} steady: {' | '.join(line) if line else 'n/a'}"]
    # main yaw axis is the axis with largest gyro spread
    for key in ("T", "G", "R"):
        e = np.array(rows[key])
        for th in (0.0, 0.2, 0.4):
            m = np.linalg.norm(stds, axis=1) > th
            if m.sum() < 15:
                continue
            ev = np.sqrt(((e[m] - truth[m]) ** 2).sum(1).mean())
            zero = np.sqrt((truth[m] ** 2).sum(1).mean())
            sp = r(np.linalg.norm(e[m], axis=1), np.linalg.norm(truth[m], axis=1))
            out.append(f"   {key} std(w)>{th}: share {m.mean():.2f} vecRMS {ev:.2f} vs zero {zero:.2f}; r x {r(e[m,0],truth[m,0]):5.2f} y {r(e[m,1],truth[m,1]):5.2f} z {r(e[m,2],truth[m,2]):5.2f}; r(speed) {sp:5.2f}")
    return out


W = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
sa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
sg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
print(f"window {W}s noise a {sa} g {sg}")
for n in [1, 2, 3, 4, 5]:
    imu = np.loadtxt(f"{ROOT}/tumvi_room{n}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{ROOT}/tumvi_room{n}/mocap0/data.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    w, a = imu[:, 1:4].copy(), imu[:, 4:7].copy()
    if sa:
        a += RNG.normal(0, sa, a.shape)
        w += RNG.normal(0, sg, w.shape)
    p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    v = np.gradient(box(p, 50), 1 / FS, axis=0)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    print("\n".join(analyse(f"room{n}", t, w, a, quat_R(q), v, W=W)))
SEQ = {"MH03": "MH_03_medium", "MH04": "MH_04_difficult", "V102": "V1_02_medium", "V103": "V1_03_difficult", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
for name, d in SEQ.items():
    path = f"{ROOT}/euroc_airio/EuRoC-Dataset/{d}/mav0"
    imu = np.loadtxt(f"{path}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{path}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    bw = np.stack([np.interp(t, tg, gt[:, 11 + i]) for i in range(3)], 1)
    ba = np.stack([np.interp(t, tg, gt[:, 14 + i]) for i in range(3)], 1)
    w, a = imu[:, 1:4] - bw, imu[:, 4:7] - ba
    v = np.stack([np.interp(t, tg, gt[:, 8 + i]) for i in range(3)], 1)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    print("\n".join(analyse(name, t, w, a, quat_R(q), v, W=W)))
