"""EuRoC: which drag-model terms raise the agreement between in-plane specific force and truth velocity?
Truth attitude and truth velocity, rotors-on flight segment only, 20 Hz after anti-alias boxes. numpy only, read-only.
Model of in-plane specific force f_p (2-vector, rotor-plane basis E):
    f_p = -k v_p                                  drag on body velocity (truth v of the IMU frame)
        + E'([w']x + [w]x[w]x) r                  IMU lever arm r from centre of mass (3 unknowns)
        + E'[w]x d                                rotor-hub velocity from body rate, d = k (r_rotor - r_imu) (3 unknowns)
        + m (T - mean T)                          thrust-axis misalignment leakage (2 unknowns)
        + c |v_p| v_p ,  e v_t v_p                quadratic drag, climb coupling (1 + 1 unknowns)
Thrust-scaled drag variants replace k by k*sqrt(T/Tbar) or k*T/Tbar.
Fit pooled, leave-one-sequence-out, on 1/64-1 Hz band-passed signals. Report on held-out file, per octave band:
    corr(v, vd) and gain of truth on vd (team's metric), vd = -(f_p - fitted nuisance terms) / k_eff.
"""
import numpy as np

ROOT = "E:/clipping-pipeline/xr_track3/data/euroc_airio/EuRoC-Dataset"
SEQ = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult", "MH05": "MH_05_difficult",
       "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult", "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
FS, DEC = 200.0, 10
FD = FS / DEC
BANDS = [(1 / 64, 1 / 32), (1 / 32, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1 / 2), (1 / 2, 1.0)]


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


def skew(w):
    z = np.zeros(len(w))
    return np.stack([z, -w[:, 2], w[:, 1], w[:, 2], z, -w[:, 0], -w[:, 1], w[:, 0], z], 1).reshape(-1, 3, 3)


def bp(x, lo, hi):
    x = x - x.mean(0)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / FD)
    X[~((f >= lo) & (f <= hi))] = 0
    return np.fft.irfft(X, n=len(x), axis=0)


def load(name):
    d = f"{ROOT}/{SEQ[name]}/mav0"
    imu = np.loadtxt(f"{d}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{d}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    w, a = imu[:, 1:4], imu[:, 4:7]
    tg = gt[:, 0] * 1e-9
    # rotors on: high-band accel power per 1.28 s window
    N, H = 256, 40
    st = np.arange(0, len(a) - N, H)
    hi = np.array([(np.abs(np.fft.rfft(a[s:s + N] - a[s:s + N].mean(0), axis=0)[64:]) ** 2).sum() for s in st])
    onw = hi > 0.2 * np.median(np.sort(hi)[-len(hi) // 2:])
    on = np.zeros(len(a), bool)
    for s, o in zip(st, onw):
        if o:
            on[s + N // 2 - H // 2:s + N // 2 + H // 2] = True
    # largest contiguous on run inside the truth time span
    inside = (t > tg[0] + 1) & (t < tg[-1] - 1)
    m = on & inside
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    runs = edges.reshape(-1, 2)
    i0, i1 = runs[np.argmax(runs[:, 1] - runs[:, 0])]
    i0, i1 = i0 + int(5 * FS), i1 - int(5 * FS)
    sl = slice(i0, i1)
    v = np.stack([np.interp(t, tg, gt[:, 8 + i]) for i in range(3)], 1)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    bw = np.stack([np.interp(t, tg, gt[:, 11 + i]) for i in range(3)], 1)
    R = quat_R(q)
    vb = np.einsum("nji,nj->ni", R, v)
    wc = w - bw
    a_l = box(box(a, 10), 10)
    w_l = box(box(wc, 10), 10)
    wd = np.gradient(box(wc, 20), 1 / FS, axis=0)
    wd = box(box(wd, 10), 10)
    vb_l = box(box(vb, 10), 10)
    idx = np.arange(i0, i1, DEC)
    a_d, w_d, wd_d, vb_d = a_l[idx], w_l[idx], wd_d if False else wd[idx], vb_l[idx]
    ta = np.median(a_d, 0)
    ta /= np.linalg.norm(ta)
    e1 = np.cross(ta, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ta, e1)
    E = np.stack([e1, e2], 1)                   # 3x2
    fp = a_d @ E
    vp = vb_d @ E
    T = a_d @ ta
    vt = vb_d @ ta
    lever = np.einsum("ij,nik->njk", E, skew(wd_d) + skew(w_d) @ skew(w_d))   # n x 2 x 3
    rate = np.einsum("ij,nik->njk", E, skew(w_d))                               # n x 2 x 3
    return dict(fp=fp, vp=vp, T=T, vt=vt, lever=lever, rate=rate, dur=(i1 - i0) / FS, ta=ta)


def design(D, var):
    """Returns list of regressor arrays (n x 2) and a list of names; drag regressors first, k_eff function."""
    vp, T = D["vp"], D["T"]
    Tn = T / T.mean()
    cols, names = [], []
    if var.get("aniso"):
        cols += [np.c_[-vp[:, 0], 0 * vp[:, 0]], np.c_[0 * vp[:, 1], -vp[:, 1]]]
        names += ["k1", "k2"]
    else:
        s = {"none": 1.0, "sqrt": np.sqrt(Tn), "lin": Tn}[var.get("tscale", "none")]
        cols += [-(vp * (s[:, None] if np.ndim(s) else s))]
        names += ["k"]
    ndrag = len(cols)
    if var.get("lever"):
        for j in range(3):
            cols.append(D["lever"][:, :, j]); names.append(f"r{j}")
    if var.get("rate"):
        for j in range(3):
            cols.append(D["rate"][:, :, j]); names.append(f"d{j}")
    if var.get("leak"):
        cols += [np.c_[T - T.mean(), 0 * T], np.c_[0 * T, T - T.mean()]]
        names += ["m1", "m2"]
    if var.get("quad"):
        sp = np.linalg.norm(vp, axis=1)
        cols.append(-(sp[:, None] * vp)); names.append("c_quad")
    if var.get("climb"):
        cols.append(-(D["vt"][:, None] * vp)); names.append("e_climb")
    return cols, names, ndrag


def fit(files, var):
    A, y = [], []
    for D in files:
        cols, names, nd = design(D, var)
        X = np.stack([bp(c, 1 / 64, 1.0).reshape(-1) for c in cols], 1)
        A.append(X)
        y.append(bp(D["fp"], 1 / 64, 1.0).reshape(-1))
    A, y = np.concatenate(A), np.concatenate(y)
    th = np.linalg.lstsq(A, y, rcond=None)[0]
    return th, names


def evaluate(D, var, th):
    cols, names, nd = design(D, var)
    nuis = sum(th[j] * cols[j] for j in range(nd, len(cols))) if len(cols) > nd else 0 * D["fp"]
    fc = D["fp"] - nuis
    # drag velocity from corrected force: divide by effective k
    vp, T = D["vp"], D["T"]
    Tn = T / T.mean()
    if var.get("aniso"):
        vd = -fc / np.array([th[0], th[1]])
    else:
        s = {"none": 1.0, "sqrt": np.sqrt(Tn), "lin": Tn}[var.get("tscale", "none")]
        keff = th[0] * (s if np.ndim(s) == 0 else s[:, None])
        vd = -fc / keff
    out = []
    for lo, hi in BANDS:
        a_, b_ = bp(vp, lo, hi), bp(vd, lo, hi)
        c = (a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum())
        g = (a_ * b_).sum() / (b_ * b_).sum()
        out.append((c, g))
    lowerr = np.sqrt(((bp(vd, 1 / 64, 1 / 4) - bp(vp, 1 / 64, 1 / 4)) ** 2).sum(1).mean())
    return out, lowerr


data = {n: load(n) for n in SEQ}
print("durations s:", {n: round(D["dur"]) for n, D in data.items()})
VARS = {
    "k only": dict(),
    "k per axis": dict(aniso=True),
    "k*sqrt(T/T0)": dict(tscale="sqrt"),
    "k*T/T0": dict(tscale="lin"),
    "k + lever arm": dict(lever=True),
    "k + rotor-rate drag": dict(rate=True),
    "k + lever + rate": dict(lever=True, rate=True),
    "k/axis + lever + rate": dict(aniso=True, lever=True, rate=True),
    "all (+leak,quad,climb)": dict(aniso=True, lever=True, rate=True, leak=True, quad=True, climb=True),
}
print("bands Hz:", [f"{lo:.3f}-{hi:.3f}" for lo, hi in BANDS])
for vname, var in VARS.items():
    res, errs, ths = [], [], []
    for held in SEQ:
        th, names = fit([data[n] for n in SEQ if n != held], var)
        o, e = evaluate(data[held], var, th)
        res.append(o)
        errs.append(e)
        ths.append(th)
    res = np.array(res)                  # files x bands x 2
    c = res[:, :, 0].mean(0)
    g = res[:, :, 1].mean(0)
    th_m = np.mean(ths, 0)
    print(f"{vname:24s} corr {' '.join(f'{x:5.2f}' for x in c)} | gain {' '.join(f'{x:5.2f}' for x in g)} | err(1/64-1/4 Hz) {np.mean(errs):.3f} | params {dict(zip(names, np.round(th_m, 3)))}")
    if vname in ("k only", "k + lever + rate"):
        print("     per file corr in 1/16-1/8 and 1/4-1/2:", {n: (round(float(r_[2, 0]), 2), round(float(r_[4, 0]), 2)) for n, r_ in zip(SEQ, res)})
        print("     per file gain in 1/32-1/16:", {n: round(float(r_[1, 1]), 2) for n, r_ in zip(SEQ, res)})
