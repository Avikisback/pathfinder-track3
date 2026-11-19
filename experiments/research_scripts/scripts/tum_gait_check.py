"""Quick check on real TUM-VI rooms 1-5 (room 6 is locked and NOT read).
Which gait models predict speed on this handheld rig, and how well does gait give walking direction?
numpy only. Reads the team's data folder read-only. Prints tables; writes nothing else.

Windows: 4 s, hop 1 s. All features come from the band-limited IMU, so slow bias cannot reach them.
Optional noise: white noise on all channels, to test robustness (sigma_a m/s^2, sigma_g rad/s).
"""
import numpy as np, sys

ROOT = "E:/clipping-pipeline/xr_track3/data"
ROOMS = [1, 2, 3, 4, 5]
FS = 200.0
WIN = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
SIG_A = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
SIG_G = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N = int(WIN * FS)
HOP = int(FS)
RNG = np.random.default_rng(0)


def load(n):
    imu = np.loadtxt(f"{ROOT}/tumvi_room{n}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{ROOT}/tumvi_room{n}/mocap0/data.csv", delimiter=",", skiprows=1)
    return imu, gt


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


def bandpass(x, lo, hi, fs=FS):
    """FFT brick-wall band-pass on one window, real input."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    mask = (f >= lo) & (f <= hi)
    if x.ndim == 2:
        X[~mask, :] = 0
    else:
        X[~mask] = 0
    return np.fft.irfft(X, n=len(x), axis=0)


def so3_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def features_room(n):
    imu, gt = load(n)
    t = imu[:, 0] * 1e-9
    w = imu[:, 1:4].copy()
    a = imu[:, 4:7].copy()
    if SIG_A > 0:
        a += RNG.normal(0, SIG_A, a.shape)
        w += RNG.normal(0, SIG_G, w.shape)
    tg = gt[:, 0] * 1e-9
    # GT velocity, smoothed over 0.25 s, on IMU clock, world frame. Skip windows that touch mocap holes.
    pg = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    hole = np.interp(t, tg[1:], (np.diff(tg) > 0.05).astype(float)) > 0
    vg = np.gradient(box(pg, 50), 1 / FS, axis=0)
    qg = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    qg /= np.linalg.norm(qg, axis=1, keepdims=True)
    # gravity direction from a 1 s average of accel, body frame. Up = +a at rest.
    up = box(a, 200)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    R = quat_R(qg)
    win = np.hanning(N)
    fr = np.fft.rfftfreq(4096, 1 / FS)
    gb = (fr >= 1.0) & (fr <= 3.2)
    out = []
    for s in range(100, len(t) - N - 100, HOP):
        e = s + N
        if hole[s:e].any():
            continue
        # truth
        vw = vg[s:e]
        vh = vw - np.outer(vw[:, 2], [0, 0, 1])
        spd = np.linalg.norm(vh, axis=1).mean()
        chord = np.linalg.norm(vh.mean(0))
        vb = np.einsum("nji,nj->ni", R[s:e], vw)     # true velocity in body frame
        # vertical accel from body gravity direction
        u = up[s:e]
        av = np.einsum("ni,ni->n", a[s:e], u)
        av0 = av - av.mean()
        amag = np.linalg.norm(a[s:e], axis=1)
        bp = bandpass(av0, 1.0, 6.0)
        # step frequency: zero-padded PSD peak of vertical accel, harmonic product with first harmonic
        S = np.abs(np.fft.rfft(av0 * win, n=4096)) ** 2
        Sg = S[gb]
        hp = S.copy()
        idx2 = np.minimum(np.round(np.arange(len(S)) * 2).astype(int), len(S) - 1)
        hp = S + 0.5 * S[idx2]
        j = np.argmax(hp[gb])
        f0 = fr[gb][j]
        band_power = Sg.sum()
        wide = S[(fr >= 0.3) & (fr <= 8)].sum()
        peak_ratio = S[gb][j] / (np.median(S[(fr >= 0.3) & (fr <= 8)]) + 1e-12)
        # amplitude at the step frequency, exact Goertzel-like projection
        c = np.exp(-2j * np.pi * f0 * np.arange(N) / FS)
        A1 = 2 * np.abs((av0 * win * c).sum()) / win.sum()
        A2 = 2 * np.abs((av0 * win * c * c).sum()) / win.sum()
        h = A1 / (2 * np.pi * f0) ** 2                 # vertical displacement amplitude, m (half peak to peak)
        h2 = 2 * h                                     # peak-to-peak COM bounce
        # step models on band-limited vertical accel; rms instead of per-step extremes
        rng_a = np.percentile(bp, 98) - np.percentile(bp, 2)
        mean_abs = np.abs(bp).mean()
        rms = bp.std()
        rms_g = bandpass(w[s:e] - w[s:e].mean(0), 1.0, 6.0).std(0)
        rms_g = np.sqrt((rms_g ** 2).sum())
        ah = a[s:e] - av[:, None] * u
        ah0 = ah - ah.mean(0)
        rms_h = np.sqrt((bandpass(ah0, 1.0, 6.0) ** 2).sum(1).mean())
        # band log powers for a tiny regression
        lp = [np.log(S[(fr >= l) & (fr < h_)].sum() + 1e-9) for l, h_ in ((0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3.5), (3.5, 6), (6, 10), (10, 20))]
        # Direction. Window-centred frame from gyro over this short window, so drift cannot matter.
        # Rotate samples into the frame at window centre, keep horizontal part, band-pass around step freq.
        mid = (s + e) // 2
        Rc = [np.eye(3)]
        for k in range(s + 1, e):
            Rc.append(Rc[-1] @ so3_exp(w[k - 1] / FS))
        Rc = np.array(Rc)                                  # R_(s)_(k)
        Rm = Rc[(mid - s)]
        Rk = np.einsum("ij,njk->nik", Rm.T, Rc)            # R_(mid)_(k)
        ac = np.einsum("nij,nj->ni", Rk, a[s:e])
        umid = ac.mean(0)
        umid /= np.linalg.norm(umid)
        acv = ac @ umid
        ach = ac - np.outer(acv, umid)
        # band around step frequency, remove quasi-static remainder
        ach_bp = bandpass(ach - ach.mean(0), max(0.8, f0 * 0.75), f0 * 1.25)
        Cov = ach_bp.T @ ach_bp / len(ach_bp)
        ev, V = np.linalg.eigh(Cov)
        d_pca = V[:, 2]
        aniso = ev[2] / (ev[1] + 1e-12)
        # wide-band PCA as used by Humaine, for comparison
        ach_w = bandpass(ach - ach.mean(0), 0.5, 5.0)
        ev2, V2 = np.linalg.eigh(ach_w.T @ ach_w / len(ach_w))
        d_pca_w = V2[:, 2]
        # quadrature sign from vertical accel (forward leads vertical by about 90 deg at step freq)
        acv_bp = bandpass(acv - acv.mean(), max(0.8, f0 * 0.75), f0 * 1.25)
        cross = np.fft.rfft(ach_bp @ d_pca * win) * np.conj(np.fft.rfft(acv_bp * win))
        csum = cross.sum()
        # truth in the same mid frame, horizontal, window average
        vbm = np.einsum("nij,nj->ni", Rk, vb)
        vbar = vbm.mean(0)
        vbar_h = vbar - (vbar @ umid) * umid
        vs = np.linalg.norm(vbar_h)
        # device fixed-axis azimuth: body axes in the horizontal plane at mid; compare with body-frame direction
        dir_true = vbar_h / (vs + 1e-9)
        def axial(d):
            d = d - (d @ umid) * umid
            d /= np.linalg.norm(d) + 1e-12
            c_ = np.clip(abs(d @ dir_true), 0, 1)
            return np.degrees(np.arccos(c_))
        ax_pca = axial(d_pca)
        ax_pca_w = axial(d_pca_w)
        sg = np.sign(d_pca @ dir_true)
        out.append(dict(spd=spd, chord=chord, f0=f0, A1=A1, A2=A2, h=h2, rng=rng_a, mean_abs=mean_abs, rms=rms, rms_g=rms_g, rms_h=rms_h, pr=peak_ratio,
                        bandfrac=band_power / (wide + 1e-9), lp=lp, ax_pca=ax_pca, ax_pca_w=ax_pca_w, aniso=aniso, sign_true=sg,
                        cre=csum.real, cim=csum.imag, vs=vs, dir_true=dir_true, umid=umid, Rmid_up=up[mid]))
    return out


def fit_eval(rows, name, F):
    """Leave-one-room-out OLS. F(row) -> feature vector; intercept added inside. Returns pooled R2, RMSE, per-room corr."""
    yp, yt, ri = [], [], []
    for held in ROOMS:
        tr = [x for r_ in ROOMS if r_ != held for x in rows[r_]]
        te = rows[held]
        Xtr = np.array([np.r_[1, F(x)] for x in tr])
        ytr = np.array([x["spd"] for x in tr])
        lam = 1e-3 * np.eye(Xtr.shape[1])
        lam[0, 0] = 0
        b = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ ytr)
        Xte = np.array([np.r_[1, F(x)] for x in te])
        yp.append(Xte @ b)
        yt.append(np.array([x["spd"] for x in te]))
    yall, pall = np.concatenate(yt), np.concatenate(yp)
    ss = ((yall - yall.mean()) ** 2).sum()
    # baseline: mean of training rooms, LORO
    base = np.concatenate([np.full(len(rows[h]), np.mean([x["spd"] for r_ in ROOMS if r_ != h for x in rows[r_]])) for h in ROOMS])
    r2 = 1 - ((yall - pall) ** 2).sum() / ((yall - base) ** 2).sum()
    rho = np.corrcoef(yall, pall)[0, 1] if pall.std() > 0 else float("nan")
    per = [np.corrcoef(a_, b_)[0, 1] if b_.std() > 0 else float("nan") for a_, b_ in zip(yt, yp)]
    print(f"{name:34s} R2 {r2:6.3f}  r {rho:5.2f}  RMSE {np.sqrt(((yall-pall)**2).mean()):5.3f}  MAE {np.abs(yall-pall).mean():5.3f}  per-room r {' '.join(f'{p:5.2f}' for p in per)}")


rows = {n: features_room(n) for n in ROOMS}
allr = [x for n in ROOMS for x in rows[n]]
S = np.array([x["spd"] for x in allr])
print(f"windows {len(allr)}  win {WIN}s  noise a {SIG_A} g {SIG_G}  speed mean {S.mean():.3f} std {S.std():.3f}  baseline RMSE (mean) {S.std():.3f}")
print("step freq Hz percentiles 10/50/90:", np.percentile([x['f0'] for x in allr], [10, 50, 90]).round(2), " speed>0.5 share", round(float((S > 0.5).mean()), 2))

L = 1.0  # inverted-pendulum length guess, m; per-model scale is fit anyway
models = {
    "const (no features)": lambda x: np.array([]),
    "f (linear in frequency)": lambda x: np.array([x["f0"]]),
    "f^2 (walk ratio form)": lambda x: np.array([x["f0"] ** 2]),
    "Weinberg f*rng^0.25": lambda x: np.array([x["f0"] * x["rng"] ** 0.25]),
    "Kim f*mean|a|^(1/3)": lambda x: np.array([x["f0"] * x["mean_abs"] ** (1 / 3)]),
    "Zijlstra f*sqrt(2Lh-h^2)": lambda x: np.array([x["f0"] * 2 * np.sqrt(max(2 * L * x["h"] - x["h"] ** 2, 0))]),
    "Tian f*h^0.5 (=rng form)": lambda x: np.array([x["f0"] * np.sqrt(x["h"])]),
    "Shin/Ladetto f, var": lambda x: np.array([x["f0"], x["rms"] ** 2]),
    "Vezocnik f,r^0.1 (times f)": lambda x: np.array([x["f0"] ** 2, x["f0"] * x["rng"] ** 0.1]),
    "rms vertical": lambda x: np.array([x["rms"]]),
    "log rms vertical": lambda x: np.array([np.log(x["rms"])]),
    "log rms v, log rms h, log rms g": lambda x: np.array([np.log(x["rms"]), np.log(x["rms_h"]), np.log(x["rms_g"])]),
    "f + log rms v,h,g": lambda x: np.array([x["f0"], np.log(x["rms"]), np.log(x["rms_h"]), np.log(x["rms_g"])]),
    "peak ratio + band fraction": lambda x: np.array([np.log(x["pr"]), x["bandfrac"]]),
    "f + logrms + band fraction + pr": lambda x: np.array([x["f0"], np.log(x["rms"]), np.log(x["rms_h"]), np.log(x["rms_g"]), x["bandfrac"], np.log(x["pr"])]),
    "8 log-band powers": lambda x: np.array(x["lp"]),
    "8 bands + f + logrms v,h,g": lambda x: np.r_[x["lp"], x["f0"], np.log(x["rms"]), np.log(x["rms_h"]), np.log(x["rms_g"])],
}
print("\nSPEED, leave-one-room-out, target = mean horizontal speed in window")
for k, F in models.items():
    fit_eval(rows, k, F)

# Direction, moving windows only
mov = [x for x in allr if x["vs"] > 0.5]
ax = np.array([x["ax_pca"] for x in mov]); axw = np.array([x["ax_pca_w"] for x in mov]); aniso = np.array([x["aniso"] for x in mov])
print(f"\nDIRECTION, windows with average horizontal velocity above 0.5 m/s: {len(mov)} of {len(allr)}")
print("axial error deg, step-band PCA  median/75/90:", np.percentile(ax, [50, 75, 90]).round(1), " wide-band PCA:", np.percentile(axw, [50, 75, 90]).round(1))
for th in (1.5, 3, 5, 10):
    m = aniso > th
    print(f"   anisotropy > {th:4.1f}: share {m.mean():.2f}, step-band median/75 {np.percentile(ax[m], 50):.1f} {np.percentile(ax[m], 75):.1f}" if m.sum() else "")
# fixed direction baseline: per-room mean body-frame direction (oracle, in body axes, using truth in body frame)
for name, sel in (("oracle fixed body axis, per room", None),):
    errs = []
    for n in ROOMS:
        m_ = [x for x in rows[n] if x["vs"] > 0.5]
        D = np.array([x["dir_true"] for x in m_])
        C = D.T @ D
        ev, V = np.linalg.eigh(C)
        ax0 = V[:, 2]
        for x in m_:
            d = ax0 - (ax0 @ x["umid"]) * x["umid"]
            d /= np.linalg.norm(d) + 1e-12
            errs.append(np.degrees(np.arccos(np.clip(abs(d @ x["dir_true"]), 0, 1))))
    print(name, "axial error median/75/90:", np.percentile(errs, [50, 75, 90]).round(1))
# concentration of signed direction per room
for n in ROOMS:
    m_ = [x for x in rows[n] if x["vs"] > 0.5]
    D = np.array([x["dir_true"] for x in m_])
    print(f"   room {n}: signed concentration |mean dir| {np.linalg.norm(D.mean(0)):.2f}, mean dir {D.mean(0).round(2)}")
# sign rule from quadrature: learned sign of (cim) relation with truth
sgn_t = np.array([x["sign_true"] for x in mov])
cim = np.array([x["cim"] for x in mov]); cre = np.array([x["cre"] for x in mov])
for name, val in (("Im cross(horiz,vert)", cim), ("Re cross(horiz,vert)", cre)):
    # d_pca sign is arbitrary, so test whether sign(val) agrees with sign_true in a consistent way
    agree = np.mean(np.sign(val) == sgn_t)
    print(f"sign test {name}: agreement {agree:.2f} (or {1-agree:.2f} with opposite convention)")
