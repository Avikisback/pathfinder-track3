"""Third round of quick checks. numpy only, read-only. TUM rooms 1-5 only (room 6 locked).
A. Turn-based speed: body-frame dynamics a_b = dv_b/dt + w x v_b. When turning steadily, v_b (perp to w) ~ (w x a_b)/|w|^2. Bias enters as b/|w|.
   Tested with truth attitude for gravity removal (an upper bound), and with accel bias 0.2 m/s^2 added.
B. Direction per room: gait PCA on 2 s windows, on straight fast windows.
C. Cadence quality features added to speed regression.
D. EuRoC: spectrum shift between low and high thrust, partial correlations with climb rate and with speed^2, yaw per knot window.
"""
import numpy as np

ROOT = "E:/clipping-pipeline/xr_track3/data"
ROOMS = [1, 2, 3, 4, 5]
FS = 200.0


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


def partial_r(a, b, c):
    A = np.c_[np.ones_like(c), c]
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return r(ra, rb)


def turn_speed(t, w, a, R, v_w, name, dec=20, lp_s=1.0, bias=np.zeros(3)):
    """All at 200 Hz. Truth R body to world. Returns metrics of v_b est vs truth for |w_perp part| above thresholds."""
    g = np.array([0, 0, 9.81])
    ab = a + bias - np.einsum("nji,j->ni", R, g)          # kinematic accel in body, gravity removed with truth attitude
    n = int(lp_s * FS)
    ab_l = box(ab, n)
    w_l = box(w, n)
    vb = np.einsum("nji,nj->ni", R, box(v_w, n))
    w2 = (w_l * w_l).sum(1)
    est = np.cross(w_l, ab_l) / (w2[:, None] + 1e-9)     # v perpendicular to w
    wn = np.sqrt(w2)
    what = w_l / (wn[:, None] + 1e-9)
    vperp = vb - (vb * what).sum(1, keepdims=True) * what
    out = []
    for th in (0.3, 0.5, 1.0, 1.5):
        m = wn[::dec] > th
        if m.sum() < 30:
            out.append(f"|w|>{th}: n/a")
            continue
        e = est[::dec][m]
        tv = vperp[::dec][m]
        err = np.sqrt(((e - tv) ** 2).sum(1).mean())
        sp_true = np.linalg.norm(tv, axis=1)
        sp_est = np.linalg.norm(e, axis=1)
        out.append(f"|w|>{th} share {m.mean():.2f} vec err {err:.2f} vs |v| {np.sqrt((sp_true**2).mean()):.2f}, r(speed) {r(sp_true, sp_est):.2f}, corr x {r(tv[:,0],e[:,0]):.2f} y {r(tv[:,1],e[:,1]):.2f}")
    print(f"  {name}: " + " | ".join(out))


print("A. TUM turn-based speed, truth attitude, 1 s low-pass")
tum = {}
for n in ROOMS:
    imu = np.loadtxt(f"{ROOT}/tumvi_room{n}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{ROOT}/tumvi_room{n}/mocap0/data.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    w, a = imu[:, 1:4], imu[:, 4:7]
    p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    v = np.gradient(box(p, 50), 1 / FS, axis=0)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_R(q)
    tum[n] = (t, w, a, R, v)
    turn_speed(t, w, a, R, v, f"room {n}")
    turn_speed(t, w, a, R, v, f"room {n} +bias 0.2", bias=np.array([0.2, 0.0, 0.0]))

print("\nB. TUM gait PCA per room, 2 s windows, gyro-compensated, band 0.75-1.25 of cadence. fast = speed>0.8, straight = chord/path>0.9")


def bp(x, lo, hi):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    X[~((f >= lo) & (f <= hi))] = 0
    return np.fft.irfft(X, n=len(x), axis=0)


def so3(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


N = 400
fr = np.fft.rfftfreq(4096, 1 / FS)
gb = (fr >= 1.2) & (fr <= 3.0)
speed_rows = {n: [] for n in ROOMS}
for n in ROOMS:
    t, w, a, R, v = tum[n]
    up = box(a, 200)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    av = (a * up).sum(1)
    res = []
    win = np.hanning(N)
    for s in range(100, len(t) - N - 100, 100):
        e = s + N
        vh = v[s:e].copy()
        vh[:, 2] = 0
        path = np.linalg.norm(vh, axis=1).mean()
        chord = np.linalg.norm(vh.mean(0))
        seg = av[s:e] - av[s:e].mean()
        S = np.abs(np.fft.rfft(seg * win, n=4096)) ** 2
        f0 = fr[gb][np.argmax(S[gb])]
        b = bp(seg, 0.8, 5.0)
        ac = np.correlate(b, b, "full")[len(b) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(0.28 * FS), int(1.0 * FS)
        j = lo + np.argmax(ac[lo:hi])
        f_ac, acp = FS / j, ac[j]
        ahv = a[s:e] - av[s:e, None] * up[s:e]
        rms_v, rms_h = b.std(), np.sqrt((bp(ahv - ahv.mean(0), 1, 6) ** 2).sum(1).mean())
        ww = w[s:e]
        rms_g = np.sqrt((bp(ww - ww.mean(0), 1, 6) ** 2).sum(1).mean())
        yaw = np.abs((ww * up[s:e]).sum(1).mean())
        speed_rows[n].append((path, f_ac, acp, rms_v, rms_h, rms_g, f0, yaw))
        if path < 0.8 or chord / (path + 1e-9) < 0.9:
            continue
        Rc = [np.eye(3)]
        for k in range(s + 1, e):
            Rc.append(Rc[-1] @ so3(w[k - 1] / FS))
        Rc = np.array(Rc)
        Rm = Rc[N // 2]
        Rk = np.einsum("ij,njk->nik", Rm.T, Rc)
        ac_ = np.einsum("nij,nj->ni", Rk, a[s:e])
        um = ac_.mean(0)
        um /= np.linalg.norm(um)
        acv = ac_ @ um
        ach = ac_ - np.outer(acv, um)
        hb = bp(ach - ach.mean(0), max(0.8, 0.75 * f0), 1.25 * f0)
        ev, V = np.linalg.eigh(hb.T @ hb / N)
        d = V[:, 2]
        vb = np.einsum("nji,nj->ni", R[s:e], v[s:e])
        vbm = np.einsum("nij,nj->ni", Rk, vb).mean(0)
        vbm -= (vbm @ um) * um
        vd = vbm / (np.linalg.norm(vbm) + 1e-9)
        c = abs(d @ vd)
        ang = np.degrees(np.arccos(np.clip(c, 0, 1)))
        # lateral candidate: second eigenvector as walking axis
        c2 = abs(V[:, 1] @ vd)
        ang2 = np.degrees(np.arccos(np.clip(c2, 0, 1)))
        res.append((ang, ang2, ev[2] / ev[1], acp))
    res = np.array(res)
    if len(res) < 5:
        print(f"  room {n}: only {len(res)} straight fast windows")
        continue
    print(f"  room {n}: straight fast windows {len(res)}; axial error of 1st PC median/75 {np.percentile(res[:,0],50):.0f} {np.percentile(res[:,0],75):.0f}; of 2nd PC median {np.percentile(res[:,1],50):.0f}; where ac peak>0.6 (n={int((res[:,3]>0.6).sum())}) 1st PC median {np.percentile(res[res[:,3]>0.6,0],50) if (res[:,3]>0.6).any() else float('nan'):.0f}")

print("\nC. Speed regression with cadence-quality features, 2 s windows, leave-one-room-out")


def loro(F, name):
    yp, yt, bl = [], [], []
    for h in ROOMS:
        tr = np.array([x for k in ROOMS if k != h for x in speed_rows[k]])
        te = np.array(speed_rows[h])
        X = np.c_[np.ones(len(tr)), F(tr)]
        reg = 1e-3 * np.eye(X.shape[1])
        reg[0, 0] = 0
        b = np.linalg.solve(X.T @ X + reg, X.T @ tr[:, 0])
        yp.append(np.c_[np.ones(len(te)), F(te)] @ b)
        yt.append(te[:, 0])
        bl.append(np.full(len(te), tr[:, 0].mean()))
    y, p, base = map(np.concatenate, (yt, yp, bl))
    print(f"  {name:38s} R2 {1-((y-p)**2).sum()/((y-base)**2).sum():6.3f} r {r(y,p):5.2f} RMSE {np.sqrt(((y-p)**2).mean()):.3f} per-room r {' '.join(f'{r(a_,b_):5.2f}' for a_,b_ in zip(yt,yp))}")


L = np.log
loro(lambda z: np.zeros((len(z), 0)), "const")
loro(lambda z: np.c_[z[:, 2]], "autocorr peak height")
loro(lambda z: np.c_[z[:, 2], z[:, 1] * (z[:, 2] > 0.5)], "ac height + f_ac gated")
loro(lambda z: np.c_[L(z[:, 3]), L(z[:, 4]), L(z[:, 5])], "log rms v,h,g")
loro(lambda z: np.c_[z[:, 2], L(z[:, 3]), L(z[:, 4]), L(z[:, 5])], "ac height + log rms v,h,g")
loro(lambda z: np.c_[z[:, 2], z[:, 1] * (z[:, 2] > 0.5), L(z[:, 3]), L(z[:, 4]), L(z[:, 5])], "ac height + gated f_ac + log rms")
loro(lambda z: np.c_[z[:, 2], z[:, 1] * (z[:, 2] > 0.5), L(z[:, 3]), L(z[:, 4]), L(z[:, 5]), L(z[:, 7] + 0.02)], "... + log |yaw rate|")

print("\nD. EuRoC")
SEQ = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult", "MH05": "MH_05_difficult",
       "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult", "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
NW, HOP = 256, 40
fr2 = np.fft.rfftfreq(NW, 1 / FS)
win2 = np.hanning(NW)
band = (fr2 >= 75) & (fr2 <= 99.5)
print("seq | spectrum shift low->high thrust Hz (accel, gyro) | gyro centroid df/dlnT | partial r(centroid, climb | lnT) | partial r(centroid, v_h^2 | lnT) | median |yaw| per 15 s, 60 s deg | turn-based speed")
for name, d in SEQ.items():
    path = f"{ROOT}/euroc_airio/EuRoC-Dataset/{d}/mav0"
    imu = np.loadtxt(f"{path}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{path}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    t = imu[:, 0] * 1e-9
    w, a = imu[:, 1:4], imu[:, 4:7]
    tg = gt[:, 0] * 1e-9
    v = np.stack([np.interp(t, tg, gt[:, 8 + i]) for i in range(3)], 1)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    Rw = quat_R(q)
    bw = np.stack([np.interp(t, tg, gt[:, 11 + i]) for i in range(3)], 1)
    ba = np.stack([np.interp(t, tg, gt[:, 14 + i]) for i in range(3)], 1)
    vb = np.einsum("nji,nj->ni", Rw, v)
    st = np.arange(0, len(t) - NW, HOP)
    Pa = np.empty((len(st), len(fr2)))
    Pg = np.empty((len(st), len(fr2)))
    T = np.empty(len(st))
    means = np.empty((len(st), 3))
    vbw = np.empty((len(st), 3))
    for i, s in enumerate(st):
        seg = a[s:s + NW]
        means[i] = seg.mean(0)
        T[i] = np.linalg.norm(means[i])
        Pa[i] = (np.abs(np.fft.rfft((seg - seg.mean(0)) * win2[:, None], axis=0)) ** 2).sum(1)
        sg = w[s:s + NW]
        Pg[i] = (np.abs(np.fft.rfft((sg - sg.mean(0)) * win2[:, None], axis=0)) ** 2).sum(1)
        vbw[i] = vb[s:s + NW].mean(0)
    hi = Pa[:, fr2 >= 50].sum(1)
    on = hi > 0.2 * np.median(np.sort(hi)[-len(hi) // 2:])
    th = np.median(means[on], 0)
    th /= np.linalg.norm(th)
    lnT = np.log(T[on])
    q1, q3 = np.quantile(lnT, [0.25, 0.75])

    def shift(P):
        lo_, hi_ = P[on][lnT <= q1].mean(0)[band], P[on][lnT >= q3].mean(0)[band]
        lo_, hi_ = np.log(lo_) - np.log(lo_).mean(), np.log(hi_) - np.log(hi_).mean()
        lags = np.arange(-12, 13)
        c = [np.sum(lo_[max(0, -l):len(lo_) - max(0, l)] * hi_[max(0, l):len(hi_) - max(0, -l)]) for l in lags]
        return lags[int(np.argmax(c))] * (fr2[1] - fr2[0]), lnT[lnT >= q3].mean() - lnT[lnT <= q1].mean()

    sa_, dln = shift(Pa)
    sg_, _ = shift(Pg)
    cen = (Pa[:, band] * fr2[band]).sum(1) / Pa[:, band].sum(1)
    ceng = (Pg[:, band] * fr2[band]).sum(1) / Pg[:, band].sum(1)
    x0 = lnT - lnT.mean()
    slope_g = (x0 * (ceng[on] - ceng[on].mean())).sum() / (x0 * x0).sum()
    vz = vbw[on] @ th
    vp = vbw[on] - np.outer(vz, th)
    v2 = (vp * vp).sum(1)
    pz = partial_r(cen[on], vz, lnT)
    ph = partial_r(cen[on], v2, lnT)
    # yaw about thrust axis using truth gyro bias
    wz = ((w - bw) * th).sum(1)
    psi = np.cumsum(wz) / FS
    y15 = np.abs(psi[3000:] - psi[:-3000])
    y60 = np.abs(psi[12000:] - psi[:-12000]) if len(psi) > 13000 else np.array([np.nan])
    print(f"{name} | {sa_:5.1f} {sg_:5.1f} (dlnT {dln:.3f}; expected fold {-0.5*(200-cen[on].mean())*dln:5.1f}, no fold {+0.5*cen[on].mean()*dln:5.1f}) | {slope_g:6.1f} | {pz:5.2f} | {ph:5.2f} | {np.degrees(np.median(y15)):5.0f} {np.degrees(np.median(y60)):5.0f}")
    turn_speed(t, w - bw, a - ba, Rw, v, f"   {name} turn")
