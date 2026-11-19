"""Second TUM-VI check, rooms 1-5 only (room 6 stays locked). numpy only, read-only.
1. Velocity split along body axes: can a lateral body-axis constraint work?
2. Where is the gait peak? PSD peaks of vertical accel while moving, and FFT vs autocorrelation cadence.
3. Slow-speed detection AUC from band-limited features, under added white noise.
"""
import numpy as np

ROOT = "E:/clipping-pipeline/xr_track3/data"
ROOMS = [1, 2, 3, 4, 5]
FS = 200.0
N = 400
HOP = 100
RNG = np.random.default_rng(1)


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


def bp(x, lo, hi):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    m = (f >= lo) & (f <= hi)
    X[~m] = 0
    return np.fft.irfft(X, n=len(x), axis=0)


def auc(score, label):
    """P(score of positive > score of negative)."""
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    np_, nn = label.sum(), (~label).sum()
    if np_ == 0 or nn == 0:
        return float("nan")
    return (rank[label].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)


data = {}
for n in ROOMS:
    imu = np.loadtxt(f"{ROOT}/tumvi_room{n}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{ROOT}/tumvi_room{n}/mocap0/data.csv", delimiter=",", skiprows=1)
    data[n] = (imu, gt)

print("1. Body-axis split of velocity (0.5 s smoothing, truth attitude). RMS m/s and share of horizontal energy per body axis")
tot = []
for n in ROOMS:
    imu, gt = data[n]
    t = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    v = np.gradient(box(p, 100), 1 / FS, axis=0)
    q = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_R(q)
    vb = np.einsum("nji,nj->ni", R, v)
    up_b = R[:, 2, :]              # world z in body
    print(f"  room {n}: rms vb x,y,z {np.sqrt((vb**2).mean(0)).round(3)}  mean vb {vb.mean(0).round(3)}  mean |up_b| comps {up_b.mean(0).round(2)}")
    # cone: axial angle between horizontal velocity and -y body axis projected to horizontal
    fwd = np.array([0.0, -1.0, 0.0])
    fh = fwd - np.outer(up_b @ fwd, np.ones(3)) * up_b
    fh /= np.linalg.norm(fh, axis=1, keepdims=True) + 1e-12
    vh = v - np.outer(v[:, 2], [0, 0, 1])
    # express fwd axis in world and compare
    fw = np.einsum("nij,nj->ni", R, fh)
    sp = np.linalg.norm(vh, axis=1)
    mv = sp > 0.5
    cosang = (fw[mv] * vh[mv]).sum(1) / sp[mv]
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    lat = sp[mv] * np.sqrt(np.maximum(1 - cosang ** 2, 0))
    print(f"     moving share {mv.mean():.2f}; signed angle to -y median/75 {np.percentile(ang,50):.0f} {np.percentile(ang,75):.0f}; axial median/75 {np.percentile(np.minimum(ang,180-ang),50):.0f} {np.percentile(np.minimum(ang,180-ang),75):.0f}; lateral part rms {np.sqrt((lat**2).mean()):.2f} of speed rms {np.sqrt((sp[mv]**2).mean()):.2f}; along-axis mean {np.mean(sp[mv]*cosang):.2f}")

print("\n2. Gait peak while moving. 2 s windows. Vertical accel PSD peaks (0.5-5 Hz) averaged over moving windows, per room")
allf = {}
for n in ROOMS:
    imu, gt = data[n]
    t = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    a = imu[:, 4:7]
    up = box(a, 200)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    av = (a * up).sum(1)
    p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    v = np.gradient(box(p, 50), 1 / FS, axis=0)
    sp = np.linalg.norm(v[:, :2], axis=1)
    nfft = 4096
    fr = np.fft.rfftfreq(nfft, 1 / FS)
    win = np.hanning(800)
    Ps = []
    Sp = []
    fac, ffft, amp = [], [], []
    for s in range(100, len(t) - 900, 200):
        seg = av[s:s + 800] - av[s:s + 800].mean()
        Ps.append(np.abs(np.fft.rfft(seg * win, n=nfft)) ** 2)
        Sp.append(sp[s:s + 800].mean())
        b = bp(seg, 0.8, 5.0)
        ac = np.correlate(b, b, "full")[len(b) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(0.28 * FS), int(1.0 * FS)
        j = lo + np.argmax(ac[lo:hi])
        fac.append(FS / j)
        amp.append(ac[j])
        g = (fr >= 1.2) & (fr <= 3.0)
        ffft.append(fr[g][np.argmax(Ps[-1][g])])
    Ps, Sp, fac, ffft, amp = np.array(Ps), np.array(Sp), np.array(fac), np.array(ffft), np.array(amp)
    m = Sp > 0.7
    mp = Ps[m].mean(0)
    g = (fr > 0.5) & (fr < 5)
    idx = np.where(g)[0]
    pk = [i for i in idx[1:-1] if mp[i] > mp[i - 1] and mp[i] > mp[i + 1]]
    pk = sorted(pk, key=lambda i: -mp[i])[:4]
    print(f"  room {n}: moving windows {m.sum()}/{len(m)}; top PSD peaks Hz {[round(fr[i],2) for i in pk]}; rel power {[round(mp[i]/mp[pk[0]],2) for i in pk]}")
    print(f"     cadence FFT 1.2-3 Hz median {np.median(ffft[m]):.2f}, autocorr median {np.median(fac[m]):.2f}, ac peak height median {np.median(amp[m]):.2f}; r(speed,f_fft) all {np.corrcoef(Sp,ffft)[0,1]:.2f} moving {np.corrcoef(Sp[m],ffft[m])[0,1]:.2f}; r(speed,f_ac) all {np.corrcoef(Sp,fac)[0,1]:.2f} moving {np.corrcoef(Sp[m],fac[m])[0,1]:.2f}; r(speed,acpeak) {np.corrcoef(Sp,amp)[0,1]:.2f}")

print("\n3. Slow-speed detection, 2 s windows (N=400, hop 0.5 s). Label: mean horizontal speed below threshold. AUC of (lower feature = slower)")
for sa, sg in ((0, 0), (0.08, 0.008), (0.4, 0.04), (2.0, 0.2)):
    rows = []
    for n in ROOMS:
        imu, gt = data[n]
        t = imu[:, 0] * 1e-9
        tg = gt[:, 0] * 1e-9
        w = imu[:, 1:4] + (RNG.normal(0, sg, imu[:, 1:4].shape) if sg else 0)
        a = imu[:, 4:7] + (RNG.normal(0, sa, imu[:, 4:7].shape) if sa else 0)
        up = box(a, 200)
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        av = (a * up).sum(1)
        p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
        v = np.gradient(box(p, 50), 1 / FS, axis=0)
        sp = np.linalg.norm(v[:, :2], axis=1)
        for s in range(100, len(t) - N - 100, HOP):
            e = s + N
            aw = a[s:e]
            ww = w[s:e]
            ah = aw - av[s:e, None] * up[s:e]
            f_v = np.std(bp(av[s:e] - av[s:e].mean(), 1, 6))
            f_h = np.sqrt((bp(ah - ah.mean(0), 1, 6) ** 2).sum(1).mean())
            f_g = np.sqrt((bp(ww - ww.mean(0), 0.5, 6) ** 2).sum(1).mean())
            f_raw = np.sqrt(((ww - ww.mean(0)) ** 2).sum(1).mean())
            f_araw = np.std(np.linalg.norm(aw, axis=1))
            shoe = ((np.linalg.norm(ww, axis=1) ** 2)).mean()
            rows.append((sp[s:e].mean(), f_v, f_h, f_g, f_raw, f_araw, shoe))
    r_ = np.array(rows)
    sp_ = r_[:, 0]
    out = []
    for th in (0.1, 0.2, 0.4):
        lab = sp_ < th
        out.append(f"th {th}: share {lab.mean():.2f} AUC v {auc(-r_[:,1],lab):.2f} h {auc(-r_[:,2],lab):.2f} gBand {auc(-r_[:,3],lab):.2f} gRaw {auc(-r_[:,4],lab):.2f} aNormStd {auc(-r_[:,5],lab):.2f} gyroEnergy(SHOE-like) {auc(-r_[:,6],lab):.2f}")
    print(f"  noise a {sa} g {sg}:")
    for o in out:
        print("     ", o)
    # combined rank score and speed correlation
    z = (np.log(r_[:, 1]) - np.log(r_[:, 1]).mean()) / np.log(r_[:, 1]).std() + (np.log(r_[:, 2]) - np.log(r_[:, 2]).mean()) / np.log(r_[:, 2]).std() + (np.log(r_[:, 3]) - np.log(r_[:, 3]).mean()) / np.log(r_[:, 3]).std()
    print(f"      combined log(v,h,g) r with speed {np.corrcoef(z,sp_)[0,1]:.2f}; AUC th 0.2 {auc(-z,sp_<0.2):.2f}; AUC th 0.4 {auc(-z,sp_<0.4):.2f}")
