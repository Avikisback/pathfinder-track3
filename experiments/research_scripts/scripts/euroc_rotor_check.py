"""Quick check on real EuRoC IMU: are the ~85-99 Hz and ~27 Hz peaks 1/rev and folded 2/rev?
Does the tracked peak follow thrust? Is body speed visible in vibration band power?
numpy only. Reads the team's data folder read-only. Prints a table; writes nothing else.
"""
import numpy as np, sys

ROOT = "E:/clipping-pipeline/xr_track3/data/euroc_airio/EuRoC-Dataset"
SEQ = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult", "MH05": "MH_05_difficult",
       "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult", "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
N, HOP, FS = 512, 100, 200.0   # 2.56 s window, 0.5 s hop


def load(name):
    d = f"{ROOT}/{SEQ[name]}/mav0"
    imu = np.loadtxt(f"{d}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{d}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    return imu, gt


def qrot_inv(q, v):
    """world vector v into body, q = w x y z body to world."""
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    return np.einsum("nji,nj->ni", R, v)


def r(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_r(a, b, c):
    """corr of a and b after removing a linear fit on c."""
    A = np.c_[np.ones_like(c), c]
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return r(ra, rb)


def peak_track(P, fr, lo, hi):
    """Parabolic-interpolated peak inside a band, per frame. Returns freq, prominence (peak over band median)."""
    idx = np.where((fr >= lo) & (fr <= hi))[0]
    S = P[:, idx]
    j = S.argmax(1)
    pk = S[np.arange(len(S)), j]
    med = np.median(S, axis=1)
    j2 = np.clip(j, 1, S.shape[1] - 2)
    a, b, c = np.log(S[np.arange(len(S)), j2 - 1] + 1e-12), np.log(S[np.arange(len(S)), j2] + 1e-12), np.log(S[np.arange(len(S)), j2 + 1] + 1e-12)
    d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return fr[idx[j2]] + np.clip(d, -1, 1) * (fr[1] - fr[0]), pk / (med + 1e-12)


print("name  frames | f1_med f2_med prom1 prom2 | slope f2~f1 icpt r(f1,f2) | r(f1,sqrtT) slope(f1 per sqrtT) | r(f1,spd) pr(f1,spd|T) | r(logP hi,spd) pr(.|T) | r(logP mid,spd) pr(.|T) | r(spd,T)")
rows = []
for name in SEQ:
    imu, gt = load(name)
    t = imu[:, 0] * 1e-9
    w, a = imu[:, 1:4], imu[:, 4:7]
    tg = gt[:, 0] * 1e-9
    win = np.hanning(N)
    st = np.arange(0, len(t) - N, HOP)
    fr = np.fft.rfftfreq(N, 1 / FS)
    Pa, T, Sp, Pg = [], [], [], []
    vg = np.stack([np.interp(t, tg, gt[:, 8 + i]) for i in range(3)], 1)
    qg = np.stack([np.interp(t, tg, gt[:, 4 + i]) for i in range(4)], 1)
    qg /= np.linalg.norm(qg, axis=1, keepdims=True)
    vb = qrot_inv(qg, vg)
    amean = np.array([np.mean(a[s:s + N], 0) for s in st])
    thrust_axis = np.median(amean, 0)
    thrust_axis /= np.linalg.norm(thrust_axis)
    for s in st:
        seg = a[s:s + N] - a[s:s + N].mean(0)
        Pa.append((np.abs(np.fft.rfft(seg * win[:, None], axis=0)) ** 2).sum(1))
        sg = w[s:s + N] - w[s:s + N].mean(0)
        Pg.append((np.abs(np.fft.rfft(sg * win[:, None], axis=0)) ** 2).sum(1))
        T.append(np.linalg.norm(a[s:s + N].mean(0)))
        vbs = vb[s:s + N]
        vp = vbs - np.outer(vbs @ thrust_axis, thrust_axis)
        Sp.append(np.linalg.norm(vp, axis=1).mean())
    Pa, Pg, T, Sp = np.array(Pa), np.array(Pg), np.array(T), np.array(Sp)
    # flight frames only: strong high-band vibration
    hi = (fr >= 50)
    power_hi = Pa[:, hi].sum(1)
    on = power_hi > 0.2 * np.median(np.sort(power_hi)[-len(power_hi) // 2:])
    f1, p1 = peak_track(Pa, fr, 70, 99.5)
    f2, p2 = peak_track(Pa, fr, 3, 50)
    g1, q1 = peak_track(Pg, fr, 70, 99.5)
    good = on & (p1 > 3)
    if good.sum() < 20:
        print(name, "too few frames", good.sum())
        continue
    A = np.c_[np.ones(good.sum()), f1[good]]
    icpt, slope = np.linalg.lstsq(A, f2[good], rcond=None)[0]
    sT = np.sqrt(T[good])
    sl = np.polyfit(sT, f1[good], 1)[0]
    lhi = np.log(Pa[good][:, fr >= 50].sum(1))
    lmid = np.log(Pa[good][:, (fr >= 5) & (fr < 50)].sum(1))
    row = dict(name=name, n=int(good.sum()), f1=np.median(f1[good]), f2=np.median(f2[good]), p1=np.median(p1[good]), p2=np.median(p2[good]), sl=slope, ic=icpt,
               r12=r(f1[good], f2[good]), rT=r(f1[good], sT), slT=sl, rS=r(f1[good], Sp[good]), prS=partial_r(f1[good], Sp[good], sT),
               rH=r(lhi, Sp[good]), prH=partial_r(lhi, Sp[good], sT), rM=r(lmid, Sp[good]), prM=partial_r(lmid, Sp[good], sT), rST=r(Sp[good], sT),
               g1=np.median(g1[good]), rg=r(g1[good], f1[good]))
    rows.append(row)
    print(f"{name} {row['n']:5d} | {row['f1']:5.1f} {row['f2']:5.1f} {row['p1']:6.1f} {row['p2']:6.1f} | {row['sl']:6.2f} {row['ic']:6.1f} {row['r12']:6.2f} | {row['rT']:6.2f} {row['slT']:7.1f} | "
          f"{row['rS']:6.2f} {row['prS']:6.2f} | {row['rH']:6.2f} {row['prH']:6.2f} | {row['rM']:6.2f} {row['prM']:6.2f} | {row['rST']:6.2f} | gyro f1 {row['g1']:5.1f} r(g,a) {row['rg']:5.2f}")
    # per-frame fold test: how often is f2 within 1.5 Hz of |200 - 2 f1| or other low-order folds
    cands = {"200-2f1": np.abs(200 - 2 * f1[good]), "3f1-200": np.abs(3 * f1[good] - 200), "f1/3": f1[good] / 3, "4f1-400": np.abs(4 * f1[good] - 400)}
    print("     fold match within 1.5 Hz:", {k: round(float(np.mean(np.abs(f2[good] - v) < 1.5)), 2) for k, v in cands.items()})

keys = ["rT", "rS", "prS", "rH", "prH", "rM", "prM", "rST", "r12"]
print("mean over sequences:", {k: round(float(np.nanmean([x[k] for x in rows])), 2) for k in keys})
