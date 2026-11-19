"""Second EuRoC check. Thrust-binned spectra: which way does the ~90 Hz tone move with thrust?
Rotor physics says f is proportional to sqrt(thrust). An unfolded tone gives df/dlnT = +0.5 f; a tone folded about 100 Hz gives -0.5 (200 - f).
Also lists top spectral peaks with rotors on and off, to see which peaks are rotor-locked. numpy only, read-only.
"""
import numpy as np

ROOT = "E:/clipping-pipeline/xr_track3/data/euroc_airio/EuRoC-Dataset"
SEQ = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult", "MH05": "MH_05_difficult",
       "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult", "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
N, HOP, FS = 256, 40, 200.0
fr = np.fft.rfftfreq(N, 1 / FS)
win = np.hanning(N)


def top_peaks(P, k=5, lo=3, hi=99.5):
    idx = np.where((fr > lo) & (fr < hi))[0]
    S = P[idx]
    loc = [i for i in range(1, len(S) - 1) if S[i] > S[i - 1] and S[i] > S[i + 1]]
    # prominence relative to a wide local median
    pr = [(S[i] / np.median(S[max(0, i - 12):i + 13]), fr[idx[i]]) for i in loc]
    pr.sort(reverse=True)
    return [(round(f, 1), round(p, 1)) for p, f in pr[:k]]


slopes = []
print("seq  | n_on n_off | T: sd(lnT) | df/dlnT(centroid 75-99.5) r | quartile centroids | expected +0.5f / -0.5(200-f)")
for name in SEQ:
    imu = np.loadtxt(f"{ROOT}/{SEQ[name]}/mav0/imu0/data.csv", delimiter=",", skiprows=1)
    w, a = imu[:, 1:4], imu[:, 4:7]
    st = np.arange(0, len(a) - N, HOP)
    Pa = np.empty((len(st), len(fr)))
    Pg = np.empty((len(st), len(fr)))
    T = np.empty(len(st))
    for i, s in enumerate(st):
        seg = a[s:s + N]
        T[i] = np.linalg.norm(seg.mean(0))
        Pa[i] = (np.abs(np.fft.rfft((seg - seg.mean(0)) * win[:, None], axis=0)) ** 2).sum(1)
        sg = w[s:s + N]
        Pg[i] = (np.abs(np.fft.rfft((sg - sg.mean(0)) * win[:, None], axis=0)) ** 2).sum(1)
    hi = Pa[:, fr >= 50].sum(1)
    on = hi > 0.2 * np.median(np.sort(hi)[-len(hi) // 2:])
    band = (fr >= 75) & (fr <= 99.5)
    cen = (Pa[:, band] * fr[band]).sum(1) / Pa[:, band].sum(1)
    x = np.log(T[on])
    x0 = x - x.mean()
    y = cen[on]
    sl = (x0 * (y - y.mean())).sum() / (x0 * x0).sum()
    rr = np.corrcoef(x0, y)[0, 1]
    q = np.quantile(x, [0.25, 0.5, 0.75])
    bins = np.digitize(x, q)
    qc = []
    for b in range(4):
        P = Pa[on][bins == b].mean(0)
        qc.append((P[band] * fr[band]).sum() / P[band].sum())
    fm = y.mean()
    slopes.append(sl)
    print(f"{name} | {on.sum():5d} {(~on).sum():5d} | {x.std():.4f} | {sl:7.1f} {rr:5.2f} | {' '.join(f'{c:5.1f}' for c in qc)} | {0.5*fm:5.1f} {-0.5*(200-fm):5.1f}")
    print("      accel on :", top_peaks(Pa[on].mean(0)))
    print("      gyro  on :", top_peaks(Pg[on].mean(0)))
    if (~on).sum() > 20:
        print("      accel off:", top_peaks(Pa[~on].mean(0)))
    # thrust dependence of the low peak: same test for a 15-35 Hz centroid
    b2 = (fr >= 15) & (fr <= 35)
    c2 = (Pa[:, b2] * fr[b2]).sum(1) / Pa[:, b2].sum(1)
    y2 = c2[on]
    print(f"      15-35 Hz centroid df/dlnT {((x0*(y2-y2.mean())).sum()/(x0*x0).sum()):7.1f} r {np.corrcoef(x0,y2)[0,1]:5.2f}")
print("mean slope", np.mean(slopes).round(1), "share negative", np.mean(np.array(slopes) < 0))
