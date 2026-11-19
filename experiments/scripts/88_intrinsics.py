"""Are scale and axis-alignment errors of the IMU large enough to matter on handheld files?

Everything so far models sensor error as an additive bias curve. A scale or misalignment error E instead adds E * (specific
force), which changes whenever the device is turned, so on a hand-waved rig it acts at motion time scales (4-16 s), where
the handheld error sits. 0.3 % of g is 0.03 m/s^2, which is 0.15 m/s after 5 s.

Step 1 (this script, ground truth, clean files): per room fit
    accel:  R (C a - c) = p'' + gamma     C 3x3, c bias, gamma gravity vector (all linear), low-passed at 1 Hz
    gyro:   w_true = G w - d              G 3x3, d bias, w_true from the mocap attitude, low-passed at 2 Hz
and print C - I and G - I in percent, and how well one room's matrices agree with the others'.
Step 2: write test files with the OTHER rooms' mean C and G applied (leave one room out), for score_bin.py set `intr`.
Usage: 88_intrinsics.py"""
import sys, os
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from ev import *
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation as Rot, Slerp
import importlib
m77 = importlib.import_module("77_lever")
OUT = EXP + "/tmp/cpp_intr"
os.makedirs(OUT, exist_ok=True)


def lp(x, fc, fs):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def fit(n):
    r = load_room(n)
    t = r["t_ns"] * 1e-9
    gt_t = r["gt_t_ns"] * 1e-9
    fs = round(1.0 / np.median(np.diff(t)))
    k = (t > gt_t[0] + 1) & (t < gt_t[-1] - 1)
    t, imu = t[k], r["imu"][k]
    # mocap holes: mark IMU rows further than 30 ms from a mocap sample, plus 1.5 s each side
    j = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1)
    far = np.minimum(np.abs(gt_t[j] - t), np.abs(gt_t[j - 1] - t)) > 0.03
    bad = np.convolve(far.astype(float), np.ones(int(3 * fs)), "same") > 0
    p = np.stack([np.interp(t, gt_t, r["gt_p"][:, i]) for i in range(3)], 1)
    acc_w = savgol_filter(p, int(0.6 * fs) | 1, 3, deriv=2, delta=1.0 / fs, axis=0)
    Rg = Slerp(gt_t, Rot.from_quat(r["gt_q"]))(t)
    Rm = Rg.as_matrix()
    a, w = imu[:, 3:], imu[:, :3]
    # accel: sum_ij C_ij R[:, i] a_j - sum_i c_i R[:, i] - gamma = acc_w
    cols = [Rm[:, :, i] * a[:, j:j + 1] for i in range(3) for j in range(3)] + [-Rm[:, :, i] for i in range(3)]
    cols += [np.tile(-np.eye(3)[kk], (len(t), 1)) for kk in range(3)]
    X = np.stack([lp(c, 1.0, fs) for c in cols], 2)            # rows x 3 x 15
    y = lp(acc_w, 1.0, fs)
    sel = np.nonzero(~bad)[0][::20]
    A = X[sel].reshape(-1, 15)
    th = np.linalg.lstsq(A, y[sel].reshape(-1), rcond=None)[0]
    C, c, gam = th[:9].reshape(3, 3), th[9:12], th[12:]
    # the same with C fixed to identity, for the residual it removes
    A0 = np.concatenate([A[:, 9:]], 1)
    y0 = y[sel].reshape(-1) - A[:, [0, 4, 8]].sum(1)
    th0 = np.linalg.lstsq(A0, y0, rcond=None)[0]
    res0 = np.sqrt(((A0 @ th0 - y0) ** 2).mean() * 3)
    res1 = np.sqrt(((A @ th - y[sel].reshape(-1)) ** 2).mean() * 3)
    # gyro: true body rate from mocap attitude increments
    dR = (Rg[:-1].inv() * Rg[1:]).as_rotvec() * fs
    wt = np.vstack([dR, dR[-1:]])
    wt = savgol_filter(wt, int(0.3 * fs) | 1, 2, axis=0)
    Xg = np.concatenate([lp(w, 2.0, fs), np.ones((len(t), 1))], 1)[sel]
    Yg = lp(wt, 2.0, fs)[sel]
    Tg = np.linalg.lstsq(Xg, Yg, rcond=None)[0]              # 4 x 3
    G, d = Tg[:3].T, -Tg[3]
    rg0 = np.sqrt((((Xg[:, :3] + (Yg - Xg[:, :3]).mean(0)) - Yg) ** 2).sum(1).mean())
    rg1 = np.sqrt(((Xg @ Tg - Yg) ** 2).sum(1).mean())
    return dict(C=C, c=c, gam=gam, G=G, d=d, res=(res0, res1), resg=(rg0, rg1))


F = {n: fit(n) for n in DEV_ROOMS}
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for n in DEV_ROOMS:
    f = F[n]
    print(f"room {n}: accel C - I (%) rows {(100 * (f['C'] - np.eye(3))).round(2).tolist()} | gravity {np.round(f['gam'], 3)} | bias {np.round(f['c'], 3)} "
          f"| world accel residual {f['res'][0]:.3f} -> {f['res'][1]:.3f} m/s2")
    print(f"        gyro  G - I (%) rows {(100 * (f['G'] - np.eye(3))).round(2).tolist()} | bias {np.round(f['d'], 4)} | rate residual {f['resg'][0]:.4f} -> {f['resg'][1]:.4f} rad/s")
Cs = np.array([F[n]["C"] for n in DEV_ROOMS]); Gs = np.array([F[n]["G"] for n in DEV_ROOMS])
print("accel C - I mean (%):", (100 * (Cs.mean(0) - np.eye(3))).round(2).tolist(), "| sd across rooms (%):", (100 * Cs.std(0)).round(2).tolist())
print("gyro  G - I mean (%):", (100 * (Gs.mean(0) - np.eye(3))).round(2).tolist(), "| sd across rooms (%):", (100 * Gs.std(0)).round(2).tolist())

# step 2: test files with the other rooms' matrices applied
for n in DEV_ROOMS:
    C = np.mean([F[m]["C"] for m in DEV_ROOMS if m != n], 0)
    G = np.mean([F[m]["G"] for m in DEV_ROOMS if m != n], 0)
    for tag, Ca, Ga in (("both", C, G), ("acc", C, np.eye(3)), ("gyr", np.eye(3), G), ("own", F[n]["C"], F[n]["G"])):
        for k in (("clean", "delete", 1), ("full", "zero", 1)):
            t, imu, room = m77.m72.file_for2(n, k)
            imu = np.array(imu, float)
            live = np.abs(imu).sum(1) > 0
            imu[live, 3:] = imu[live, 3:] @ Ca.T
            imu[live, :3] = imu[live, :3] @ Ga.T
            common.write_csv(f"{OUT}/room{n}_{tag}{k[0]}.csv", t, imu)
print("wrote", OUT)
