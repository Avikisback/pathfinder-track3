"""Follow-up to 79_dragoffset.py. Is the 'hover reading' h = mean(accel) + k * mean(true in-plane body velocity) a stable
platform constant across EuRoC flights? If it is, a declared h recovers the mean body velocity c that the file-mean
offset throws away: c_est = -P (mean(accel) - h_decl) / k. Leave-one-environment-out. Clean files, ground truth, no solver.
Usage: 79b_hover.py"""
import sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from phys import rotors_on, _lpf
from scipy.spatial.transform import Rotation as Rot

K = 0.195
SEQ = [s for v in common.ENV.values() for s in v]


def one(name):
    r = common.load_room(name)
    t = r["t_ns"] * 1e-9
    imu = r["imu"]
    fs = round(1.0 / np.median(np.diff(t)))
    gt_t = r["gt_t_ns"] * 1e-9
    k = (t >= gt_t[0]) & (t <= gt_t[-1])
    t, imu = t[k], imu[k]
    p = np.stack([np.interp(t, gt_t, r["gt_p"][:, i]) for i in range(3)], 1)
    j = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
    Rw = Rot.from_quat(r["gt_q"][j])
    vb = Rw.inv().apply(np.gradient(_lpf(p, 5.0, fs), t, axis=0))
    a = imu[:, 3:6]
    on = rotors_on(a, fs)
    idx = np.nonzero(on)[0]
    m = np.zeros(len(t), bool)
    m[idx[0] + 3 * fs: idx[-1] - 3 * fs] = True
    m &= on
    am = a[m].mean(0)
    ta = am / np.linalg.norm(am)
    P = np.eye(3) - np.outer(ta, ta)
    c3 = P @ vb[m].mean(0)
    return dict(am=am, c3=c3, h=am + K * c3, P=P, rms=float(np.sqrt(((vb[m] @ P) ** 2).sum(1).mean())))


D = {n: one(n) for n in SEQ}
H = np.array([D[n]["h"] for n in SEQ])
print("per-file hover reading h (IMU axes, m/s2), and mean accel:")
for n in SEQ:
    print(f"  {n}  h {np.round(D[n]['h'], 3)}  mean accel {np.round(D[n]['am'], 3)}  k*c {np.round(K * D[n]['c3'], 3)}")
print("sd of h across files", np.round(H.std(0), 4), "| sd of mean accel", np.round(np.array([D[n]['am'] for n in SEQ]).std(0), 4))
print("\nleave-one-environment-out: error of the estimated mean body velocity (m/s)")
print(f"{'file':6s} |c| true  err if c=0 (now)  err with declared h   c/rms")
e0, e1 = [], []
for env, names in common.ENV.items():
    hd = np.mean([D[n]["h"] for n in SEQ if n not in names], 0)
    for n in names:
        d = D[n]
        ce = -d["P"] @ (d["am"] - hd) / K
        a0, a1 = np.linalg.norm(d["c3"]), np.linalg.norm(ce - d["c3"])
        e0.append(a0); e1.append(a1)
        print(f"{n:6s} {a0:6.3f}    {a0:6.3f}            {a1:6.3f}              {a0/d['rms']:.2f}")
print(f"mean   {np.mean(e0):6.3f}    {np.mean(e0):6.3f}            {np.mean(e1):6.3f}   | better on {sum(b < a for a, b in zip(e0, e1))} of {len(e0)}")
# same, with the declared constant taken inside the environment group (other flights of the same room and days)
e2 = []
for env, names in common.ENV.items():
    for n in names:
        hd = np.mean([D[x]["h"] for x in names if x != n], 0)
        d = D[n]
        e2.append(np.linalg.norm(-d["P"] @ (d["am"] - hd) / K - d["c3"]))
print(f"leave-one-flight-out inside the same environment: mean err {np.mean(e2):.3f} m/s")
