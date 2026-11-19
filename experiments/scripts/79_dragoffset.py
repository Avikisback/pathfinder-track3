"""Audit of one assumption in the drone drag target: `off = mean(f_p)` over the rotors-on rows (DragCal.set_target).
Since f_p = -k v_body_p + b_p, subtracting the file mean of f_p also subtracts -k * (mean in-plane BODY velocity). That is
right only if the drone flies as much backwards as forwards in its own frame. Uses ground truth, clean files, no solver.
Prints, per flight: the true mean body velocity c in the rotor plane, the rms of the true in-plane body velocity, the implied
true offset b_p = mean(f_p) + k c in IMU axes, and how much error (m, SE3-aligned, horizontal) the missing c alone puts into a
path integrated from perfect drag velocity with the true attitude.
Usage: 79_dragoffset.py"""
import sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from phys import rotors_on, _lpf
from scipy.spatial.transform import Rotation as Rot

K = 0.195
SEQ = [s for v in common.ENV.values() for s in v]
BB = ["bb_test_clover_maxSpeed5p0", "bb_test_egg_maxSpeed8p0", "bb_test_halfMoon_maxSpeed4p0", "bb_test_star_maxSpeed5p0",
      "bb_test_winter_maxSpeed4p0"]


def umeyama_err(est, gt):
    R, t = common.umeyama(est, gt)[:2] if False else (None, None)
    mu_e, mu_g = est.mean(0), gt.mean(0)
    H = (est - mu_e).T @ (gt - mu_g)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(est.shape[1])
    D[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ D @ U.T
    al = (est - mu_e) @ Rm.T + mu_g
    return float(np.sqrt(((al - gt) ** 2).sum(1).mean()))


def one(name, load):
    r = load(name)
    t = r["t_ns"] * 1e-9
    imu = r["imu"]
    fs = round(1.0 / np.median(np.diff(t)))
    gt_t = r["gt_t_ns"] * 1e-9
    k = (t >= gt_t[0]) & (t <= gt_t[-1])
    t, imu = t[k], imu[k]
    p = np.stack([np.interp(t, gt_t, r["gt_p"][:, i]) for i in range(3)], 1)
    j = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
    Rw = Rot.from_quat(r["gt_q"][j])
    vw = np.gradient(_lpf(p, 5.0, fs), t, axis=0)
    vb = Rw.inv().apply(vw)
    a = imu[:, 3:6]
    on = rotors_on(a, fs)
    if on.mean() < 0.3:
        return None
    # keep the main rotors-on stretch minus 3 s each side, where the vehicle is in the air
    idx = np.nonzero(on)[0]
    s, e = idx[0] + 3 * fs, idx[-1] - 3 * fs
    m = np.zeros(len(t), bool)
    m[s:e] = True
    m &= on
    ta = a[m].mean(0)
    ta /= np.linalg.norm(ta)
    e1 = np.cross(ta, [0, 1, 0.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ta, e1)
    E = np.stack([e1, e2], 1)
    fp = _lpf(a, 0.2, fs) @ E
    vbp = _lpf(vb, 0.2, fs) @ E
    off = fp[m].mean(0)
    c = vbp[m].mean(0)
    rms = np.sqrt((vbp[m] ** 2).sum(1).mean())
    bp = (off + K * c) @ E.T
    # path error from c alone: true in-plane body velocity with its mean removed, rotated by the true attitude
    dt = np.gradient(t)
    v_true = Rw.apply(vbp @ E.T)
    v_miss = Rw.apply((vbp - c) @ E.T)
    P0 = np.cumsum(v_true[m] * dt[m, None], 0)
    P1 = np.cumsum(v_miss[m] * dt[m, None], 0)
    err = umeyama_err(P1[::20, :2], P0[::20, :2])
    spread = float(np.sqrt(((P0[::20, :2] - P0[::20, :2].mean(0)) ** 2).sum(1).mean()))
    yaw = np.unwrap(Rw.as_euler("zyx")[:, 0])
    dur = t[m][-1] - t[m][0]
    return dict(name=name, c=c, cn=float(np.linalg.norm(c)), rms=float(rms), bp=bp, off=off @ E.T, err=err, spread=spread,
                dur=dur, yaw_tot=float(np.abs(np.diff(yaw[m])).sum() * 180 / np.pi))


def load_bb(n):
    d = np.load(f"{common.CACHE}/{n}.npz")
    return {k: d[k] for k in d.files}


for title, names, load in (("EuRoC", SEQ, common.load_room), ("Blackbird", BB, load_bb)):
    print(f"\n{title}: k = {K}", flush=True)
    print(f"{'file':30s} dur   |c| m/s  rms m/s  c/rms   b_p in IMU axes (m/s2)        path err from c (m) / true spread (m)")
    rows = []
    for n in names:
        try:
            o = one(n, load)
        except Exception as ex:
            print(n, "failed:", repr(ex)[:120]); continue
        if o is None:
            print(n, "not a drone by the detector"); continue
        rows.append(o)
        print(f"{n:30s} {o['dur']:4.0f}  {o['cn']:6.3f}  {o['rms']:6.3f}  {o['cn']/o['rms']:5.2f}   "
              f"{o['bp'][0]:+.3f} {o['bp'][1]:+.3f} {o['bp'][2]:+.3f}   {o['err']:5.2f} / {o['spread']:5.2f}", flush=True)
    if rows:
        B = np.array([o["bp"] for o in rows]); O = np.array([o["off"] for o in rows])
        print(f"mean |c| {np.mean([o['cn'] for o in rows]):.3f} m/s; k|c| {K*np.mean([o['cn'] for o in rows]):.3f} m/s2; "
              f"b_p sd across files {np.round(B.std(0), 3)}; file-mean offset sd {np.round(O.std(0), 3)}; "
              f"mean path err from c {np.mean([o['err'] for o in rows]):.2f} m against spread {np.mean([o['spread'] for o in rows]):.2f} m")
