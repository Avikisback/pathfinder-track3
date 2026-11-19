"""One-off edit of common.py and attitude.py: adds EuRoC loading and bias estimators."""
s = open("common.py").read()
if "EUROC = {" not in s:
    s = s.replace('''def load_room(n):
    """Returns dict: t_ns, imu (gyro, accel), gt_t_ns, gt_p, gt_q (x y z w)."""
    os.makedirs(CACHE, exist_ok=True)
''', '''EUROC = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult",
         "MH05": "MH_05_difficult", "V101": "V1_01_easy", "V102": "V1_02_medium", "V103": "V1_03_difficult",
         "V201": "V2_01_easy", "V202": "V2_02_medium", "V203": "V2_03_difficult"}
ENV = {"MH": ["MH01", "MH02", "MH03", "MH04", "MH05"], "V1": ["V101", "V102", "V103"], "V2": ["V201", "V202", "V203"]}


def euroc_dir(name):
    return f"{ROOT}/data/euroc_airio/EuRoC-Dataset/{EUROC[name]}/mav0"


def load_room(n):
    """n: int TUM-VI room or EuRoC short name. Returns t_ns, imu (gyro, accel), gt_t_ns, gt_p, gt_q (x y z w)."""
    os.makedirs(CACHE, exist_ok=True)
    if isinstance(n, str):
        c = f"{CACHE}/{n}.npz"
        if os.path.exists(c):
            d = np.load(c)
            return {k: d[k] for k in d.files}
        t, imu = _read_csv_ns(euroc_dir(n) + "/imu0/data.csv")
        gt_t, gt = _read_csv_ns(euroc_dir(n) + "/state_groundtruth_estimate0/data.csv")
        out = dict(t_ns=t, imu=imu, gt_t_ns=gt_t, gt_p=gt[:, :3], gt_q=gt[:, [4, 5, 6, 3]], gt_bw=gt[:, 10:13])
        np.savez(c, **out)
        return out
''')
    s = s.replace("def score(est_t, est_p, est_q_xyzw, gt_t, gt_p, gt_q_xyzw, rpe_frames=120):",
                  "def score(est_t, est_p, est_q_xyzw, gt_t, gt_p, gt_q_xyzw, rpe_frames=None):")
    s = s.replace('''    gi, ei = associate(gt_t, est_t)
    g, e = gt_p[gi], est_p[ei]
    Rg, Re''', '''    if rpe_frames is None:  # 1 s of ground truth, whatever its rate (V101 is 20 Hz)
        rpe_frames = int(round(1.0 / np.median(np.diff(gt_t))))
    gi, ei = associate(gt_t, est_t)
    g, e = gt_p[gi], est_p[ei]
    Rg, Re''')
    open("common.py", "w").write(s)

s = open("attitude.py").read()
if "def gravity_bias" not in s:
    s = s.replace("residual_s=0.0, fit_bias=True, return_info=False, s_r=0.03):\n    gyro, acc = imu[:, :3], imu[:, 3:]\n    n = len(t)\n    q0, g = quiet_init(t, imu, ok)",
                  "residual_s=0.0, fit_bias=True, return_info=False, s_r=0.03, b0=None):\n    gyro, acc = imu[:, :3], imu[:, 3:]\n    n = len(t)\n    b0 = np.zeros(3) if b0 is None else np.asarray(b0)\n    imu2 = imu.copy()\n    imu2[ok, :3] -= b0\n    q0, g = quiet_init(t, imu2, ok)")
    s = s.replace("        bias = Bk @ c\n", "        bias = Bk @ c + b0\n")
    s += '''

def still_bias(t, imu, ok, std_thr=0.01):
    """Mean gyro over truly still seconds, or None. Same test as main.cpp: gyro std per axis under 0.01 rad/s."""
    fs = int(round(1.0 / np.median(np.diff(t))))
    means, first = [], None
    for s in range(0, len(t) - fs, fs):
        m = ok[s:s + fs]
        if m.sum() < 0.8 * fs:
            continue
        w = imu[s:s + fs, :3][m]
        if np.sqrt(w.var(0).mean()) < std_thr:
            means.append(w.mean(0))
            if first is None:
                first = float(t[s])
    if not means:
        return None, 0.0, None
    return np.median(np.array(means), 0), float(len(means)), first


def gravity_bias(t, imu, ok, sig_thr=0.004, W=1.0, return_all=False):
    """Closed-form constant gyro bias from how gravity moves in the body frame. Needs no starting attitude.

    Mean accel over a window gives up-in-body u. If the gyro were right, the gyro rotation between two windows
    maps u[k+L] onto u[k]. The mismatch is linear in bias: dR u[k+L] - u[k] = L dR (b x u[k+L]). Directions of b
    that gravity does not constrain stay at zero, by truncating on the estimated standard deviation.
    """
    fs = int(round(1.0 / np.median(np.diff(t))))
    wn = int(W * fs)
    last = np.nonzero(ok)[0][-1]
    starts = np.arange(0, last - wn, wn)
    cs = np.cumsum(np.vstack([np.zeros((1, 3)), imu[:, 3:] * ok[:, None]]), 0)
    cn = np.cumsum(np.concatenate([[0], ok.astype(float)]))
    cnt_w = cn[starts + wn] - cn[starts]
    good = cnt_w > 0.9 * wn
    am = (cs[starts + wn] - cs[starts]) / np.maximum(cnt_w, 1)[:, None]
    an = np.linalg.norm(am, axis=1)
    g = np.median(an[good])
    u = am / np.maximum(an, 1e-6)[:, None]
    good &= np.abs(an - g) < 1.0
    centre = starts + wn // 2
    gap_id = np.cumsum(~ok)
    b = np.zeros(3)
    sig = np.full(3, 9.0)
    V = np.eye(3)
    for lags in ((1, 2), (1, 2, 4, 8), (1, 2, 4, 8, 16)):
        coast = np.zeros_like(imu[:, :3])
        q = integrate_rows(t, imu[:, :3], ok, np.tile(b, (len(t), 1)), coast, np.array([1.0, 0, 0, 0]), np.zeros((len(t), 3)))
        Rq = R.from_quat(q[centre][:, [1, 2, 3, 0]])
        H, y, ss, cnt = np.zeros((3, 3)), np.zeros(3), 0.0, 0
        for L in lags:
            i = np.arange(0, len(centre) - L)
            i = i[good[i] & good[i + L] & (gap_id[centre[i]] == gap_id[centre[i + L]])]
            if len(i) < 10:
                continue
            dR = Rq[i].inv() * Rq[i + L]
            uj = u[i + L]
            r = dR.apply(uj) - u[i]
            dt = t[centre[i + L]] - t[centre[i]]
            Dm = dR.as_matrix()
            cx = np.zeros((len(i), 3, 3))   # matrix of b -> b x u
            cx[:, 0, 1], cx[:, 0, 2] = uj[:, 2], -uj[:, 1]
            cx[:, 1, 0], cx[:, 1, 2] = -uj[:, 2], uj[:, 0]
            cx[:, 2, 0], cx[:, 2, 1] = uj[:, 1], -uj[:, 0]
            A = dt[:, None, None] * (Dm @ cx)
            H += np.einsum("nij,nik->jk", A, A)
            y += np.einsum("nij,ni->j", A, r)
            ss += (r ** 2).sum()
            cnt += 3 * len(i)
        if cnt == 0:
            break
        s2 = max(ss / cnt, 1e-6)
        ev, V = np.linalg.eigh(H)
        sig = np.sqrt(s2 / np.maximum(ev, 1e-12))
        thr = sig_thr * (1 if len(lags) > 2 else 4)
        b = b + V @ np.where(sig < thr, (V.T @ y) / np.maximum(ev, 1e-12), 0.0)
    if return_all:
        return b, sig, V
    return b
'''
    open("attitude.py", "w").write(s)
print("patched")
