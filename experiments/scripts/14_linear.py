"""H4: a small learned zero-phase linear filter on integrated world acceleration. About 9 gains.

p_hat = IFFT( sum_j g_j B_j(f) A(f) / (-w^2) ), B_j triangular bands on a log frequency axis. Fitted by non-negative
least squares on training sequences (the convex version of 'gradient descent on ATE'), alternating with an SE(3)
alignment per training sequence. Trained with randomised degradation; tested on held-out degraded files.
"""
import json, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *
from scipy.optimize import nnls

KNOTS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])


def bands(nf, fs=FS):
    f = np.fft.rfftfreq(nf, 1 / fs)
    lf = np.log(np.maximum(f, 1e-4))
    lk = np.log(KNOTS)
    B = np.zeros((len(KNOTS), len(f)))
    for j in range(len(KNOTS)):
        e = np.zeros(len(KNOTS))
        e[j] = 1
        B[j] = np.interp(lf, lk, e, left=e[0], right=0.0)
    w2 = (2 * np.pi * f) ** 2
    w2[0] = np.inf
    return B, w2


def band_positions(aw):
    """aw (n,3) m/s^2 at 50 Hz. Returns array (bands, n, 3)."""
    n = len(aw)
    nf = int(2 ** np.ceil(np.log2(2 * n)))
    B, w2 = bands(nf)
    A = np.fft.rfft(aw, nf, axis=0)
    out = np.zeros((len(KNOTS), n, 3))
    for j in range(len(KNOTS)):
        out[j] = np.fft.irfft(A * (-(B[j] / w2))[:, None], nf, axis=0)[:n]
    return out


def world_accel(t, imu, ok, q):
    X, yaw, tc = features(t, imu, ok, q)
    return X[:, :3].astype(np.float64) * 4.0, tc


def train(train_seqs, rng, reps=3):
    items = []
    for n in train_seqs:
        room = load_room(n)
        t = (room["t_ns"] - room["t_ns"][0]) * 1e-9
        gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
        for _ in range(reps):
            imu, ok = random_degrade(t, room["imu"], rng)
            q = attitude_fast(t, imu, ok)
            aw, tc = world_accel(t, imu, ok, q)
            bp = band_positions(aw)
            m = (tc >= gt_t[0]) & (tc <= gt_t[-1])
            g = np.stack([np.interp(tc[m], gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
            items.append((bp[:, m], g))
    g_w = np.ones(len(KNOTS)) * 0.3
    for it in range(4):
        As, ys = [], []
        for bp, g in items:
            e = np.tensordot(g_w, bp, 1)
            r, tr = umeyama(e, g) if np.ptp(e, axis=0).max() > 1e-6 else (np.eye(3), g.mean(0))
            gl = (r.T @ (g - g.mean(0)).T).T     # ground truth in the estimate frame, centred
            b = bp - bp.mean(1, keepdims=True)
            As.append(b[:, ::10].reshape(len(KNOTS), -1).T)
            ys.append(gl[::10].ravel())
        g_w, _ = nnls(np.vstack(As), np.concatenate(ys))
    return g_w


def run(P, q, g_w):
    aw, tc = world_accel(P["t"], P["imu"], P["ok"], q)
    x = np.tensordot(g_w, band_positions(aw), 1)
    return to_rows(P, tc, x)


def evaluate(folds, label, refs):
    rows, gains = [], []
    rng = np.random.default_rng(7)
    for test, tr in folds:
        g_w = train(tr, rng)
        gains.append(g_w)
        for (n, s, m, sd) in eval_set(test):
            t, imu, _ = eval_file(n, s, m, sd)
            P = prepare(t, imu)
            q = attitude_auto(P["t"], P["imu"], P["ok"])
            rows.append((f"{n}|{s}|{m}|{sd}", score_all(P, load_room(n), run(P, q, g_w), q)))
    KEYS = ("ate_se3", "ate_org", "rpe_t", "blind_rel")
    ks = [k for k, _ in rows]
    print(f"== {label}")
    show("stationary", summarise([(k, refs[k]["stationary"]) for k in ks]), KEYS)
    show("baseline binary", summarise([(k, refs[k]["baseline"]) for k in ks]), KEYS)
    show("linear filter", summarise(rows), KEYS)
    print("   mean gains by band Hz:", {float(k): round(float(v), 3) for k, v in zip(KNOTS, np.mean(gains, 0))})
    for mt in ("ate_se3", "ate_org", "rpe_t"):
        print(f"   beats stationary on {mt}: {np.mean([d[mt] < refs[k]['stationary'][mt] for k, d in rows]):.2f}", flush=True)
    json.dump(rows, open(f"{RES}/h4_{label}.json", "w"))


refs_t = references(eval_set(DEV_ROOMS), f"{RES}/refs_dev.json")
refs_e = references(eval_set(list(EUROC)), f"{RES}/refs_euroc.json")
SEQS = list(EUROC)
evaluate([([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS], "tum_loro", refs_t)
evaluate([([s], [x for x in SEQS if x != s]) for s in SEQS], "euroc_loso", refs_e)
evaluate([(v, [x for x in SEQS if x not in v]) for v in ENV.values()], "euroc_loeo", refs_e)
evaluate([(SEQS, list(DEV_ROOMS))], "tum2euroc", refs_e)
evaluate([(list(DEV_ROOMS), SEQS)], "euroc2tum", refs_t)
