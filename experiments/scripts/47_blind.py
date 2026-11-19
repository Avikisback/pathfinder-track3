"""Blind 5 s: which forecast shape? Failure at the real end of each file, and failure mid-file, since the deck does not say.
Uses velocity from the solver on mild-degraded files. Mid-file velocity uses two-sided data, so that part is a little
optimistic. Forecast: p(s) = g * v * tau * (1 - exp(-s / tau)). tau 0 means hold position.
"""
import importlib, multiprocessing as mp, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
m46 = importlib.import_module("46_drag_energy")
from phys import *

TAUS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)
GAINS = (0.5, 1.0)
S = np.arange(0.05, 5.0001, 0.05)


def one(n):
    t, imu, _ = eval_file(n, "mild", "zero", 1)
    P = prepare(t, imu)
    room = load_room(n)
    q, v, tb, last, info = m46.solve_any(P["t"], P["imu"], P["ok"], b0=m41.calib_excl(n))
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    gp = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    fs = 1.0 / np.median(np.diff(tb))
    p = np.cumsum(v, axis=0) / fs
    m = (tb >= gt_t[0]) & (tb <= gt_t[-1])
    r, tr = umeyama(p[m], gp(tb[m]))
    vw = (r @ v.T).T
    out = {}
    Ts = {"end": [tb[-1]], "mid": list(np.arange(max(10.0, gt_t[0] + 5), tb[-1] - 5, 5.0))}
    for nm, tl in Ts.items():
        errs = {}
        for T in tl:
            if T + 1 > gt_t[-1]:
                continue
            k = np.searchsorted(tb, T)
            v0 = vw[max(0, k - int(0.5 * fs)) : k].mean(0)
            d = gp(T + S) - gp(np.array([T]))
            for tau in TAUS:
                for g in GAINS:
                    f = (g * v0[None] * (tau * (1 - np.exp(-S / tau)))[:, None]) if tau > 0 else np.zeros_like(d)
                    errs.setdefault((tau, g), []).append(((f - d) ** 2).sum(1).mean())
        out[nm] = {k2: float(np.mean(x)) for k2, x in errs.items()} if errs else None
    return n, out


if __name__ == "__main__":
    names = m41.SEQS + DEV_ROOMS
    with mp.get_context("spawn").Pool(10) as p:
        res = dict(p.imap_unordered(one, names))
    for plat, ns in (("EuRoC", m41.SEQS), ("TUM", DEV_ROOMS)):
        for case in ("end", "mid"):
            print(f"-- {plat} failure at {case} of file: RMSE over 5 s, by tau, gain 1 / gain 0.5")
            print("   " + "  ".join(f"tau{tau}: {np.sqrt(np.mean([res[n][case][(tau,1.0)] for n in ns if res[n][case]])):.3f}/{np.sqrt(np.mean([res[n][case][(tau,0.5)] for n in ns if res[n][case]])):.3f}" for tau in TAUS))
