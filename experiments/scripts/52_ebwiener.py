"""New idea 3: empirical-Bayes Wiener shrink. The right high-pass differs by room (MH wants none, V1 wants about 8 s).

Instead of one tau, each file picks its own gains from its own spectrum. Velocity estimate = truth + error. Training
files give the error power per octave band (sigma_b^2). A test file's estimate has power P_b in each band, so the truth
has about P_b - sigma_b^2, and the gain is max(0, 1 - sigma_b^2 / P_b). No room size, no sequence identity, no ground
truth at test time. Error power is learned only on other environments.

Usage: 52_ebwiener.py <tag> <scheme>
Needs saved weights from 50_seqnet.
"""
import glob, importlib, json, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
e2e = importlib.import_module("32_e2e")
seq = importlib.import_module("50_seqnet")
from phys import *

# octave bands on 10 Hz velocity, top band up to Nyquist
EDGES_B = np.array([0, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 1e9])
NB = len(EDGES_B) - 1
DT = 0.1


def spec(x):
    """Mirrored FFT, so the ends do not leak. Returns spectrum, frequencies."""
    X = np.fft.rfft(np.concatenate([x, x[::-1]], 0), axis=0)
    return X, np.fft.rfftfreq(2 * len(x), DT)


def band_power(X, f, n):
    b = np.digitize(f, EDGES_B) - 1
    # power per unit length of signal, horizontal and vertical apart
    Pxy = np.bincount(b, (np.abs(X[:, :2]) ** 2).sum(1), minlength=NB)[:NB] / (2 * n) ** 2
    Pz = np.bincount(b, np.abs(X[:, 2]) ** 2, minlength=NB)[:NB] / (2 * n) ** 2
    return Pxy, Pz


def apply_gains(X, f, gxy, gz, n):
    cen = np.sqrt(np.maximum(EDGES_B[:-1], 1 / 512) * np.minimum(EDGES_B[1:], 4))
    lf = np.log(np.maximum(f, 1e-6))
    Gxy = np.interp(lf, np.log(cen), gxy)
    Gz = np.interp(lf, np.log(cen), gz)
    Gxy[0] = 0
    Gz[0] = 0
    Y = X.copy()
    Y[:, :2] *= Gxy[:, None]
    Y[:, 2] *= Gz
    return np.fft.irfft(Y, n=2 * n, axis=0)[:n]


def yaw_gauge(vh, v):
    cov = vh[:, :2].T @ v[:, :2]
    th = np.arctan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def truth_v(room, P, tk):
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    p = room["gt_p"]
    h = 0.05
    f = lambda x: np.stack([np.interp(x, gt_t, p[:, i]) for i in range(3)], 1)
    v = (f(tk + h) - f(tk - h)) / (2 * h)
    idx = np.clip(np.searchsorted(gt_t, tk), 1, len(gt_t) - 1)
    good = (tk - h >= gt_t[0]) & (tk + h <= gt_t[-1]) & ((gt_t[idx] - gt_t[idx - 1]) < 0.15)
    return v, good


def err_power(c, tk, v):
    """Error power per band on a file the net did not train on (ground truth, training stage only)."""
    P = c["P"]
    t_last = P["t"][c["ph"]["last"]]
    m = tk <= t_last
    vg, good = truth_v(c["room"], P, tk[m])
    vh = v[m]
    Rg = yaw_gauge(vh[good], vg[good])
    vs = (Rg @ vh.T).T
    e = np.where(good[:, None], vs - vg, 0.0)
    n = len(e)
    X, f = spec(e - e.mean(0))
    return band_power(X, f, n)


def eb_track(c, tk, v, Sxy, Sz, floor=0.0, alpha=1.0, tail_tau=1.0, tail_gain=1.0):
    P = c["P"]
    t_last = P["t"][c["ph"]["last"]]
    pre = tk <= t_last
    vh = v[pre] - v[pre].mean(0)
    n = len(vh)
    X, f = spec(vh)
    Pxy, Pz = band_power(X, f, n)
    gxy = np.clip(1 - alpha * Sxy / np.maximum(Pxy, 1e-12), floor, 1)
    gz = np.clip(1 - alpha * Sz / np.maximum(Pz, 1e-12), floor, 1)
    vf = apply_gains(X, f, gxy, gz, n)
    out = np.zeros_like(v)
    out[pre] = vf
    tail = ~pre
    if tail.any():
        v0 = vf[-5:].mean(0)
        out[tail] = tail_gain * v0[None] * np.exp(-(tk[tail] - t_last) / tail_tau)[:, None]
    x = np.cumsum(out, axis=0) * DT
    return to_rows(P, tk, x), gxy


if __name__ == "__main__":
    tag, scheme = sys.argv[1], sys.argv[2]
    folds = seq.folds_for(scheme)
    rows = {}
    allg = []
    for fi, (test, train) in enumerate(folds):
        ck = torch.load(f"{WGT}/{tag}_{scheme}_f{fi}.pt", weights_only=False)
        nets = []
        for st in ck["states"]:
            net = seq.SeqNet(hidden=ck["hidden"]).to("cuda")
            net.load_state_dict(st)
            net.eval()
            nets.append(net)
        e2e.CALIB[0] = e2e.calib_for(train)
        e2e._F.clear()
        # error power from files of OTHER environments is what would be available at test time. Training sequences were
        # seen by this net, so their error is optimistic: alpha below compensates, and is studied rather than picked here.
        Sxy, Sz = [], []
        for n in train:
            for f in (("clean", "delete", 1), ("mild", "zero", 1), ("full", "zero", 1)):
                c = e2e.file_ctx(n, *f)
                tk, v = seq.infer_seq(nets, c)
                a, b = err_power(c, tk, v)
                Sxy.append(a)
                Sz.append(b)
        Sxy, Sz = np.median(Sxy, 0), np.median(Sz, 0)
        e2e._F.clear()
        for key in eval_set(test):
            c = e2e.file_ctx(*key)
            tk, v = seq.infer_seq(nets, c)
            o = {}
            for al in (1, 2, 4, 8, 16):
                x, g = eb_track(c, tk, v, Sxy, Sz, alpha=al)
                o[f"eb{al}"] = c["sc"](x)
                if al == 4:
                    allg.append((key, g))
            rows["|".join(map(str, key))] = o
        print(f"fold {fi} done; train error power xy per band {np.round(Sxy*1e3,2)}", flush=True)
    pickle.dump(dict(rows=rows), open(f"{RES}/{tag}eb_{scheme}_f0.pkl", "wb"))
    for s in sorted({k[0][0] for k in allg}, key=str):
        gs = [g for k, g in allg if k[0] == s and k[1] == "full"]
        print(f"  {s} gains at alpha 4 (full): {np.round(np.mean(gs,0),2)}")
