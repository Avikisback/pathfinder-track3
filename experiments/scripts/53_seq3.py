"""The stack: self-calibrated attitude and velocity as inputs, whole-trajectory net on top, self-adjusting Wiener at the end.

Everything per fold: net trained on training sequences only, error power measured on training sequences only.
Usage: 53_seq3.py make_pool | train --scheme loeo|loro|loso [--tag s3] [--members 1] [--steps 2500]
"""
import argparse, importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
e2e = importlib.import_module("32_e2e")
seq = importlib.import_module("50_seqnet")
eb = importlib.import_module("52_ebwiener")
from phys import *
from energy2 import energy_cal

NR = 8
EN_KW = dict(sig_a=0.1, sig_p=5.0)


def build3(t, imu, ok, b0):
    q, v, tb, last, info = energy_cal(t, imu, ok, b0=b0, return_info=True, **EN_KW)
    X, yaw, tc = features(t, imu, ok, q)
    ph = phys_channels(t, imu, ok, q)
    n = len(X)
    idx = DEC // 2 + np.arange(n) * DEC
    ven = np.stack([np.interp(tc, tb, v[:, i]) for i in range(3)], 1)
    ven[tc > tb[-1]] = 0.0
    extra = np.concatenate([ven / 2.0, ph["v_drag"][idx] / 2.0, ph["on"][idx, None].astype(float)], 1).astype(np.float32)
    return np.concatenate([X, extra], 1), yaw, tc, ph, q


def pool_one(n):
    f = f"{CACHE}/pool3_{n}.npz"
    if os.path.exists(f):
        return n
    room = load_room(n)
    lab = gt_labels(room)
    t = lab["t"] - lab["t"][0]
    rng = np.random.default_rng(7000 + sum(ord(c) for c in str(n)))
    Xs = []
    b0 = e2e.calib_for([x for x in e2e.SEQS if x != n]) if isinstance(n, str) else None
    for _ in range(NR):
        imu, ok = random_degrade(t, room["imu"], rng)
        X, yaw, tc, ph, q = build3(t, imu, ok, b0)
        Xs.append(X)
    k = len(Xs[0])
    idx = DEC // 2 + np.arange(k) * DEC
    np.savez(f, X=np.stack(Xs), v=lab["v"][idx].astype(np.float32), ok=lab["ok"][idx])
    return n


_P3 = {}


def get_pool3(n):
    if n not in _P3:
        d = np.load(f"{CACHE}/pool3_{n}.npz")
        _P3[n] = {k: d[k] for k in d.files}
    return _P3[n]


# the net and batch code read pools through e2e.get_pool
e2e.get_pool = get_pool3
_F = {}


def ctx3(n, s, m, sd, b0):
    key = (n, s, m, sd)
    if key not in _F:
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        X, yaw, tc, ph, q = build3(P["t"], P["imu"], P["ok"], b0)
        room = load_room(n)
        _F[key] = dict(P=P, q=q, X=X, yaw=yaw, tc=tc, ph=ph, room=room, sc=PosScorer(P, room, q))
    return _F[key]


if __name__ == "__main__":
    if sys.argv[1] == "make_pool":
        t0 = time.time()
        with mp.get_context("spawn").Pool(16) as p:
            for n in p.imap_unordered(pool_one, list(DEV_ROOMS) + e2e.SEQS):
                print("pool3", n, f"{time.time()-t0:.0f}s", flush=True)
        print("POOL3_DONE")
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--tag", default="s3")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--members", type=int, default=1)
    a = ap.parse_args()
    for fi, (test, train) in enumerate(seq.folds_for(a.scheme)):
        out = f"{RES}/{a.tag}_{a.scheme}_f{fi}.pkl"
        if os.path.exists(out):
            continue
        t0 = time.time()
        b0 = e2e.calib_for(train)
        _F.clear()
        nets = [seq.train_seq(train, 800 + fi + 31 * m, a.hidden, a.steps) for m in range(a.members)]
        torch.save(dict(states=[n.state_dict() for n in nets], hidden=a.hidden, train=train), f"{WGT}/{a.tag}_{a.scheme}_f{fi}.pt")
        Sxy, Sz = [], []
        for n in train:
            for f in (("clean", "delete", 1), ("mild", "zero", 1), ("full", "zero", 1)):
                c = ctx3(n, *f, b0)
                tk, v = seq.infer_seq(nets, c)
                x, y = eb.err_power(c, tk, v)
                Sxy.append(x)
                Sz.append(y)
        Sxy, Sz = np.median(Sxy, 0), np.median(Sz, 0)
        _F.clear()
        rows = {}
        for key in eval_set(test):
            c = ctx3(*key, b0)
            tk, v = seq.infer_seq(nets, c)
            o = {}
            for tau in (None, 40, 16, 8, 4):
                o[f"tau{tau}"] = c["sc"](seq.track(c, tk, v, tau))
            for al in (1, 2, 4):
                x, g = eb.eb_track(c, tk, v, Sxy, Sz, alpha=al)
                o[f"eb{al}"] = c["sc"](x)
            rows["|".join(map(str, key))] = o
        pickle.dump(dict(rows=rows, S=(Sxy, Sz), a=float(nets[0].a), b=float(nets[0].b)), open(out, "wb"))
        print(f"{a.tag} {a.scheme} fold {fi} test {test[:3]} {time.time()-t0:.0f}s a {float(nets[0].a):.2f} b {float(nets[0].b):.2f}", flush=True)
    print("S3_DONE", flush=True)
