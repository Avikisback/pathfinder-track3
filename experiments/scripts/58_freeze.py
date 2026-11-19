"""Choose the self-check thresholds per platform from the drift shapes seen so far, freeze them, then test on one new shape
that has played no part in any choice: bias that jumps in steps.
"""
import importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m57 = importlib.import_module("57_nonet")
m41, m48, shapes, eb = m57.m41, m57.m48, m57.shapes, m57.eb
from phys import *

LVL = "L15"
SEQS = m57.SEQS
DT = m57.DT


def steps_file(room, sd):
    rng = np.random.default_rng(sd + 500)
    t = room["t_ns"]
    keep = t <= int(t[-1]) - 5 * 10 ** 9
    t = t[keep]
    v = room["imu"][keep].copy()
    ts = (t - t[0]) * 1e-9
    n, T = len(t), ts[-1]
    bg, ba = np.zeros((n, 3)), np.zeros((n, 3))
    s = rng.uniform(5, 30)
    cg, ca = np.zeros(3), np.zeros(3)
    while s < T:
        m = ts >= s
        cg = rng.uniform(-0.015, 0.015, 3)
        ca = rng.uniform(-0.15, 0.15, 3)
        bg[m] = cg
        ba[m] = ca
        s += rng.uniform(20, 40)
    v[:, :3] += bg + rng.standard_normal((n, 3)) * 0.006
    v[:, 3:] += ba + rng.standard_normal((n, 3)) * 0.06
    g = np.zeros(n, bool)
    for _ in range(3):
        a = rng.uniform(0.2 * T, T - 3)
        g |= (ts >= a) & (ts < a + 2.0)
    return t[~g], v[~g]


def solve_steps(key):
    n, sd = key
    f = f"{m57.D8}/{n}_steps_x_{sd}_{LVL}.pkl"
    if not os.path.exists(f):
        t, imu = steps_file(load_room(n), sd)
        P = prepare(t, imu)
        q, v, tb, last, info = m48.solve_level(P, m41.calib_excl(n), m48.LEVELS[LVL])
        tk = np.arange(tb[0], tb[-1], DT)
        vk = np.stack([np.interp(tk, tb, v[:, i]) for i in range(3)], 1)
        pickle.dump(dict(q=q.astype(np.float32), tk=tk, v=vk, drone=info["drone"]), open(f, "wb"))
    return key


def hp(d, tau):
    v = d["v"]
    vm = v - v.mean(0)
    X, f = eb.spec(vm)
    G = (2 * np.pi * f * tau) ** 2 / (1 + (2 * np.pi * f * tau) ** 2)
    return np.fft.irfft(X * G[:, None], n=2 * len(vm), axis=0)[: len(vm)]


def final_v(d, fit, thr_d, thr_h, tau_d, tau_h):
    r = m57.slow_rms(d)
    if d["drone"]:
        return hp(d, tau_d) if r > thr_d else m57.apply(d, fit, "eb")
    return hp(d, tau_h) if r > thr_h else m57.apply(d, fit, "band")


def run(names, folds, kinds, cfg):
    rooms = {n: load_room(n) for n in names}
    rows = {}
    for test, train in folds:
        fit = m57.spec_fit([(n, m57.load(n, *k, LVL)) for n in train for k in m57.KNOWN], rooms)
        for n in test:
            for k in kinds:
                room = rooms[n]
                if k[0] == "steps":
                    t, imu = steps_file(room, k[2])
                else:
                    t, imu, _r = m57.get_file(n, *k)
                P = prepare(t, imu)
                d = m57.load(n, *k, LVL)
                sc = PosScorer(P, room, d["q"].astype(np.float64))
                o = dict(stationary=sc(np.zeros((len(P["t"]), 3))), rvs=m57.slow_rms(d))
                o["final"] = sc(m57.track(P, d, final_v(d, fit, *cfg)))
                rows[(n, k[0], k[1], k[2])] = o
    return rows


def show(rows, title):
    print(f"\n=== {title}")
    for kind in ("clean", "mild", "full", "burst", "randomwalk", "sine", "steps"):
        ks = [k for k in rows if k[1] == kind]
        if not ks:
            continue
        s = [rows[k]["stationary"] for k in ks]
        f = [rows[k]["final"] for k in ks]
        print(f"  {kind:10s} ({len(ks):2d}) " + " | ".join(f"{m} {np.nanmean([x[m] for x in f]):.2f} vs {np.nanmean([x[m] for x in s]):.2f} (worst {np.nanmax([x[m] for x in f]):.2f})" for m in ("ate_se3", "ate_org", "rpe_t", "blind_rel"))
              + f" | beats stationary on ATE {np.mean([a['ate_se3'] < b['ate_se3'] for a, b in zip(f, s)]):.2f}, RPE {np.mean([a['rpe_t'] < b['rpe_t'] for a, b in zip(f, s)]):.2f}", flush=True)


if __name__ == "__main__":
    EU = [(v, [x for x in SEQS if x not in v]) for v in ENV.values()]
    TU = [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
    # the self-check values per file
    for nm, names in (("EuRoC", SEQS), ("TUM", DEV_ROOMS)):
        for kind in ("known", "burst", "randomwalk", "sine"):
            ks = m57.KNOWN if kind == "known" else [(kind, "x", 1), (kind, "x", 2)]
            vals = [m57.slow_rms(m57.load(n, *k, LVL)) for n in names for k in ks]
            print(f"{nm:5s} {kind:10s} slow rms min {np.min(vals):.2f} median {np.median(vals):.2f} max {np.max(vals):.2f}")
    # search on the shapes already seen
    best = None
    kinds = m57.KNOWN + m57.UNSEEN
    for thr_d in (0.7, 0.8, 0.9, 1.0, 1.2):
        for tau_d in (2.0, 4.0, 8.0):
            r = run(SEQS, EU, kinds, (thr_d, 0.5, tau_d, 2.0))
            known = np.mean([r[k]["final"]["ate_se3"] for k in r if k[1] in ("clean", "mild", "full", "burst")])
            hostile = np.mean([r[k]["final"]["ate_se3"] for k in r if k[1] in ("randomwalk", "sine")])
            print(f"drone thr {thr_d} tau {tau_d}: known {known:.3f} hostile {hostile:.3f}", flush=True)
            sc_ = known + 0.25 * hostile
            if best is None or sc_ < best[0]:
                best = (sc_, thr_d, tau_d)
    print("chosen drone", best)
    best_h = None
    for thr_h in (0.4, 0.5, 0.6, 9.9):
        for tau_h in (1.0, 2.0):
            r = run(DEV_ROOMS, TU, kinds, (1.0, thr_h, 4.0, tau_h))
            known = np.mean([r[k]["final"]["ate_se3"] for k in r if k[1] in ("clean", "mild", "full", "burst")])
            hostile = np.mean([r[k]["final"]["ate_se3"] for k in r if k[1] in ("randomwalk", "sine")])
            print(f"hand thr {thr_h} tau {tau_h}: known {known:.3f} hostile {hostile:.3f}", flush=True)
            sc_ = known + 0.25 * hostile
            if best_h is None or sc_ < best_h[0]:
                best_h = (sc_, thr_h, tau_h)
    print("chosen hand", best_h)
    cfg = (best[1], best_h[1], best[2], best_h[2])
    # new shape, played no part in any choice above
    keys = [(n, sd) for n in SEQS + DEV_ROOMS + [FINAL_ROOM] for sd in (1, 2)]
    with mp.get_context("spawn").Pool(16) as p:
        for _ in p.imap_unordered(solve_steps, keys):
            pass
    steps = [("steps", "x", 1), ("steps", "x", 2)]
    r1 = run(SEQS, EU, kinds + steps, cfg)
    show(r1, f"FROZEN {cfg}: EuRoC, unseen environment")
    r2 = run(DEV_ROOMS, TU, kinds + steps, cfg)
    show(r2, f"FROZEN {cfg}: TUM, unseen room")
    r3 = run(DEV_ROOMS + [FINAL_ROOM], [([FINAL_ROOM], DEV_ROOMS)], kinds + steps, cfg)
    show(r3, f"FROZEN {cfg}: room 6 (seen once before, by an earlier version)")
    pickle.dump(dict(cfg=cfg, eu=r1, tum=r2, room6=r3), open(f"{RES}/nonet_frozen.pkl", "wb"))
