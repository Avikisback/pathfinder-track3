"""Final checks with the configuration frozen: solver, whole-trajectory net, self-adjusting Wiener at alpha 1, near-hold tail.

  dev     leave-one-environment-out EuRoC, leave-one-room-out TUM, known schedules
  unseen  same held-out nets on drift patterns they never trained or tuned on
  room6   train on rooms 1-5, test on the locked room 6, once
  evo     confirm the fast scorer against the real evo tools on a few files

Usage: 55_final.py dev|unseen|room6|evo
"""
import importlib, json, os, pickle, subprocess, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
s4 = importlib.import_module("54_seq4")
from phys import *

e2e, seq, eb, m46 = s4.e2e, s4.seq, s4.eb, s4.m46
TAIL_TAU, TAIL_GAIN = 0.35, 0.5
KEYS = ("ate_se3", "ate_org", "rpe_t", "rpe_r", "att_org", "blind_rel", "ate_se3_pre")


def degrade_custom(room, seed, kind):
    """Drift shapes that no training pool or setting has seen."""
    rng = np.random.default_rng(seed + 100)
    t = room["t_ns"]
    t1 = int(t[-1]) - 5 * 10 ** 9
    keep = t <= t1
    t = t[keep]
    v = room["imu"][keep].copy()
    ts = (t - t[0]) * 1e-9
    n, T = len(t), ts[-1]
    fs = 1.0 / np.median(np.diff(ts))

    def gaps(num, length, lo=0.2):
        mask = np.zeros(n, bool)
        for _ in range(num):
            s = rng.uniform(lo * T, T - length - 1)
            mask |= (ts >= s) & (ts < s + length)
        return mask

    if kind == "randomwalk":
        bg = np.cumsum(rng.standard_normal((n, 3)), 0)
        bg = bg - bg.mean(0)
        bg *= 0.012 / bg.std(0)
        ba = np.cumsum(rng.standard_normal((n, 3)), 0)
        ba = ba - ba.mean(0)
        ba *= 0.12 / ba.std(0)
        sg, sa = np.full(n, 0.006), np.full(n, 0.06)
        g, mode = gaps(3, 2.0), "delete"
    elif kind == "sine":
        ph = rng.uniform(0, 6.28, 6)
        per = rng.uniform(40, 120, 6)
        bg = 0.02 * np.stack([np.sin(2 * np.pi * ts / per[i] + ph[i]) for i in range(3)], 1)
        ba = 0.2 * np.stack([np.sin(2 * np.pi * ts / per[i + 3] + ph[i + 3]) for i in range(3)], 1)
        sg, sa = np.full(n, 0.004), np.full(n, 0.04)
        g, mode = gaps(3, 2.0), "zero"
    elif kind == "burst":
        x = (ts / T)[:, None]
        bg = 0.01 * x * rng.uniform(-1, 1, 3)
        ba = 0.1 * x * rng.uniform(-1, 1, 3)
        sg, sa = np.full(n, 0.004), np.full(n, 0.04)
        for _ in range(4):
            s = rng.uniform(0.1 * T, T - 4)
            m = (ts >= s) & (ts < s + 3)
            sg[m], sa[m] = 0.04, 0.4
        g = gaps(6, 0.5) | gaps(1, 2.0)
        mode = "zero"
    else:
        raise ValueError(kind)
    v[:, :3] += bg + rng.standard_normal((n, 3)) * sg[:, None]
    v[:, 3:] += ba + rng.standard_normal((n, 3)) * sa[:, None]
    if mode == "delete":
        return t[~g], v[~g]
    v[g] = 0.0
    return t, v


def stack_track(c, nets, S):
    tk, v = seq.infer_seq(nets, c)
    x, _ = eb.eb_track(c, tk, v, S[0], S[1], alpha=1.0, tail_tau=TAIL_TAU, tail_gain=TAIL_GAIN)
    return x


def solver_track(c, vs, tb, last):
    P = c["P"]
    fs = 1.0 / np.median(np.diff(tb))
    t_last = tb[-1]
    tt = P["t"]
    px = np.cumsum(vs, axis=0) / fs
    out = np.stack([np.interp(tt, tb, px[:, i]) for i in range(3)], 1)
    tail = tt > t_last
    if tail.any():
        v0 = vs[-int(0.5 * fs) :].mean(0)
        s = tt[tail] - t_last
        out[tail] = px[-1] + TAIL_GAIN * v0 * (TAIL_TAU * (1 - np.exp(-s / TAIL_TAU)))[:, None]
    return out


def eval_one(t, imu, room, b0, nets, S):
    P = prepare(t, imu)
    q, vs, tb, last, info = m46.solve_any(P["t"], P["imu"], P["ok"], b0=b0)
    X, yaw, tc = features(P["t"], P["imu"], P["ok"], q)
    ph = phys_channels(P["t"], P["imu"], P["ok"], q)
    n = len(X)
    idx = DEC // 2 + np.arange(n) * DEC
    ven = np.stack([np.interp(tc, tb, vs[:, i]) for i in range(3)], 1)
    ven[tc > tb[-1]] = 0.0
    extra = np.concatenate([ven / 2.0, ph["v_drag"][idx] / 2.0, ph["on"][idx, None].astype(float)], 1).astype(np.float32)
    c = dict(P=P, q=q, X=np.concatenate([X, extra], 1), yaw=yaw, tc=tc, ph=ph, room=room)
    o = {}
    o["stationary"] = score_all(P, room, np.zeros((len(P["t"]), 3)), q)
    o["solver"] = score_all(P, room, solver_track(c, vs, tb, last), q)
    o["stack"] = score_all(P, room, stack_track(c, nets, S), q)
    return o, c


def load_fold(tag, scheme, fi):
    ck = torch.load(f"{WGT}/{tag}_{scheme}_f{fi}.pt", weights_only=False)
    nets = []
    for st in ck["states"]:
        net = seq.SeqNet(hidden=ck["hidden"]).to("cuda")
        net.load_state_dict(st)
        net.eval()
        nets.append(net)
    d = pickle.load(open(f"{RES}/{tag}_{scheme}_f{fi}.pkl", "rb"))
    return nets, d["S"], ck["train"]


def table(res, title):
    print(f"\n=== {title}: {len(res)} files")
    groups = sorted({k.split("|")[1] for k in res})
    for g in groups:
        ks = [k for k in res if k.split("|")[1] == g]
        print(f"-- {g} ({len(ks)} files)")
        for m in ("stationary", "solver", "stack"):
            print(f"  {m:10s} " + " ".join(f"{x} {np.nanmean([res[k][m][x] for k in ks]):.3f}/{np.nanmax([res[k][m][x] for k in ks]):.3f}" for x in KEYS))
        w = {x: round(float(np.mean([res[k]['stack'][x] < res[k]['stationary'][x] for k in ks])), 2) for x in ("ate_se3", "ate_org", "rpe_t", "blind_rel")}
        print(f"  stack beats stationary on share of files: {w}")
    sys.stdout.flush()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("dev", "unseen"):
        for scheme, label in (("loeo", "EuRoC, unseen environment"), ("loro", "TUM rooms, unseen room")):
            res = {}
            for fi, (test, train) in enumerate(seq.folds_for(scheme)):
                nets, S, _ = load_fold("s4", scheme, fi)
                b0 = e2e.calib_for(train)
                if cmd == "dev":
                    for key in eval_set(test, seeds=(1, 2, 3), modes=("zero", "delete")):
                        t, imu, _ = eval_file(*key)
                        res["|".join(map(str, key))], _c = eval_one(t, imu, load_room(key[0]), b0, nets, S)
                else:
                    for n in test:
                        room = load_room(n)
                        for kind in ("randomwalk", "sine", "burst"):
                            for sd in (1, 2):
                                t, imu = degrade_custom(room, sd, kind)
                                res[f"{n}|{kind}|{sd}"], _c = eval_one(t, imu, room, b0, nets, S)
            pickle.dump(res, open(f"{RES}/final_{cmd}_{scheme}.pkl", "wb"))
            table(res, f"{cmd} {label}")
    elif cmd == "room6":
        train = list(DEV_ROOMS)
        s4.pool_one(6) if False else None
        out = f"{WGT}/s4_final_tum.pt"
        if os.path.exists(out):
            ck = torch.load(out, weights_only=False)
            net = seq.SeqNet(hidden=48).to("cuda")
            net.load_state_dict(ck["state"])
            net.eval()
            S = ck["S"]
        else:
            net = seq.train_seq(train, 811, 48, 2500)
            Sxy, Sz = [], []
            for n in train:
                for f in (("clean", "delete", 1), ("mild", "zero", 1), ("full", "zero", 1)):
                    c = s4.ctx3(n, *f, None)
                    tk, v = seq.infer_seq([net], c)
                    a, b = eb.err_power(c, tk, v)
                    Sxy.append(a)
                    Sz.append(b)
            S = (np.median(Sxy, 0), np.median(Sz, 0))
            torch.save(dict(state=net.state_dict(), S=S), out)
        room = load_room(6)
        res = {}
        for key in eval_set([6], seeds=(1, 2, 3), modes=("zero", "delete")):
            t, imu, _ = eval_file(*key)
            res["|".join(map(str, key))], _c = eval_one(t, imu, room, None, [net], S)
        for kind in ("randomwalk", "sine", "burst"):
            for sd in (1, 2):
                t, imu = degrade_custom(room, sd, kind)
                res[f"6|{kind}|{sd}"], _c = eval_one(t, imu, room, None, [net], S)
        pickle.dump(res, open(f"{RES}/final_room6.pkl", "wb"))
        table(res, "LOCKED room 6, first and only look")
    elif cmd == "evo":
        for scheme, fi, key in (("loeo", 0, ("MH04", "full", "zero", 1)), ("loro", 2, (3, "full", "zero", 1))):
            nets, S, train = load_fold("s4", scheme, fi)
            b0 = e2e.calib_for(train)
            t, imu, _ = eval_file(*key)
            room = load_room(key[0])
            o, c = eval_one(t, imu, room, b0, nets, S)
            P = c["P"]
            x = stack_track(c, nets, S)
            et, ep, eq = finish(P, x, c["q"])
            d = f"{TMP}/evo_{key[0]}"
            os.makedirs(d, exist_ok=True)
            write_tum(f"{d}/est.txt", et, ep, eq)
            write_tum(f"{d}/gt.txt", room["gt_t_ns"], room["gt_p"], room["gt_q"])
            evo = ROOT + "/.venv-wsl/bin"
            frames = int(round(1.0 / np.median(np.diff(room["gt_t_ns"] * 1e-9))))
            for cmdl, lab in (([f"{evo}/evo_ape", "tum", f"{d}/gt.txt", f"{d}/est.txt", "-a", "--t_max_diff", "0.01"], "ate_se3"),
                              ([f"{evo}/evo_ape", "tum", f"{d}/gt.txt", f"{d}/est.txt", "--align_origin", "--t_max_diff", "0.01"], "ate_org"),
                              ([f"{evo}/evo_rpe", "tum", f"{d}/gt.txt", f"{d}/est.txt", "-r", "trans_part", "--delta", str(frames), "--delta_unit", "f", "--t_max_diff", "0.01"], "rpe_t")):
                r = subprocess.run(cmdl, capture_output=True, text=True)
                rm = [l for l in r.stdout.splitlines() if l.strip().startswith("rmse")]
                print(key[0], lab, "evo", rm[0].split()[-1] if rm else r.stdout[-200:] + r.stderr[-200:], "ours", round(o["stack"][lab], 4), flush=True)
