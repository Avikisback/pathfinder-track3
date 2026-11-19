"""H5 two-sided gap bridging and H6 blind forecast, on held-out predictions only.

All displacement errors are measured in the heading frame at the start of the gap or failure, so global drift and
frame choice do not hide or inflate the local skill. Ground truth is used only to score, and for 'oracle' rows that
show the limit of a method if its boundary velocity were perfect.
"""
import argparse, glob, json, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

H = np.arange(0.1, 5.001, 0.1)
HOR = np.array([1, 2, 3, 4, 5.0])


def load_preds(tag):
    d = {}
    for f in glob.glob(f"{RES}/pred_{tag}_f*.pkl") + glob.glob(f"{RES}/pred_{tag}_r?.pkl"):
        d.update(pickle.load(open(f, "rb")))
    return d


class GT:
    def __init__(self, room):
        self.t = room["gt_t_ns"] * 1e-9
        self.p = room["gt_p"]
        self.q = room["gt_q"]
        self.rot = R.from_quat(self.q)
        self.sl = Slerp(self.t, self.rot)

    def pos(self, tt):
        return np.stack([np.interp(tt, self.t, self.p[:, i]) for i in range(3)], -1)

    def vel(self, t, h=0.05):
        return (self.pos(t + h) - self.pos(t - h)) / (2 * h)

    def yaw(self, t):
        q = self.sl(np.array([t])).as_quat()[0]
        return 2 * np.arctan2(q[2], q[3])

    def covered(self, a, b, max_hole=0.1):
        if a < self.t[0] or b > self.t[-1]:
            return False
        i, j = np.searchsorted(self.t, [a, b])
        return np.diff(self.t[max(i - 1, 0):j + 1]).max() < max_hole


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_of(q_wxyz):
    return 2 * np.arctan2(q_wxyz[3], q_wxyz[0])


def gaps_study(tag, att, schedules=("full", "mild")):
    preds = load_preds(tag)
    res = {}
    rot_res = {"decay": [], "lerp": [], "zero": []}
    seen = 0
    for key, o in preds.items():
        n, s, m, sd = key.split("|")
        if m != "zero" or s not in schedules:
            continue
        n = int(n) if n.isdigit() else n
        room = load_room(n)
        t_ns, imu, gaps = eval_file(n, s, m, int(sd))
        P = prepare(t_ns, imu)
        q = att(P)
        g = GT(room)
        t0 = P["t_ns"][0] * 1e-9
        ok_row = P["ok"]
        b = auto_bias(P["t"], P["imu"], P["ok"])
        qd = offline_attitude(P["t"], P["imu"], P["ok"], b0=b, fit_bias=False, gap_mode="decay")
        ql = offline_attitude(P["t"], P["imu"], P["ok"], b0=b, fit_bias=False, gap_mode="lerp")
        qz = offline_attitude(P["t"], P["imu"], P["ok"], b0=b, fit_bias=False, gap_mode="zero")
        for gs in gaps:
            a = gs * 1e-9
            e = a + 2.0
            if not g.covered(a - 0.2, e + 0.2):
                continue
            ta, te = a - t0, e - t0
            tt = np.arange(0.05, 2.0001, 0.05)
            true = g.pos(a + tt) - g.pos(np.array([a]))[0]
            Rg = rz(-g.yaw(a))
            true = (Rg @ true.T).T
            ia = np.searchsorted(P["t"], ta)
            Rq = rz(-yaw_of(q[ia]))
            tk, v = o["tk"], o["v"]
            k_a = max(np.searchsorted(tk, ta) - 1, 0)
            k_e = min(np.searchsorted(tk, te) + 1, len(tk) - 1)
            v0, v1 = Rq @ v[k_a], Rq @ v[k_e]
            vg0, vg1 = Rg @ g.vel(a - 0.05), Rg @ g.vel(e + 0.05)
            # net left running across the gap
            inside = np.stack([np.interp(ta + tt, tk, np.cumsum(v[:, i]) * 0.1) for i in range(3)], 1)
            inside = (Rq @ (inside - np.array([np.interp(ta, tk, np.cumsum(v[:, i]) * 0.1) for i in range(3)])).T).T
            tau = 0.75
            est = {
                "hold": np.zeros_like(true),
                "decay from net v": np.outer(tau * (1 - np.exp(-tt / tau)), v0),
                "net through gap": inside,
                "two-sided lerp, net v": np.outer(tt - tt ** 2 / 4, v0) + np.outer(tt ** 2 / 4, v1),
                "oracle decay, true v": np.outer(tau * (1 - np.exp(-tt / tau)), vg0),
                "oracle two-sided lerp, true v": np.outer(tt - tt ** 2 / 4, vg0) + np.outer(tt ** 2 / 4, vg1),
            }
            for name, x in est.items():
                res.setdefault(name, []).append((float(np.sqrt(((x - true) ** 2).sum(1).mean())), float(np.linalg.norm(x[-1] - true[-1])), str(n)))
            # rotation across the gap, relative
            ie = np.searchsorted(P["t"], te)
            Rt = g.sl(np.array([a, e]))
            dRt = (Rt[0].inv() * Rt[1])
            for nm, qq in (("decay", qd), ("lerp", ql), ("zero", qz)):
                r0, r1 = R.from_quat(qq[ia][[1, 2, 3, 0]]), R.from_quat(qq[ie][[1, 2, 3, 0]])
                rot_res[nm].append(float(np.degrees((dRt.inv() * (r0.inv() * r1)).magnitude())))
            seen += 1
    print(f"gaps scored {seen}")
    for name, v in res.items():
        a = np.array([x[0] for x in v])
        b = np.array([x[1] for x in v])
        print(f"  {name:34s} RMSE in gap: mean {a.mean():.3f}  rms {np.sqrt((a**2).mean()):.3f}  max {a.max():.3f} | endpoint: mean {b.mean():.3f}  max {b.max():.3f}")
    for nm, v in rot_res.items():
        v = np.array(v)
        print(f"  rotation across gap, rate {nm:6s} mean {v.mean():.1f} deg, median {np.median(v):.1f}, rms {np.sqrt((v**2).mean()):.1f}, max {v.max():.1f}", flush=True)


def blind_study(tag_causal, att, seqs, schedule="clean", tag_fc=None):
    """Failure placed every 1 s through each held-out file, and also at the file's real failure point."""
    preds = load_preds(tag_causal)
    fc = load_preds(tag_fc) if tag_fc else {}
    taus = [0.25, 0.5, 1, 2, 4]
    agg = {}
    def add(name, x, true, bucket):
        agg.setdefault((bucket, name), []).append((float(np.sqrt(((x - true) ** 2).sum(1).mean())), float(np.linalg.norm(x[-1] - true[-1]))))
    for n in seqs:
        key = f"{n}|{schedule}|delete|1"
        if key not in preds:
            continue
        o = preds[key]
        room = load_room(n)
        t_ns, imu, _ = eval_file(n, schedule, "delete", 1)
        P = prepare(t_ns, imu)
        q = att(P)
        g = GT(room)
        t0 = P["t_ns"][0] * 1e-9
        tk, v = o["tk"], o["v"]
        tend = P["t"][P["last_good"]]
        fails = np.arange(3.0, tend - 5.0, 1.0)
        for ta in list(fails) + [tend]:
            a = ta + t0
            bucket = "end" if ta == tend else "anywhere"
            if not g.covered(a - 0.1, a + 5.0):
                continue
            true = (rz(-g.yaw(a)) @ (g.pos(a + H) - g.pos(np.array([a]))[0]).T).T
            ia = np.searchsorted(P["t"], ta)
            Rq = rz(-yaw_of(q[min(ia, len(q) - 1)]))
            k = max(np.searchsorted(tk, ta) - 1, 0)
            v0 = Rq @ v[k]
            vg = rz(-g.yaw(a)) @ g.vel(a - 0.05)
            add("hold", np.zeros_like(true), true, bucket)
            add("constant velocity, net v", np.outer(H, v0), true, bucket)
            for tau in taus:
                add(f"damped tau {tau}, net v", np.outer(tau * (1 - np.exp(-H / tau)), v0), true, bucket)
                add(f"oracle damped tau {tau}, true v", np.outer(tau * (1 - np.exp(-H / tau)), vg), true, bucket)
            if key in fc:
                f = fc[key]
                kf = max(np.searchsorted(f["tk"], ta) - 1, 0)
                d = (Rq @ f["d"][kf].reshape(5, 3).T).T if False else f["d"][kf].reshape(5, 3)
                x = np.stack([np.interp(H, np.r_[0, HOR], np.r_[0, d[:, i]]) for i in range(3)], 1)
                add("tiny learned forecast", x, true, bucket)
    for bucket in ("anywhere", "end"):
        print(f" failure {bucket}")
        names = sorted({n for b, n in agg if b == bucket})
        for name in names:
            v = np.array(agg[(bucket, name)])
            print(f"  {name:34s} RMSE: rms {np.sqrt((v[:,0]**2).mean()):.3f} mean {v[:,0].mean():.3f} | endpoint rms {np.sqrt((v[:,1]**2).mean()):.3f} | n {len(v)}", flush=True)


if __name__ == "__main__":
    from scipy.spatial.transform import Slerp
    base = lambda P: baseline_attitude(P["t"], P["imu"], P["ok"])
    auto = lambda P: attitude_auto(P["t"], P["imu"], P["ok"])
    print("=== H5 EuRoC, net eu2s loso"); gaps_study("eu2s_loso", auto)
    print("=== H5 TUM dev, net center2s"); gaps_study("center2s", base)
    fc = sys.argv[1] if len(sys.argv) > 1 else None
    print("=== H6 EuRoC, causal net eu2sc loso"); blind_study("eu2sc_loso", auto, list(EUROC), tag_fc=(fc + "_euroc") if fc else None)
    print("=== H6 TUM dev, causal net causal2s"); blind_study("causal2s", base, DEV_ROOMS, tag_fc=(fc + "_tum") if fc else None)
