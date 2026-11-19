"""Two questions the post-hoc budget cannot answer.

A. How good is the drag velocity at each time scale, on clean data with the true attitude? Gain and coherence per octave band, and the
   ATE from integrating drag alone. If drag is good at 1/64..1/16 Hz, the final velocity should take that band from drag.
B. If the solver is rerun with the true attitude, how far does ATE fall? Tilt error enters velocity through integration, so rotating the
   final velocity after the fact (what the budget does) cannot show this.

Analysis only. Uses ground truth throughout.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
import gls
from scipy.spatial.transform import Slerp
m46 = m57.m46


def truth_q_rows(P, room):
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    return Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(P["t"], gt_t[0], gt_t[-1])).as_quat()[:, [3, 0, 1, 2]]


def drag_band(n, fc=0.1):
    room = load_room(n)
    t_ns, imu = room["t_ns"], room["imu"]
    P = prepare(t_ns, imu)
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=np.zeros(3))
    ec.set_target(fc=fc)
    gt_t = (room["gt_t_ns"] - t_ns[0]) * 1e-9
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(ec.tb, gt_t[0], gt_t[-1])).as_matrix()
    f = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    vg = (f(ec.tb + 0.05) - f(ec.tb - 0.05)) / 0.1
    # body-plane truth
    vbt = np.einsum("nji,nj->ni", Rg, vg)
    E = ec.E
    pt = vbt @ E
    pd = ec.vb_t @ E
    on = ec.on_b
    bl = 25
    N = len(pt) // bl
    dn = lambda x: x[: N * bl].reshape((N, bl) + x.shape[1:]).mean(1)
    Yt, Yd = gls.dct_rows(dn(pt)), gls.dct_rows(dn(pd))
    fr = np.arange(N) / (2 * N * gls.RDT)
    b = gls.band_of(fr)
    gain, coh, pwr_t, pwr_e = np.full(gls.NB, np.nan), np.full(gls.NB, np.nan), np.full(gls.NB, np.nan), np.full(gls.NB, np.nan)
    for i in range(gls.NB):
        m = b == i
        if m.any():
            a, c, d = (Yt[m] * Yd[m]).sum(), (Yd[m] ** 2).sum(), (Yt[m] ** 2).sum()
            gain[i] = a / max(c, 1e-12)
            coh[i] = a * a / max(c * d, 1e-12)
            pwr_t[i], pwr_e[i] = d / m.sum(), ((Yt[m] - Yd[m]) ** 2).sum() / m.sum()
    # scoring position from drag alone, in the true attitude; plus perfect high frequencies as a limit
    vd_w = np.einsum("nij,nj->ni", Rg, ec.vb_t) * on[:, None]
    fs = 1 / np.median(ec.dtb)
    ate_drag, st = m41.ate(np.cumsum(vd_w, axis=0) / fs, ec.tb, room)
    k_fit = float((pt * pd).sum() / max((pd * pd).sum(), 1e-9))
    return dict(n=n, gain=gain, coh=coh, pt=pwr_t, pe=pwr_e, ate_drag=ate_drag, st=st, kscale=k_fit)


def rerun_true_att(a):
    n, k, lvl = a
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    lv = m48.LEVELS[lvl]
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=m41.calib_excl(n), knot_a=lv["knot"], knot_g=lv["knot"])
    ec.set_target()
    qt = truth_q_rows(P, room)
    ec.gyro_q = lambda x: qt
    ec.x["S"][:] = 0.0
    ec.x["cg"][:] = 0.0
    ec._seed_tilts(20.0)
    if ec.drone:
        q, v, tb, last, info = m46.solve_drag(ec, sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_a=lv["sig_a"], sig_g=1e-5)
    else:
        q, v, tb, last, info = ec.solve(sig_a=lv["sig_a"], sig_g=1e-5, sig_p=5.0)
    d = to_cache(q, v, tb, bool(ec.drone))
    sc = PosScorer(P, room, q)
    raw = sc(m57.track(P, d, d["v"] - d["v"].mean(0)))
    return (n, *k, lvl), raw["ate_se3"], sc(np.zeros((len(P["t"]), 3)))["ate_se3"]


if __name__ == "__main__":
    t0 = time.time()
    print("=== A. drag velocity against truth, by octave band, clean rows, true attitude. Band top frequency:", np.round(gls.EDG[1:], 4))
    with mp.get_context("spawn").Pool(11) as p:
        res = p.map(drag_band, SEQS)
    print("mean gain ", np.round(np.nanmean([r["gain"] for r in res], 0), 2))
    print("mean coh  ", np.round(np.nanmean([r["coh"] for r in res], 0), 2))
    print("truth pow ", np.round(np.nanmean([r["pt"] for r in res], 0), 3))
    print("error pow ", np.round(np.nanmean([r["pe"] for r in res], 0), 3))
    for r in res:
        print(f"  {r['n']:5s} ATE from drag alone with true attitude {r['ate_drag']:.2f} (stationary {r['st']:.2f}), scale {r['kscale']:.2f}; coh " + " ".join(f"{c:.2f}" for c in r["coh"]))
    print(f"mean ATE drag alone {np.mean([r['ate_drag'] for r in res]):.2f} vs stationary {np.mean([r['st'] for r in res]):.2f}", flush=True)
    print("\n=== B. solver rerun with the true attitude: raw ATE")
    jobs = [(n, k, lvl) for lvl in ("L60", "L15") for n in SEQS + DEV_ROOMS for k in (("clean", "delete", 1), ("full", "zero", 1), ("sine", "x", 1), ("randomwalk", "x", 1))]
    with mp.get_context("spawn").Pool(16) as p:
        rows = p.map(rerun_true_att, jobs, chunksize=2)
    for plat, names in (("EuRoC", SEQS), ("TUM", DEV_ROOMS)):
        for lvl in ("L60", "L15"):
            for kd in ("clean", "full", "sine", "randomwalk"):
                s = [r for r in rows if r[0][0] in names and r[0][1] == kd and r[0][4] == lvl]
                print(f"  {plat:5s} {lvl} {kd:10s} ATE {np.mean([x[1] for x in s]):.2f} worst {np.max([x[1] for x in s]):.2f} stationary {np.mean([x[2] for x in s]):.2f}", flush=True)
    pickle.dump(dict(drag=res, rerun=rows), open(f"{RES}/oracles.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
