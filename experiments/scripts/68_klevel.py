"""Two checks after the first oracles.

1. Drag gain per band with rotors on only (masking the time on the ground), in the true attitude.
2. The drag constant. k = 0.187 came from an earlier fit, but the per-file scale fit keeps asking for a target 2-3 times smaller. Run the
   frozen solver at L15 with other k values, raw and final ATE, known shapes only.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
import gls
from scipy.spatial.transform import Slerp
m46 = m57.m46
D11 = EXP + "/cache/v11"
os.makedirs(D11, exist_ok=True)


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
    vbt = np.einsum("nji,nj->ni", Rg, vg)
    E = ec.E
    on = ec.on_b & (ec.tb > gt_t[0] + 0.5) & (ec.tb < gt_t[-1] - 0.5)
    # longest run with rotors on
    idx = np.nonzero(np.diff(np.r_[0, on.astype(int), 0]))[0].reshape(-1, 2)
    a, b = idx[np.argmax(idx[:, 1] - idx[:, 0])]
    pt = (vbt @ E)[a:b]
    pd = (ec.vb_t @ E)[a:b]
    bl = 25
    N = len(pt) // bl
    dn = lambda x: x[: N * bl].reshape((N, bl) + x.shape[1:]).mean(1)
    tt, dd = dn(pt), dn(pd)
    Yt, Yd = gls.dct_rows(tt - tt.mean(0)), gls.dct_rows(dd - dd.mean(0))
    fr = np.arange(N) / (2 * N * gls.RDT)
    b_ = gls.band_of(fr)
    gain, coh, pt_, pe = [np.full(gls.NB, np.nan) for _ in range(4)]
    for i in range(gls.NB):
        m = b_ == i
        if m.any():
            aa, c, d = (Yt[m] * Yd[m]).sum(), (Yd[m] ** 2).sum(), (Yt[m] ** 2).sum()
            gain[i], coh[i] = aa / max(c, 1e-12), aa * aa / max(c * d, 1e-12)
            pt_[i], pe[i] = d / m.sum(), ((Yt[m] - Yd[m]) ** 2).sum() / m.sum()
    # overall fit of a scale with flight mean removed
    tv, dv = tt - tt.mean(0), dd - dd.mean(0)
    return dict(n=n, gain=gain, coh=coh, pt=pt_, pe=pe, scale=float((tv * dv).sum() / max((dv * dv).sum(), 1e-9)), r=float((tv * dv).sum() / np.sqrt((tv * tv).sum() * (dv * dv).sum())), offs=(ec.vb_t[on] ** 2).sum(1).mean() ** 0.5)


def one(a):
    n, k, kd = a
    f = f"{D11}/{n}_{k[0]}_{k[1]}_{k[2]}_{kd}.pkl"
    if os.path.exists(f):
        return a
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    lv = m48.LEVELS["L15"]
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=m41.calib_excl(n), knot_a=lv["knot"], knot_g=lv["knot"])
    ec.set_target(k=kd)
    q, v, tb, last, info = m46.solve_drag(ec, sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_a=lv["sig_a"], sig_g=lv["sig_g"])
    pickle.dump(to_cache(q, v, tb, True), open(f, "wb"))
    return a


if __name__ == "__main__":
    t0 = time.time()
    np.set_printoptions(linewidth=200, suppress=True)
    with mp.get_context("spawn").Pool(11) as p:
        res = p.map(drag_band, SEQS)
    print("band top", np.round(gls.EDG[1:], 4))
    print("mean gain ", np.round(np.nanmean([r["gain"] for r in res], 0), 2))
    print("mean coh  ", np.round(np.nanmean([r["coh"] for r in res], 0), 2))
    print("truth pow ", np.round(np.nanmean([r["pt"] for r in res], 0), 3))
    print("error pow ", np.round(np.nanmean([r["pe"] for r in res], 0), 3))
    for r in res:
        print(f"  {r['n']:5s} scale {r['scale']:.2f} corr {r['r']:.2f} drag rms {r['offs']:.2f} gain " + " ".join(f"{c:5.2f}" for c in r["gain"]) + " | coh " + " ".join(f"{c:.2f}" for c in r["coh"]), flush=True)
    KS = (0.187, 0.3, 0.45, 0.7, 1.2)
    kinds = KNOWN + UNSEEN + STEPS
    jobs = [(n, k, kd) for kd in KS for n in SEQS for k in kinds]
    with mp.get_context("spawn").Pool(16) as p:
        for _ in p.imap_unordered(one, jobs, chunksize=2):
            pass
    print(f"solved {time.time()-t0:.0f}s", flush=True)
    out = {}
    for kd in KS:
        L = lambda n, k, kd=kd: pickle.load(open(f"{D11}/{n}_{k[0]}_{k[1]}_{k[2]}_{kd}.pkl", "rb"))
        r = evaluate(SEQS, EU_FOLDS, kinds, L)
        out[kd] = r
        print(f"k={kd}: raw " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, "raw").items()))
        print(f"       nocheck " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, "nocheck").items()))
        print(f"       final " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, "final").items()), flush=True)
    pickle.dump(out, open(f"{RES}/klevel.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
