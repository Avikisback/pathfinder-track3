"""EuRoC survey and attitude. Are there still starts? How big is gyro bias? What does the baseline do with it?
Nothing estimator-side uses ground truth, except 'constant from other sequences', which is a sensor calibration
fitted on the OTHER sequences' ground truth (leave-one-sequence-out and leave-one-environment-out)."""
import json, sys, time, subprocess
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *
from common import _read_csv_ns
from scipy.spatial.transform import Slerp

SEQS = list(EUROC)
# real tool against the in-memory copy, on EuRoC
src = euroc_dir("MH02") + "/imu0/data.csv"
for mode in ("delete", "zero"):
    p = f"{TMP}/eu.csv"
    subprocess.run(["python3", f"{ROOT}/tools/degrade.py", src, p, "--seed", "2", "--gap-mode", mode], check=True, stdout=subprocess.DEVNULL)
    t2, v2 = _read_csv_ns(p)
    t1, v1, _ = degrade(load_room("MH02"), 2, mode)
    print(f"degrade check MH02 {mode}: same stamps {np.array_equal(t1,t2)} max diff {np.abs(v1-v2).max():.1e}", flush=True)

print("\nSURVEY seq dur path radius speed | max dt | still: secs, first at, quietest s gyro std, first s gyro std, gt start | gt bias median | still-mean minus gt bias | gravity-fit minus gt bias, sig | hand-eye deg, clock ms, world accel")
gtb = {}
for s in SEQS:
    d = load_room(s)
    t = (d["t_ns"] - d["t_ns"][0]) * 1e-9
    imu = d["imu"]; ok = np.ones(len(t), bool)
    gt_t = (d["gt_t_ns"] - d["t_ns"][0]) * 1e-9
    b = np.median(d["gt_bw"], 0)
    gtb[s] = b
    sb, n_still, first = still_bias(t, imu, ok)
    fs = 200
    stds = [np.sqrt(imu[i:i + fs, :3].var(0).mean()) for i in range(0, len(t) - fs, fs)]
    gb, sig, V = gravity_bias(t, imu, ok, return_all=True)
    Rq = R.from_quat(d["gt_q"])
    grid = np.arange(max(t[0], gt_t[0]) + 1, min(t[-1], gt_t[-1]) - 1, 0.05)
    Rg = Slerp(gt_t, Rq)(grid)
    w_gt = (Rg[:-1].inv() * Rg[1:]).as_rotvec() / 0.05
    tm = grid[:-1] + 0.025
    w_i = np.stack([np.interp(tm, t, imu[:, i]) for i in range(3)], 1) - b
    u_, _, vt = np.linalg.svd(w_gt.T @ w_i)
    Rx = u_ @ np.diag([1, 1, np.linalg.det(u_ @ vt)]) @ vt
    best = max(((w_gt * np.stack([np.interp(tm + o, t, imu[:, i]) - b[i] for i in range(3)], 1)).sum(), o) for o in np.arange(-0.03, 0.0301, 0.0025))
    aw = Rg[:-1].apply(np.stack([np.interp(tm, t, imu[:, i + 3]) for i in range(3)], 1)).mean(0)
    p = d["gt_p"]
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    print(f"{s} {t[-1]:.1f}s path {path:.1f} rad {np.sqrt(((p-p.mean(0))**2).sum(1).mean()):.2f} v {path/(gt_t[-1]-gt_t[0]):.2f} | maxdt {np.diff(t).max()*1e3:.1f}ms | still {n_still:.0f}s first {first} quiet {min(stds):.4f} first1s {stds[0]:.4f} gt_start {gt_t[0]:.2f}s | bias {np.round(b,4)} "
          f"| still-gt {np.round(sb-b,4) if sb is not None else None} | grav-gt {np.round(gb-b,4)} sig {np.round(sig,4)} | R {np.degrees(R.from_matrix(Rx).magnitude()):.2f} clk {best[1]*1e3:+.1f} a {np.round(aw,2)}", flush=True)
print("bias spread over sequences: mean", np.round(np.mean(list(gtb.values()), 0), 4), "std", np.round(np.std(list(gtb.values()), 0), 4))


def others(s, loeo):
    env = [e for e, v in ENV.items() if s in v][0]
    use = [x for x in SEQS if (x not in ENV[env] if loeo else x != s)]
    return np.mean([gtb[x] for x in use], 0)


def tilt(P, b, **k):
    i2 = P["imu"].copy()
    i2[P["ok"], :3] -= b
    return baseline_attitude(P["t"], i2, P["ok"], **k)


def b_still(P):
    sb = still_bias(P["t"], P["imu"], P["ok"])[0]
    return sb if sb is not None else np.zeros(3)


def b_auto(P):
    sb = still_bias(P["t"], P["imu"], P["ok"])[0]
    return sb if sb is not None else gravity_bias(P["t"], P["imu"], P["ok"])


STRONG = dict(knot_s=60, s_r=0.3, sigma_b=0.003, residual_s=5.0)
M = {
    "gyro only, no bias": lambda P, s: baseline_attitude(P["t"], P["imu"], P["ok"], kp=0, boost=0),
    "tilt filter, no bias": lambda P, s: baseline_attitude(P["t"], P["imu"], P["ok"]),
    "tilt filter, still bias": lambda P, s: tilt(P, b_still(P)),
    "tilt filter, gravity bias": lambda P, s: tilt(P, gravity_bias(P["t"], P["imu"], P["ok"])),
    "tilt filter, still else gravity": lambda P, s: tilt(P, b_auto(P)),
    "tilt filter, calib other seqs": lambda P, s: tilt(P, others(s, False)),
    "tilt filter, calib other envs": lambda P, s: tilt(P, others(s, True)),
    "offline, still else gravity": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=b_auto(P), **STRONG),
    "offline, calib other seqs": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=others(s, False), **STRONG),
    "offline, calib other envs": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=others(s, True), **STRONG),
}
KEYS = ("att_org_pre", "tilt_pre", "rpe_r_pre", "att_org", "rpe_r")
files = eval_set(SEQS)
res = {}
refs = references(files, f"{RES}/refs_euroc.json")
print("\nreferences on EuRoC (mean/worst over sequences, seeds, gap modes)")
for w in ("stationary", "baseline"):
    show(w, summarise([(k, v[w]) for k, v in refs.items()]), ("ate_se3", "ate_org", "rpe_t", "att_org", "blind_rel"))
    show(w, summarise([(k, v[w]) for k, v in refs.items()]), ("rpe_r", "tilt", "ate_se3_pre", "ate_org_pre", "rpe_t_pre"))
print("\nattitude methods")
for name, fn in M.items():
    rows = []
    for (n, s, m, sd) in files:
        room = load_room(n)
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        rows.append((f"{n}|{s}|{m}|{sd}", score_all(P, room, np.zeros((len(P["t"]), 3)), fn(P, n))))
    res[name] = rows
    show(name, summarise(rows), KEYS)
    json.dump(res, open(f"{RES}/h3_euroc.json", "w"))
print("\nsafety on TUM-VI dev rooms for the automatic bias")
rows = []
for (n, s, m, sd) in eval_set(DEV_ROOMS):
    room = load_room(n)
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    rows.append((f"{n}|{s}|{m}|{sd}", score_all(P, room, np.zeros((len(P["t"]), 3)), M["offline, still else gravity"](P, n))))
show("offline, still else gravity, TUM", summarise(rows), KEYS)
