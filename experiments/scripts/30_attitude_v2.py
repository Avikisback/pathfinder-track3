"""Coordinator item 2: hardened rest search under injected drift, both datasets. Compares with the file-wide threshold
test of main.cpp, the fixed gravity fit, and a sensor calibration fitted on other sequences (LOSO, declared)."""
import json, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *

SEQS = list(EUROC)
gtb = {s: np.median(load_room(s)["gt_bw"], 0) for s in SEQS}
print("rest detection on clean and degraded EuRoC and TUM-VI (blocks found, file-level bias error rad/s against ground truth bias)")
for s in SEQS + DEV_ROOMS:
    row = []
    for sch in ("clean", "full"):
        t_ns, imu, _ = eval_file(s, sch, "zero", 1)
        P = prepare(t_ns, imu)
        tb, mb = hardened_rest(P["t"], P["imu"], P["ok"])
        sb = still_bias(P["t"], P["imu"], P["ok"])[0]
        ref = gtb.get(s, np.zeros(3))
        gb = gravity_bias(P["t"], P["imu"], P["ok"])
        row.append(f"{sch}: hard {0 if tb is None else len(tb)} blk err {np.round(np.median(mb,0)-ref,4) if tb is not None else None} first {None if tb is None else round(tb[0],1)} | main.cpp err {np.round(sb-ref,4) if sb is not None else None} | grav err {np.round(gb-ref,4)}")
    print(s, " || ".join(row), flush=True)


def others(s, loeo):
    if not isinstance(s, str):
        return None
    env = [e for e, v in ENV.items() if s in v][0]
    use = [x for x in SEQS if (x not in ENV[env] if loeo else x != s)]
    return np.mean([gtb[x] for x in use], 0)


def still_tilt(P, s):
    sb = still_bias(P["t"], P["imu"], P["ok"])[0]
    i2 = P["imu"].copy()
    i2[P["ok"], :3] -= sb if sb is not None else 0
    return baseline_attitude(P["t"], i2, P["ok"])


M = {
    "main.cpp-style: still test, tilt filter": still_tilt,
    "gravity fit (truncated), offline": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], mode="gravity")[0],
    "hardened rest, else zero, offline": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"])[0],
    "hardened rest with ramp, else zero": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], mode="rest_ramp")[0],
    "hardened rest, else calib other envs": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], calib=others(s, True))[0],
    "calib other envs only, offline": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=others(s, True) if isinstance(s, str) else np.zeros(3), **STRONG),
    "true gt bias, offline (oracle bound)": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=gtb[s] if isinstance(s, str) else np.zeros(3), **STRONG),
}
KEYS = ("att_org_pre", "tilt_pre", "rpe_r_pre", "att_org", "rpe_r")
res = {}
for label, seqs in (("EuRoC", SEQS), ("TUM dev", DEV_ROOMS)):
    print(f"\n{label} (blind tail excluded in _pre columns)")
    for name, fn in M.items():
        rows = []
        for (n, s, m, sd) in eval_set(seqs, seeds=(1, 2)):
            room = load_room(n)
            t, imu, _ = eval_file(n, s, m, sd)
            P = prepare(t, imu)
            sc_q = fn(P, n)
            et, _, eq = finish(P, np.zeros((len(P["t"]), 3)), sc_q)
            pre = et <= P["t_fail_ns"]
            d = score(et[pre] * 1e-9, np.zeros((pre.sum(), 3)), eq[pre], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
            d2 = score(et * 1e-9, np.zeros((len(et), 3)), eq, room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
            rows.append((f"{n}|{s}|{m}|{sd}", dict(att_org_pre=d["att_org"], tilt_pre=d["tilt"], rpe_r_pre=d["rpe_r"], att_org=d2["att_org"], rpe_r=d2["rpe_r"])))
        res[f"{label}|{name}"] = rows
        show(name, summarise(rows), KEYS)
json.dump(res, open(f"{RES}/h3_v2.json", "w"))
print("ATT_V2_DONE")
