"""Tables from stored 32_e2e results, with references. Mean/worst over sequences, seeds and gap modes."""
import glob, json, pickle, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *

refs = {}
for f in ("refs_euroc.json", "refs_dev.json"):
    refs.update(json.load(open(f"{RES}/{f}")))
KEYS = ("ate_se3", "ate_org", "rpe_t", "rpe_r", "att_org", "blind_rel")
for tag in sys.argv[1:]:
    rows, meta = {}, []
    for f in sorted(glob.glob(f"{RES}/e2e_{tag}_f*.pkl")):
        d = pickle.load(open(f, "rb"))
        rows.update(d["rows"])
        meta.append((d["hp"], d["tau"], round(d["a"], 2), round(d["b"], 2), {k: round(v["scale"], 2) for k, v in d["cal"].items()}))
    if not rows:
        continue
    print(f"\n=== {tag}: {len(rows)} files")
    for grp, sel in (("EuRoC", lambda k: not k.split("|")[0].isdigit()), ("TUM dev", lambda k: k.split("|")[0].isdigit())):
        ks = [k for k in rows if sel(k)]
        if not ks:
            continue
        print(f"-- {grp}, {len(ks)} files")
        show("stationary", summarise([(k, refs[k]["stationary"]) for k in ks]), KEYS)
        show("baseline binary", summarise([(k, refs[k]["baseline"]) for k in ks]), KEYS)
        for m in ("physics", "band", "net"):
            show(m, summarise([(k, rows[k][m]) for k in ks]), KEYS)
            for sc in ("full", "mild", "clean"):
                sk = [k for k in ks if k.split("|")[1] == sc]
                w = {x: round(float(np.mean([rows[k][m][x] < refs[k]["stationary"][x] for k in sk])), 2) for x in ("ate_se3", "ate_org", "rpe_t")}
                print(f"     {m} {sc} beats stationary on share of files {w}")
        for sc in ("full", "clean"):
            seqs = sorted({k.split("|")[0] for k in ks}, key=str)
            print(f"   {sc} by sequence ATE SE3 stationary/physics/band/net:", " | ".join(
                f"{s} " + "/".join(f"{np.mean([(refs[k]['stationary'] if m == 'stationary' else rows[k][m])['ate_se3'] for k in ks if k.split('|')[0] == s and k.split('|')[1] == sc]):.2f}" for m in ("stationary", "physics", "band", "net")) for s in seqs))
    print("   fold settings:", meta)
