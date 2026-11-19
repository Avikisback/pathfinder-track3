"""Table for 50_seqnet results against references and the e2e methods, on the files both have."""
import glob, json, pickle, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *

refs = {}
for f in ("refs_euroc.json", "refs_dev.json"):
    refs.update(json.load(open(f"{RES}/{f}")))
tag, scheme = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(f"{RES}/{tag}_{scheme}_f*.pkl")):
    rows.update(pickle.load(open(f, "rb"))["rows"])
e = {}
for f in sorted(glob.glob(f"{RES}/e2e_{scheme}_f*.pkl")):
    e.update(pickle.load(open(f, "rb"))["rows"])
print(f"{tag} {scheme}: {len(rows)} files, e2e {len(e)}")
ks = [k for k in rows]
for sc in ("clean", "mild", "full"):
    sk = [k for k in ks if k.split("|")[1] == sc]
    if not sk:
        continue
    print(f"-- {sc} ({len(sk)} files)")
    def line(nm, get):
        vals = {m: [get(k).get(m, np.nan) for k in sk] for m in ("ate_se3", "ate_org", "rpe_t", "blind_rel")}
        print(f"  {nm:12s} " + " ".join(f"{m} {np.nanmean(v):.3f}/{np.nanmax(v):.3f}" for m, v in vals.items()))
    line("stationary", lambda k: refs[k]["stationary"])
    if all(k in e for k in sk):
        for m in ("physics", "band", "net"):
            line(m, lambda k: e[k][m])
    for t in rows[sk[0]]:
        line(t, lambda k: rows[k][t])
seqs = sorted({k.split("|")[0] for k in ks}, key=str)
print("full ATE by sequence, stationary / best fixed taus:")
for s in seqs:
    sk = [k for k in ks if k.split("|")[0] == s and k.split("|")[1] == "full"]
    if sk:
        print(f"  {s}: stat {np.mean([refs[k]['stationary']['ate_se3'] for k in sk]):.2f} " + " ".join(f"{t} {np.mean([rows[k][t]['ate_se3'] for k in sk]):.2f}" for t in rows[sk[0]]))
