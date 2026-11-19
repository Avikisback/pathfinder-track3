"""The official degradation with other random seeds (the script fixes SEED = 42), designed column order, so that choices
are not fitted to one noise draw. Writes tmp/cpp_off_s<seed>. Usage: make_official_seeds.py <seed> [<seed> ...]"""
import sys, os
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/tools")
import pandas as pd
import common
import degrade_imu as D
SEQ = [s for v in common.ENV.values() for s in v]
files = [(s, common.euroc_dir(s) + "/imu0/data.csv") for s in SEQ] + [(f"room{n}", f"{common.ROOT}/data/tumvi_room{n}/imu0/data.csv") for n in common.DEV_ROOMS]
for seed in map(int, sys.argv[1:]):
    out_dir = common.EXP + f"/tmp/cpp_off_s{seed}"
    os.makedirs(out_dir, exist_ok=True)
    for name, path in files:
        df = pd.read_csv(path, header=0)
        cols = list(df.columns)
        out, st = D.degrade(df[[cols[0]] + cols[4:7] + cols[1:4]], dict(D.CONFIG), seed)
        out[cols].to_csv(f"{out_dir}/{name}_sw.csv", index=False, header=True)
    print("seed", seed, "done", flush=True)
