"""Test files made with the organisers' own script (tools/degrade_imu.py, received 17 Nov), for every EuRoC flight and
TUM-VI room 1-5. The script expects columns timestamp, accel, gyro; EuRoC files are timestamp, gyro, accel. Which way the
organisers ran it is not known, so both are written:
  <name>_sw.csv  columns swapped to accel-first for the script, then swapped back (noise lands where it was designed to)
  <name>_og.csv  the script run on the EuRoC file as it is (the gyro gets the accelerometer's noise and vice versa)
Output keeps the EuRoC column order and header either way. Usage: make_official.py"""
import sys, os, json
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/tools")
import pandas as pd
import common
import degrade_imu as D

OUT = common.EXP + "/tmp/cpp_off"
os.makedirs(OUT, exist_ok=True)
SEQ = [s for v in common.ENV.values() for s in v]
files = [(s, common.euroc_dir(s) + "/imu0/data.csv") for s in SEQ] + [(f"room{n}", f"{common.ROOT}/data/tumvi_room{n}/imu0/data.csv") for n in common.DEV_ROOMS]
for name, path in files:
    df = pd.read_csv(path, header=0)
    cols = list(df.columns)
    for tag in ("sw", "og"):
        d = df[[cols[0]] + cols[4:7] + cols[1:4]] if tag == "sw" else df
        out, st = D.degrade(d, dict(D.CONFIG), D.SEED)
        out = out[cols]
        out.to_csv(f"{OUT}/{name}_{tag}.csv", index=False, header=True)
        json.dump(st, open(f"{OUT}/{name}_{tag}.json", "w"), indent=1)
        print(f"{name}_{tag}: rows {st['rows_in']} -> {st['rows_out']} | dropped {st['dropped_total']} ({st['dropped_burst_gaps']} bursts) dup {st['duplicated_rows']} "
              f"stuck {st['stuck_samples']} spikes {st['spikes']} | units accel {st['accel_unit']} | final bias accel {st['final_accel_bias_m_s2']} gyro {st['final_gyro_bias_rad_s']}", flush=True)
