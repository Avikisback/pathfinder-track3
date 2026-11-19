"""Builds a disk cache of randomised degraded training realisations, one file per sequence. Training folds then only
sample from it, so 11+ folds do not each redo attitude estimation. Rooms 1-5 and all EuRoC. Room 6 stays locked."""
import multiprocessing as mp
import sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

R_PER_SEQ = 12


def work(n):
    f = f"{CACHE}/pool_{n}.npz"
    if os.path.exists(f):
        return n
    room = load_room(n)
    lab = gt_labels(room)
    rng = np.random.default_rng(abs(hash(str(n))) % (2 ** 31) if False else 1000 + sum(ord(c) for c in str(n)))
    items = [pool_item(room, lab, rng, "auto") for _ in range(R_PER_SEQ)]
    np.savez(f, X=np.stack([i["X"] for i in items]), yawh=np.stack([i["yawh"] for i in items]), v=items[0]["v"],
             yaw=items[0]["yaw"], ok=items[0]["ok"])
    return n


if __name__ == "__main__":
    seqs = list(DEV_ROOMS) + list(EUROC)
    with mp.get_context("spawn").Pool(16) as p:
        for n in p.imap_unordered(work, seqs):
            print("pool", n, flush=True)
    print("POOL_DONE")
