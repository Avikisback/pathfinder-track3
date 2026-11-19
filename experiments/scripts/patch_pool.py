"""One-off edit: training reads realisations from the disk cache, and a TUM leave-one-room-out scheme with auto attitude."""
s = open("pipeline.py").read()
if "pool_cached" not in s:
    s = s.replace('''            items = [pool_item(rooms[n], labs[n], rng, cfg.get("attitude")) for n in train_rooms for _ in range(cfg.get("pool_per_room", 4))]
''', '''            if cfg.get("cached"):
                items = pool_cached(train_rooms, rng, cfg.get("pool_per_room", 2))
            else:
                items = [pool_item(rooms[n], labs[n], rng, cfg.get("attitude")) for n in train_rooms for _ in range(cfg.get("pool_per_room", 4))]
''')
    s = s.replace('''class Pool:
    """Concatenates''', '''_POOLS = {}


def pool_cached(seqs, rng, per):
    items = []
    for n in seqs:
        if n not in _POOLS:
            d = np.load(f"{CACHE}/pool_{n}.npz")
            _POOLS[n] = {k: d[k] for k in d.files}
        d = _POOLS[n]
        for i in rng.integers(0, len(d["X"]), per):
            items.append(dict(X=d["X"][i], yawh=d["yawh"][i], v=d["v"], yaw=d["yaw"], ok=d["ok"]))
    return items


class Pool:
    """Concatenates''')
    s = s.replace('''    rooms = {n: load_room(n) for n in train_rooms}
    labs = {n: gt_labels(rooms[n]) for n in train_rooms}
    T = cfg["T"]''', '''    if not cfg.get("cached"):
        rooms = {n: load_room(n) for n in train_rooms}
        labs = {n: gt_labels(rooms[n]) for n in train_rooms}
    T = cfg["T"]''')
    open("pipeline.py", "w").write(s)
s = open("20_cross.py").read()
if "cached=True" not in s:
    s = s.replace('no_up=a.no_up, pool_per_room=a.per_room, attitude="auto")', 'no_up=a.no_up, pool_per_room=a.per_room, attitude="auto", cached=True)')
    s = s.replace('''else:
    raise SystemExit("unknown scheme")''', '''elif a.scheme == "loro":
    folds = [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
else:
    raise SystemExit("unknown scheme")''')
    open("20_cross.py", "w").write(s)
s = open("run_cross.sh").read()
if "loro" not in s:
    s = s.replace("--name eu2s --scheme loso --T 100 --center 1\n", "--name eu2s --scheme loso --T 100 --center 1\n--name tum --scheme loro --T 100 --center 1\n--name tumc --scheme loro --T 100 --center 0\n", 1)
    open("run_cross.sh", "w").write(s)
print("patched")
