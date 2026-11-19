#!/usr/bin/env python3
"""Download the IMU and ground truth of the Samsung Odyssey+ recordings in the Monado SLAM dataset (approved by the user
on 16 Nov). Dataset: https://huggingface.co/datasets/collabora/monado-slam-datasets (CC BY 4.0, no sign-up).

Each recording is a zip of 0.2-2 GB that is mostly camera images. Only three members are needed (imu0/data.csv,
imu0/data.raw.csv and gt/data.csv), so the zip is opened over HTTP with range requests and just those members are read:
about 10-25 MB per recording instead of gigabytes.

Usage: get_monado.py <out_dir> [name-filter ...]
"""
import io, json, os, sys, urllib.request, zipfile

API = "https://huggingface.co/api/datasets/collabora/monado-slam-datasets/tree/main/M_monado_datasets/MO_odyssey_plus/MOO_others"
RES = "https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/"
WANT = ("imu0/data.csv", "imu0/data.raw.csv", "gt/data.csv")
DEFAULT = ("MOO01", "MOO02", "MOO03", "MOO04", "MOO05", "MOO06", "MOO07", "MOO08", "MOO13", "MOO14")
UA = {"User-Agent": "pathfinder-research-script"}


class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched in 2 MB blocks on demand."""
    BLOCK = 2 << 20

    def __init__(self, url):
        self.url = url
        r = urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0")), timeout=120)
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.url = r.geturl()                      # keep the redirect target for later range requests
        self.pos = 0
        self.cache = {}
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a, b = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            for attempt in range(4):
                try:
                    d = urllib.request.urlopen(urllib.request.Request(self.url, headers=dict(UA, Range=f"bytes={a}-{b}")), timeout=300).read()
                    break
                except Exception:
                    if attempt == 3:
                        raise
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = d
            self.fetched += len(d)
        return self.cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def main():
    out_dir = sys.argv[1]
    filt = sys.argv[2:] or DEFAULT
    tree = json.load(urllib.request.urlopen(urllib.request.Request(API, headers=UA), timeout=120))
    zips = sorted(e["path"] for e in tree if e["path"].endswith(".zip") and any(f in os.path.basename(e["path"]) for f in filt))
    print(len(zips), "recordings", flush=True)
    total = 0
    for path in zips:
        name = os.path.basename(path)[:-4]
        dst = os.path.join(out_dir, name)
        if all(os.path.exists(os.path.join(dst, "mav0", w)) for w in ("imu0/data.csv", "gt/data.csv")):
            print(name, "already there", flush=True)
            continue
        try:
            f = HttpFile(RES + path)
            z = zipfile.ZipFile(f)
            got = []
            for m in z.namelist():
                for w in WANT:
                    if m.endswith("mav0/" + w):
                        p = os.path.join(dst, "mav0", w)
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        with z.open(m) as src, open(p, "wb") as o:
                            while True:
                                b = src.read(1 << 20)
                                if not b:
                                    break
                                o.write(b)
                        got.append(w)
            total += f.fetched
            print(f"{name}: {', '.join(got)} | fetched {f.fetched / 1e6:.1f} MB of {f.size / 1e9:.2f} GB", flush=True)
        except Exception as ex:
            print(name, "FAILED", repr(ex)[:200], flush=True)
    print(f"total fetched {total / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
