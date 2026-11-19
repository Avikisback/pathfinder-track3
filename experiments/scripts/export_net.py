"""Writes a trained VelNet to the flat binary read by cpp/bench_net.cpp, with one test vector from PyTorch."""
import struct, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *

ck = torch.load(sys.argv[1], map_location="cpu")
cfg = ck["cfg"]
net = VelNet(T=cfg["T"], ch=cfg["ch"], k=cfg["k"], drop=0.0, c_in=cfg["c_in"], out=cfg["out"], fc=cfg["fc"])
net.load_state_dict(ck["state"])
net.eval()
torch.manual_seed(1)
x = torch.randn(1, cfg["c_in"], cfg["T"])
y = net(x).detach().numpy()[0]
blob = []
for name, p in net.state_dict().items():
    a = p.numpy().astype(np.float32)
    # Conv weight (out, in, k) and Linear weight (out, in) are stored in the order the C++ expects
    blob.append(a.ravel())
blob = np.concatenate(blob)
ch = list(cfg["ch"]) + [0] * (8 - len(cfg["ch"]))
hdr = [cfg["T"], cfg["c_in"], len(cfg["ch"]), cfg["k"], cfg["fc"], cfg["out"]] + ch + [len(blob), 0]
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("16i", *hdr))
    f.write(x.numpy().astype(np.float32).tobytes())
    f.write(y.astype(np.float32).tobytes())
    f.write(blob.tobytes())
print("params", len(blob), "macs per forward", macs(cfg), "bytes", len(blob) * 4)
