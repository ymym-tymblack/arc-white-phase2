"""PyTorch 移植と flopscope 推定器の全層予測を開発集合で照合する(GPU 環境で実行)。

前提: prep_data.py dev64 と ref_dump.py の出力。
例: research/gpu_env.sh research/torch_port/verify_port.py --ref /kaggle/ext/torch_port_ref/combined
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/kaggle/ext/torch_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--indices", default="0-3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--source", default=str(ROOT / "research/frozen_phase2/combined.py"))
    ap.add_argument("--no-sym", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.indices.split("-"))
    idxs = list(range(lo, hi + 1))
    ids = json.loads((DATA / "dev64/ids.json").read_text())
    truth = np.load(DATA / "dev64/truth.npy")
    dev = torch.device("cuda")
    P = chain.ChainParams(a.source).to(dev)
    cfg = chain.ChainConfig
    cfg.symmetrize_gram = not a.no_sym
    out = []
    for s in range(0, len(idxs), a.batch):
        chunk = idxs[s:s + a.batch]
        W = torch.from_numpy(np.stack([np.load(DATA / f"dev64/w_{ids[i]:04d}.npy") for i in chunk])).to(dev)
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            pred = chain.predict(P, W, cfg).double().cpu().numpy()
        torch.cuda.synchronize()
        dt = time.time() - t
        for j, i in enumerate(chunk):
            ref = np.load(a.ref / f"dev_{i:02d}.npy")
            p = pred[j]
            mse_t = float(((p[-1] - truth[i, -1]) ** 2).mean())
            mse_r = float(((ref[-1] - truth[i, -1]) ** 2).mean())
            rec = dict(idx=i, mse_torch=mse_t, mse_ref=mse_r, rel=mse_t / mse_r - 1.0,
                       maxdiff_all=float(np.abs(p - ref).max()),
                       maxdiff_last=float(np.abs(p[-1] - ref[-1]).max()),
                       maxdiff_layer=[float(x) for x in np.abs(p - ref).max(1)], sec=dt)
            out.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "maxdiff_layer"}), flush=True)
            print("  per-layer maxdiff", " ".join(f"{x:.1e}" for x in rec["maxdiff_layer"]), flush=True)
    print("peak_mem_GiB", torch.cuda.max_memory_allocated() / 2**30)
    ok = all(abs(r["rel"]) <= 0.005 and r["maxdiff_all"] <= 2e-5 for r in out)
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
