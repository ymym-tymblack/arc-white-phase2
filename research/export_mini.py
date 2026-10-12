"""mini の MLP(重みと N=1e9 の正解)を npz に書き出す。GPU 研究環境(別の Python)から読むため。

  uv run python research/export_mini.py 0 12   # 0〜11 番
"""
import sys
from pathlib import Path

import numpy as np
from whestbench import load_dataset

OUT = Path("/kaggle/ext/dumps")

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    ds = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2",
                      split="mini").with_format("numpy")
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(a, b):
        row = ds[i]
        np.savez(OUT / f"mlp_{i:03d}.npz", weights=np.asarray(row["weights"], dtype=np.float32),
                 all_layer_means=np.asarray(row["all_layer_means"], dtype=np.float64))
        print("saved", i, flush=True)
