"""参照実装(ARC mlp_kprop, K=3 SIMPLE)の内部量を層ごとに取り出す(GPU)。

  research/gpu_env.sh research/gpu_ref_internals.py <MLP番号> <層数>
事前活性化 pre{l-1}(= 私の表記の層 l の z)について m, var, D3=κ3_iii, D21=κ3_iik, K4=κ4_iiii を保存。
"""
import sys

import numpy as np
import torch
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.mlp import MLP

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    i, L = int(sys.argv[1]), int(sys.argv[2])
    d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
    w = d["weights"][:L]
    n = w.shape[1]
    dev = torch.device("cuda")
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    mlp = mlp.double().to(dev)
    for j in range(L):
        mlp.Ws[j].weight.data = torch.from_numpy(w[j].astype(np.float64).T).contiguous().to(dev)
    K_in = {1: torch.zeros(n, dtype=torch.float64, device=dev),
            2: torch.eye(n, dtype=torch.float64, device=dev)}
    K = mlp_kprop(mlp, K_in, k_max=3, kind=Kind.SIMPLE, factor=True, output_all=True,
                  up_to_layer=f"act{L - 1}")
    out = {}
    for l in range(1, L + 1):
        t = K[f"pre{l - 1}"]
        out[f"m_{l}"] = t[1].to_tensor().cpu().numpy()
        out[f"var_{l}"] = t[2].get_dslice((2,)).cpu().numpy()
        if 3 in t:
            out[f"D3_{l}"] = t[3].get_dslice((3,)).cpu().numpy()
            out[f"D21_{l}"] = t[3].get_dslice((2, 1)).float().cpu().numpy()
        if 4 in t:
            out[f"K4_{l}"] = t[4].get_dslice((4,)).cpu().numpy()
    np.savez(f"/kaggle/ext/ref/int_{i:03d}.npz", **out)
    print("saved", sorted(out)[:6], "...", flush=True)
