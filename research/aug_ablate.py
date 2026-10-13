"""参照実装 AUGMENT の項を 1 つずつ外し、層 2〜L の D3・D21 誤差(MC 真値に対して)と層の平均の MSE を測る。

  AUG_DROP=k3leg research/gpu_env.sh research/aug_ablate.py <MLP番号> <層数>
"""
import os
import sys

import numpy as np
import torch
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.mlp import MLP

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    i, L = int(sys.argv[1]), int(sys.argv[2])
    d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
    mc = np.load(f"/kaggle/ext/mc/mc_{i:03d}.npz")
    w, t = d["weights"][:L], d["all_layer_means"]
    n = w.shape[1]
    dev = torch.device("cuda")
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).double().to(dev)
    for j in range(L):
        mlp.Ws[j].weight.data = torch.from_numpy(w[j].astype(np.float64).T).contiguous().to(dev)
    K_in = {1: torch.zeros(n, dtype=torch.float64, device=dev), 2: torch.eye(n, dtype=torch.float64, device=dev)}
    K = mlp_kprop(mlp, K_in, k_max=3, kind=Kind.AUGMENT, factor=True, output_all=True, up_to_layer=f"act{L - 1}")
    off = ~np.eye(n, dtype=bool)
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    out = []
    for l in range(2, L + 1):
        pre = K[f"pre{l - 1}"]
        d3 = pre[3].get_dslice((3,)).cpu().numpy()
        d21 = pre[3].get_dslice((2, 1)).cpu().numpy()
        h = K[f"act{l - 1}"][1].to_tensor().cpu().numpy()
        out.append(f"層{l} D3 {rel(d3, mc[f'D3_{l}']):.2e} D21 {rel(d21[off], mc[f'D21_{l}'].astype(float)[off]):.2e} "
                   f"hMSE {((h - t[l - 1]) ** 2).mean():.2e}")
    print(f"AUG_DROP={os.environ.get('AUG_DROP', '')}: " + " | ".join(out), flush=True)
