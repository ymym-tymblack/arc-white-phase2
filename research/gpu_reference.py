"""ARC の公式参照実装(mlp_kprop、論文の著者による実装)を RTX 5090 で動かし、mini の MLP で
層ごとの平均を出す。私の因子化版と、どの層からずれるかを比べる基準にする。

実行(GPU 研究環境。提出用の環境とは別):
  uv run --no-project --python 3.12 --with torch --with einops --with jaxtyping --with joblib \
    --with matplotlib --with opt_einsum --with pandas --with tqdm --with scipy --with numpy \
    --with wandb --with-editable /kaggle/ext/mlp_cumulant_propagation \
    python research/gpu_reference.py <k_max> <MLP番号...>

入力は research/export_mini.py が書き出した /kaggle/ext/dumps/mlp_XXX.npz。
出力は /kaggle/ext/ref/kprop_k{k_max}_{番号}.npz(層ごとの平均)。wandb はオフラインで記録する。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.mlp import MLP

DUMPS = Path("/kaggle/ext/dumps")
OUT = Path("/kaggle/ext/ref")


def reference_means(weights: np.ndarray, k_max: int, device) -> np.ndarray:
    """競技の MLP(全 L 層に ReLU)を参照実装の規約(最後の線形層の後は活性化なし)に合わせて
    L+1 層として作り、act0..act{L-1} の平均(1 次キュムラント)を返す。"""
    L, n = weights.shape[0], weights.shape[1]
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    mlp = mlp.double().to(device)
    for i in range(L):
        # 競技は x @ W(W は 入力×出力)、参照実装は nn.Linear(weight は 出力×入力)
        mlp.Ws[i].weight.data = torch.from_numpy(weights[i].astype(np.float64).T).contiguous().to(device)
    K_in = {1: torch.zeros(n, dtype=torch.float64, device=device),
            2: torch.eye(n, dtype=torch.float64, device=device)}
    K = mlp_kprop(mlp, K_in, k_max=k_max, kind=Kind.SIMPLE, factor=(k_max >= 3),
                  output_all=True, up_to_layer=f"act{L - 1}")
    return torch.stack([K[f"act{i}"][1].to_tensor() for i in range(L)]).cpu().numpy()


def main():
    torch.set_grad_enabled(False)
    k_max = int(sys.argv[1])
    idxs = [int(x) for x in sys.argv[2:]]
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    run = wandb.init(project="arc-whest-phase2", name=f"kprop_reference_k{k_max}",
                     config=dict(k_max=k_max, idxs=idxs, kind="SIMPLE", device="RTX 5090"),
                     dir="/kaggle/ext/wandb")
    for i in idxs:
        d = np.load(DUMPS / f"mlp_{i:03d}.npz")
        t = time.time()
        pred = reference_means(d["weights"], k_max, device)
        dt = time.time() - t
        truth = d["all_layer_means"]
        per_layer = ((pred - truth) ** 2).mean(1)
        np.savez(OUT / f"kprop_k{k_max}_{i:03d}.npz", pred=pred)
        print(f"MLP {i}: 最終層 MSE {per_layer[-1]:.3e} 全層 {per_layer.mean():.3e} ({dt:.1f}s)",
              flush=True)
        wandb.log({"mlp": i, "final_mse": per_layer[-1], "all_mse": per_layer.mean(), "sec": dt,
                   **{f"layer_mse/{l + 1}": v for l, v in enumerate(per_layer)}})
    run.finish()


if __name__ == "__main__":
    main()
