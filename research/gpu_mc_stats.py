"""RTX 5090 で MC を回し、事前活性化 z の層ごとの統計量の「真値」を作る。

  research/gpu_env.sh research/gpu_mc_stats.py <MLP番号> <最大層> <サンプル数(億)>

各層 l(1 始まり)について、z = h_{l-1} @ W_l の
  m     平均                         (n,)
  var   分散                         (n,)
  D3    κ3[z_i,z_i,z_i]              (n,)
  D21   κ3[z_i,z_i,z_k]              (n,n)   ← 私の推定器の D21[i,k] と同じ並び
  K4    κ4[z_i,z_i,z_i,z_i](超過尖度×var²) (n,)
  hmean ReLU 後の平均                (n,)
を 2 パスで推定する(1 パス目で平均、2 パス目で中心化モーメント)。
出力: /kaggle/ext/mc/mc_{番号}.npz。wandb にはオフラインで要約を記録する。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DUMPS = Path("/kaggle/ext/dumps")
OUT = Path("/kaggle/ext/mc")
CHUNK = 500_000


def forward_layers(x, Ws, upto):
    zs = []
    for l in range(upto):
        z = x @ Ws[l]
        zs.append(z)
        x = torch.relu(z)
    return zs


def main():
    torch.set_grad_enabled(False)
    idx, upto, n_e8 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    N = int(n_e8 * 1e8)
    dev = torch.device("cuda")
    d = np.load(DUMPS / f"mlp_{idx:03d}.npz")
    Ws = [torch.from_numpy(w).to(dev) for w in d["weights"][:upto]]
    n = Ws[0].shape[0]
    gen = torch.Generator(device=dev).manual_seed(1234 + idx)
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    run = wandb.init(project="arc-whest-phase2", name=f"gpu_mc_stats_{idx}",
                     config=dict(idx=idx, upto=upto, N=N), dir="/kaggle/ext/wandb")
    t0 = time.time()
    # 1 パス目: 平均(z と h)
    s1 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    sh = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    n1 = N // 4
    done = 0
    while done < n1:
        x = torch.randn(CHUNK, n, device=dev, generator=gen)
        for l, z in enumerate(forward_layers(x, Ws, upto)):
            s1[l] += z.sum(0, dtype=torch.float64)
            sh[l] += torch.relu(z).sum(0, dtype=torch.float64)
        done += CHUNK
    mean = [s / done for s in s1]
    # 2 パス目: 中心化モーメント(別の乱数列)
    m2 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    m3 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    m4 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    m21 = [torch.zeros(n, n, dtype=torch.float64, device=dev) for _ in range(upto)]
    hsum = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(upto)]
    done = 0
    while done < N:
        x = torch.randn(CHUNK, n, device=dev, generator=gen)
        for l, z in enumerate(forward_layers(x, Ws, upto)):
            hsum[l] += torch.relu(z).sum(0, dtype=torch.float64)
            c = z - mean[l].float()
            c2 = c * c
            m2[l] += c2.sum(0, dtype=torch.float64)
            m3[l] += (c2 * c).sum(0, dtype=torch.float64)
            m4[l] += (c2 * c2).sum(0, dtype=torch.float64)
            m21[l] += (c2.T @ c).double()  # Σ c_i² c_k
        done += CHUNK
    out = {}
    for l in range(upto):
        var = m2[l] / done
        k3 = m3[l] / done
        k4 = m4[l] / done - 3 * var * var
        d21 = m21[l] / done
        hm_all = (hsum[l] + sh[l]) / (done + n1)
        L = l + 1
        out.update({f"m_{L}": mean[l].cpu().numpy(), f"var_{L}": var.cpu().numpy(),
                    f"D3_{L}": k3.cpu().numpy(), f"D21_{L}": d21.float().cpu().numpy(),
                    f"K4_{L}": k4.cpu().numpy(), f"hmean_{L}": hm_all.cpu().numpy()})
        off = d21 - torch.diag(torch.diag(d21))
        wandb.log({"layer": L, "D3_rms": k3.pow(2).mean().sqrt().item(),
                   "D21_offdiag_rms": off.pow(2).mean().sqrt().item(),
                   "omega_mean": (k4 / var**2).mean().item()})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"mc_{idx:03d}.npz", N=done, **out)
    print(f"MLP {idx}: N={done:.2e} 層1〜{upto} を {time.time() - t0:.0f}s で保存", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
