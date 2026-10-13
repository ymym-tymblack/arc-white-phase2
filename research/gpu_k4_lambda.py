"""幅 1024 の mini MLP で、κ̃4[z_i,z_j,z_k,z_k](i,j,k 相異)を GPU MC で測り、ρ_ij への回帰係数 λ_ℓ を出す。

  research/gpu_env.sh research/gpu_k4_lambda.py <MLP番号> <層数> <サンプル数(億)> <k の本数>

k はランダムに選んだ一部(既定 64 本)だけを測る(全部だと n³ 個になるため)。
κ4[i,j,k,k] = E[c_i c_j c_k²] - E[c_i c_j] E[c_k²] - 2 E[c_i c_k] E[c_j c_k](c は中心化した z)。
"""
import os
import sys
import time

import numpy as np
import torch

CHUNK = 250_000

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    idx, L, n_e8, nk = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    N = int(n_e8 * 1e8)
    dev = torch.device("cuda")
    d = np.load(f"/kaggle/ext/dumps/mlp_{idx:03d}.npz")
    Ws = [torch.from_numpy(w).to(dev) for w in d["weights"][:L]]
    n = Ws[0].shape[0]
    ks = torch.from_numpy(np.random.default_rng(0).choice(n, nk, replace=False)).to(dev)
    gen = torch.Generator(device=dev).manual_seed(77 + idx)
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    run = wandb.init(project="arc-whest-phase2", name=f"gpu_k4_lambda_{idx}",
                     config=dict(idx=idx, L=L, N=N, nk=nk), dir="/kaggle/ext/wandb")
    t0 = time.time()
    # 1 パス目: 平均
    mean = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(L)]
    done = 0
    while done < N // 4:
        x = torch.randn(CHUNK, n, device=dev, generator=gen)
        for l in range(L):
            z = x @ Ws[l]
            mean[l] += z.sum(0, dtype=torch.float64)
            x = torch.relu(z)
        done += CHUNK
    mean = [m / done for m in mean]
    # 2 パス目: E[c c^T], E[c_i c_j c_k^2](k は部分集合)
    C2 = [torch.zeros(n, n, dtype=torch.float64, device=dev) for _ in range(L)]
    E4 = [torch.zeros(nk, n, n, dtype=torch.float64, device=dev) for _ in range(L)]
    done = 0
    while done < N:
        x = torch.randn(CHUNK, n, device=dev, generator=gen)
        for l in range(L):
            z = x @ Ws[l]
            x = torch.relu(z)
            c = z - mean[l].float()
            C2[l] += (c.T @ c).double()
            ck2 = c[:, ks] ** 2  # (B, nk)
            # Σ_b c_bi c_bj ck2_bk → (nk, n, n)
            for kk in range(nk):
                E4[l][kk] += ((c * ck2[:, kk : kk + 1]).T @ c).double()
        done += CHUNK
    print(f"MC {time.time() - t0:.0f}s", flush=True)
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    for l in range(1, L):
        C = C2[l] / done
        v = torch.diag(C)
        s = v.sqrt()
        rho = C / torch.outer(s, s)
        E = E4[l] / done
        k4 = E - C[None] * v[ks][:, None, None] - 2.0 * C[ks][:, :, None] * C[ks][:, None, :]
        K = k4 / (torch.outer(s, s)[None] * v[ks][:, None, None])
        mask = off[None].expand(nk, n, n).clone()
        kk = torch.arange(nk, device=dev)
        mask[kk, ks, :] = False
        mask[kk, :, ks] = False
        y = K[mask]
        x1 = rho[None].expand(nk, n, n)[mask]
        x2 = (rho[ks][:, :, None] * rho[ks][:, None, :])[mask]
        lam = (x1 @ y / (x1 @ x1)).item()
        r2 = 1 - ((y - lam * x1) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()
        X = torch.stack([x1, x2], 1)
        coef = torch.linalg.solve(X.T @ X, X.T @ y)  # 正規方程式(行数が多く lstsq は失敗する)
        r2b = 1 - ((y - X @ coef) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()
        omega = 0.0
        print(f"層{l + 1}: λ={lam:.5f} R²={r2:.3f} | 2変数 ρ_ij {coef[0].item():.5f} ρ_ikρ_jk "
              f"{coef[1].item():.5f} R²={r2b:.3f} | K rms {y.pow(2).mean().sqrt().item():.2e}", flush=True)
        wandb.log({"layer": l + 1, "lambda": lam, "r2": r2})
    run.finish()
