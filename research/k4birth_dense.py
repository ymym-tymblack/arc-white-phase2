"""B の事前検証(幅 128、密版): κ̃4[z_i,z_j,z_k,z_k] ≈ λ_ℓ ρ_ij(i≠j)と置いた 3 次の誕生項を足すと
誤差が下がるか。λ_ℓ = β ω_ℓ(ω_ℓ = G[ℓ]/n)として β を振る。κ4 の放射部分(ω, ω/3)も入れる。

  uv run python research/k4birth_dense.py <depth>
"""
import sys

import numpy as np

sys.path.insert(0, "research")
import k3_dense as k3  # noqa: E402
from ablate_k3 import mc_means  # noqa: E402

G = [0.05, 9.9, 16.94, 21.82, 26.71, 31.42, 35.48, 39.19, 42.84, 45.89, 49.41, 52.42, 55.03, 58.67, 60.49, 62.31]

if __name__ == "__main__":
    n, depth = 128, int(sys.argv[1])
    k3.GROUPS = {"rho", "tau1", "tau1e", "txxy", "txxx", "pairtau", "meantau", "iik", "iii"}
    for seed in (0, 1, 2):
        r = np.random.default_rng(100 + seed)
        W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
        ref = mc_means(W, 8_000_000)
        Wd = [w.astype(np.float64) for w in W]
        om = [G[l] / n for l in range(depth)]
        kap4 = [(np.full(n, o), np.full((n, n), o / 3.0), np.zeros((n, n))) for o in om]
        # ρ は層ごとに密版の中で決まるので、1 回目の伝播で記録して 2 回目に使う
        tr = []
        k3.propagate(Wd, kmax=6, emax=2, kap4=kap4, trace=tr)
        rhos = []
        for (m, S, T, mu, C, U) in tr:
            s = np.sqrt(np.diag(S))
            rho = S / np.outer(s, s)
            np.fill_diagonal(rho, 0.0)
            rhos.append(rho)
        line = []
        for beta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
            kb = [np.broadcast_to((beta * om[l] * rhos[l])[:, :, None], (n, n, n)).copy() for l in range(depth)]
            mu = k3.propagate(Wd, kmax=6, emax=2, kap4=kap4, k4birth=kb if beta else None)
            e = ((mu - ref) ** 2).mean(1)
            line.append(f"β={beta}: 最終 {e[-1]:.2e} 全層 {e.mean():.2e}")
        print(f"seed {seed}: " + " | ".join(line), flush=True)
