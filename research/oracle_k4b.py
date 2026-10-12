"""κ4 の対角(超過尖度 ω)だけを入れる効果: 真の ω_i / 層平均 / 表 g(l)/n。"""
import sys
import numpy as np
sys.path.insert(0, "research")
import k3_dense as k3
from ablate_k3 import mc_means
from mc_cumulants import mc_z_kappa4

G = [0.05, 9.9, 16.94, 21.82, 26.71, 31.42, 35.48, 39.19, 42.84, 45.89, 49.41, 52.42, 55.03, 58.67, 60.49, 62.31]
n, depth = int(sys.argv[1]), int(sys.argv[2])
FULL = {"rho","tau1","tau1e","txxy","txxx","pairtau","meantau","iik","iii"}
for seed in (0, 1):
    r = np.random.default_rng(100 + seed)
    W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
    ref = mc_means(W, 4_000_000)
    k4 = mc_z_kappa4(W, 2_000_000)
    f = lambda mu: ((mu - ref) ** 2).mean(1)
    k3.GROUPS = set(FULL)
    Wd = [w.astype(np.float64) for w in W]
    z = np.zeros((n, n))
    res = {"K3 (κ4なし)": f(k3.propagate(Wd, kmax=6, emax=2))}
    res["+ω 真の ω_i"] = f(k3.propagate(Wd, kmax=6, emax=2, kap4=[(a, z, z) for a, _, _ in k4]))
    res["+ω 層平均(真)"] = f(k3.propagate(Wd, kmax=6, emax=2, kap4=[(np.full(n, a.mean()), z, z) for a, _, _ in k4]))
    res["+ω 表 g(l)/n"] = f(k3.propagate(Wd, kmax=6, emax=2, kap4=[(np.full(n, G[l] / n), z, z) for l in range(depth)]))
    print(f"seed {seed} n={n} depth={depth} 最終層 / 全層平均")
    for k, v in res.items():
        print(f"  {k:16s} {v[-1]:.2e}  {v.mean():.2e}")
