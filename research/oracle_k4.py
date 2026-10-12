"""κ4 の効果の上限: MC で測った真の κ4 スライスを密 K3 に注入して誤差が下がる量を見る。"""
import sys
import numpy as np
sys.path.insert(0, "research")
import k3_dense as k3
from ablate_k3 import mc_means
from mc_cumulants import mc_z_kappa4

n, depth = int(sys.argv[1]), int(sys.argv[2])
FULL = {"rho","tau1","tau1e","txxy","txxx","pairtau","meantau","iik","iii"}
for seed in (0, 1):
    r = np.random.default_rng(100 + seed)
    W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
    ref = mc_means(W, 4_000_000)
    k4 = mc_z_kappa4(W, 4_000_000)
    f = lambda mu: ((mu - ref) ** 2).mean(1)
    k3.GROUPS = set(FULL)
    base = k3.propagate([w.astype(np.float64) for w in W], kmax=6, emax=2)
    res = {"K3 (κ4なし)": f(base)}
    # κ4 注入: (omega のみ) / (omega + iijj) / (全部)
    om = [(a, np.zeros_like(b), np.zeros_like(c)) for a, b, c in k4]
    oi = [(a, b, np.zeros_like(c)) for a, b, c in k4]
    res["+omega"] = f(k3.propagate([w.astype(np.float64) for w in W], kmax=6, emax=2, kap4=om))
    res["+omega+iijj"] = f(k3.propagate([w.astype(np.float64) for w in W], kmax=6, emax=2, kap4=oi))
    res["+omega+iijj+iiij"] = f(k3.propagate([w.astype(np.float64) for w in W], kmax=6, emax=2, kap4=k4))
    print(f"seed {seed} n={n} depth={depth}  最終層 / 全層平均")
    for k, v in res.items():
        print(f"  {k:18s} {v[-1]:.2e}  {v.mean():.2e}")
    print("  実測 omega 平均(層ごと):", " ".join(f"{a.mean():.3f}" for a, _, _ in k4))
