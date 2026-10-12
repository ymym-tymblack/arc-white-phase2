"""K=3 密テンソル版の検証: (1) Hermite 係数を MC と照合、(2) 層別の平均誤差を MC と比較。"""
import sys, time
import numpy as np
sys.path.insert(0, "research")
import k3_dense as k3

rng = np.random.default_rng(0)
# (1) 係数 c_k^{(p)} = E[ReLU(m+sY)^p He_k(Y)]
m = np.array([0.7, -0.4, 0.0, 1.5]); s = np.array([1.0, 0.6, 0.8, 0.3])
Y = rng.standard_normal(4_000_000)[:, None]
g = np.maximum(m + s * Y, 0.0)
cs = k3.relu_coeffs(m, s, 8)
for p in (1, 2, 3):
    worst = 0.0
    for kk in range(9):
        mc = (g**p * k3.he(kk, Y)).mean(0)
        worst = max(worst, np.abs(mc - cs[p][kk]).max() / (1 + np.abs(cs[p][kk]).max()))
    print(f"coeff p={p}: 最大相対差 {worst:.2e}")

# (2) 層別誤差
def mc_means(W, N, seed=1, chunk=100000):
    n = W[0].shape[0]; r = np.random.default_rng(seed); acc = np.zeros((len(W), n)); done = 0
    while done < N:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for l, w in enumerate(W):
            x = np.maximum(x @ w, 0); acc[l] += x.sum(0, dtype=np.float64)
        done += chunk
    return acc / done

n, depth = int(sys.argv[1]), int(sys.argv[2])
for seed in (0, 1):
    r = np.random.default_rng(100 + seed)
    W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
    ref = mc_means(W, 4_000_000)
    noise = None
    res = {}
    for name, kw in [("K2", dict(use_tau=False, kmax=6)), ("K3 e2", dict(use_tau=True, kmax=6, emax=2)),
                     ("K3 e3", dict(use_tau=True, kmax=6, emax=3))]:
        t = time.time(); mu = k3.propagate(W, **kw); res[name] = ((mu - ref) ** 2).mean(1)
        print(f"  seed{seed} {name}: {time.time()-t:.1f}s")
    print(f"seed {seed} n={n}  層ごと MSE (L=1..{depth})")
    for name, v in res.items():
        print(f"  {name:6s}", " ".join(f"{x:.1e}" for x in v))
