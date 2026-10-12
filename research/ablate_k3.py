"""どの項が効くかのアブレーション(n=128, 8層, 3 seed)。最終層 MSE と全層平均 MSE。

使い方: uv run python research/ablate_k3.py   (構成は下の CONFIGS を編集)
"""

import sys

import numpy as np

sys.path.insert(0, "research")
import k3_dense as k3  # noqa: E402

FULL = {"rho", "tau1", "tau1e", "txxy", "txxx", "pairtau", "meantau", "iik", "iii"}

# (名前, 有効な項の集合, emax)
CONFIGS = [
    ("full", FULL, 2),
    ("-iik", FULL - {"iik"}, 2),
    ("-iii", FULL - {"iii"}, 2),
    ("-iik-iii", FULL - {"iik", "iii"}, 2),
    ("min(rho,tau1,pair,mean,iik,iii)", {"rho", "tau1", "pairtau", "meantau", "iik", "iii"}, 2),
    ("min+txxy", {"rho", "tau1", "pairtau", "meantau", "iik", "iii", "txxy"}, 2),
]


def mc_means(W, N, seed=1, chunk=100000):
    n = W[0].shape[0]
    r = np.random.default_rng(seed)
    acc = np.zeros((len(W), n))
    done = 0
    while done < N:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for lay, w in enumerate(W):
            x = np.maximum(x @ w, 0)
            acc[lay] += x.sum(0, dtype=np.float64)
        done += chunk
    return acc / done


def main(n=128, depth=8, seeds=(0, 1, 2)):
    rows = {c[0]: [] for c in CONFIGS}
    rows["K2 (cov only)"] = []
    for seed in seeds:
        r = np.random.default_rng(100 + seed)
        W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
        ref = mc_means(W, 4_000_000)
        for name, grp, emax in CONFIGS:
            k3.GROUPS = set(grp)
            mu = k3.propagate(W, kmax=6, emax=emax, use_tau=True)
            rows[name].append(((mu - ref) ** 2).mean(1))
        k3.GROUPS = set(FULL)
        rows["K2 (cov only)"].append(((k3.propagate(W, kmax=6, use_tau=False) - ref) ** 2).mean(1))
    print(f"{'config':34s} 最終層   全層平均  (n={n}, depth={depth}, seeds={list(seeds)})")
    for name, v in rows.items():
        a = np.mean(v, 0)
        print(f"{name:34s} {a[-1]:.2e}  {a.mean():.2e}")


if __name__ == "__main__":
    nn = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    dd = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(nn, dd, seeds=(0, 1))
