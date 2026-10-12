"""計画2: κ̃4[z_i,z_j,z_k,z_k] を MC で測り、(1) ρ_ik ρ_jk への比例を回帰で調べ、
(2) 真の値/比例則の値を密版の 3 次の誕生に入れたときの効果を測る。"""
import sys

import numpy as np

sys.path.insert(0, "research")
import k3_dense as k3  # noqa: E402
from ablate_k3 import mc_means  # noqa: E402


def mc_k4_ijkk(W, N, seed=7, chunk=10_000):
    """層ごとの K[i,j,k] = κ4[z_i,z_j,z_k,z_k] / (s_i s_j s_k²) と ρ。"""
    depth, n = len(W), W[0].shape[0]
    r = np.random.default_rng(seed)
    # 平均を先に(別標本)
    mean = np.zeros((depth, n)); done = 0
    while done < N // 4:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for lay, w in enumerate(W):
            z = x @ w; mean[lay] += z.sum(0, dtype=np.float64); x = np.maximum(z, 0)
        done += chunk
    mean /= done
    M4 = np.zeros((depth, n, n, n)); M2 = np.zeros((depth, n, n)); done = 0
    r = np.random.default_rng(seed + 1)
    while done < N:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for lay, w in enumerate(W):
            z = x @ w; x = np.maximum(z, 0)
            c = z - mean[lay].astype(np.float32)
            M2[lay] += (c.T @ c).astype(np.float64)
            pair = (c[:, :, None] * c[:, None, :]).reshape(chunk, n * n)
            M4[lay] += (pair.T @ (c * c)).astype(np.float64).reshape(n, n, n)
        done += chunk
    out = []
    for lay in range(depth):
        C = M2[lay] / done; v = np.diag(C); s = np.sqrt(v)
        E4 = M4[lay] / done
        # κ4[i,j,k,k] = E[c_i c_j c_k²] - E[c_i c_j] E[c_k²] - 2 E[c_i c_k] E[c_j c_k]
        k4 = E4 - C[:, :, None] * v[None, None, :] - 2.0 * np.einsum("ik,jk->ijk", C, C)
        K = k4 / (s[:, None, None] * s[None, :, None] * v[None, None, :])
        rho = C / np.outer(s, s)
        out.append((K, rho))
    return out


if __name__ == "__main__":
    n, depth = int(sys.argv[1]), int(sys.argv[2])
    FULL = {"rho", "tau1", "pairtau", "meantau", "iik", "iii"}
    for seed in (0, 1):
        r = np.random.default_rng(100 + seed)
        W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(depth)]
        ref = mc_means(W, 4_000_000)
        meas = mc_k4_ijkk(W, 2_000_000)
        ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        keep = (ii != jj) & (jj != kk) & (ii != kk)
        idx = np.stack([ii[keep], jj[keep], kk[keep]], 1)
        lams = []
        print(f"seed {seed}: 層ごとの回帰 K_ijkk ≈ λ ρ_ik ρ_jk (+ μ ρ_ij)")
        for lay, (K, rho) in enumerate(meas):
            y = K[idx[:, 0], idx[:, 1], idx[:, 2]]
            x1 = rho[idx[:, 0], idx[:, 2]] * rho[idx[:, 1], idx[:, 2]]
            x2 = rho[idx[:, 0], idx[:, 1]]
            lam = (x1 @ y) / (x1 @ x1)
            r2a = 1 - ((y - lam * x1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            X = np.stack([x1, x2], 1); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2b = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            lams.append(lam)
            print(f"  層{lay+1}: λ={lam:.4f} R²={r2a:.3f} | 2変数 {coef.round(4)} R²={r2b:.3f}  |K|rms={np.sqrt((y**2).mean()):.4f}")
        Wd = [w.astype(np.float64) for w in W]
        f = lambda mu: ((mu - ref) ** 2).mean(1)  # noqa: E731
        k3.GROUPS = set(FULL)
        base = f(k3.propagate(Wd, kmax=3, emax=2))
        true_ = f(k3.propagate(Wd, kmax=3, emax=2, k4birth=[K for K, _ in meas]))
        model = f(k3.propagate(Wd, kmax=3, emax=2,
                               k4birth=[lams[l] * np.einsum("ik,jk->ijk", rho, rho) for l, (_, rho) in enumerate(meas)]))
        print(f"  最終層 MSE: iik版 {base[-1]:.2e} | +真のK {true_[-1]:.2e} | +比例則 {model[-1]:.2e}")
        print(f"  全層 MSE  : iik版 {base.mean():.2e} | +真のK {true_.mean():.2e} | +比例則 {model.mean():.2e}")
