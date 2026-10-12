"""MC で事前活性化 z の 4 次キュムラントのスライスを層ごとに測る(小幅用)。"""
import numpy as np


def mc_z_kappa4(W, N, seed=2, chunk=100000):
    """返り値: 層ごとの (omega, k_iijj, k_iiij)(標準化済み: κ4 / (s...))。

    omega[i]    = κ4[z_i,z_i,z_i,z_i] / s_i^4
    k_iijj[i,j] = κ4[z_i,z_i,z_j,z_j] / (s_i² s_j²)
    k_iiij[i,j] = κ4[z_i,z_i,z_i,z_j] / (s_i³ s_j)
    """
    depth, n = len(W), W[0].shape[0]
    r = np.random.default_rng(seed)
    S1 = np.zeros((depth, n)); S2 = np.zeros((depth, n)); S3 = np.zeros((depth, n))
    S4 = np.zeros((depth, n))
    Z = np.zeros((depth, n, n))  # Σ z_i z_j
    Q = np.zeros((depth, n, n))  # Σ z_i² z_j²
    B = np.zeros((depth, n, n))  # Σ z_i³ z_j
    # 中心化が必要なので 2 パス: 平均を先に推定
    mean = np.zeros((depth, n)); done = 0
    while done < N // 4:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for l, w in enumerate(W):
            z = x @ w; mean[l] += z.sum(0, dtype=np.float64); x = np.maximum(z, 0)
        done += chunk
    mean /= done
    r = np.random.default_rng(seed + 1); done = 0
    while done < N:
        x = r.standard_normal((chunk, n)).astype(np.float32)
        for l, w in enumerate(W):
            z = x @ w; x = np.maximum(z, 0)
            c = (z - mean[l].astype(np.float32)).astype(np.float64)
            c2 = c * c
            S2[l] += c2.sum(0); Z[l] += c.T @ c; Q[l] += c2.T @ c2; B[l] += (c2 * c).T @ c
            S4[l] += (c2 * c2).sum(0)
        done += chunk
    out = []
    for l in range(depth):
        cov = Z[l] / done; v = np.diag(cov); s = np.sqrt(v)
        Eq = Q[l] / done
        k_iijj = Eq - np.outer(v, v) - 2 * cov**2
        Eb = B[l] / done
        k_iiij = Eb - 3 * v[:, None] * cov
        omega = S4[l] / done / v**2 - 3.0
        out.append((omega, k_iijj / np.outer(v, v), k_iiij / (s[:, None] ** 3 * s[None, :])))
    return out
