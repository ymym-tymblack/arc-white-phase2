"""K=3 キュムラント伝播(密テンソル版、研究用・小幅専用)。

提出用ではない(NumPy を直接使う)。数式の検証が目的で、幅 n=64〜128 を想定する。

論文 arXiv:2605.05179 の Theorem A.2(Hermite ベースの diagram summation formula)を使う。
各ニューロンの事前活性化 z_i を平均 m_i・分散 s_i² で標準化して Y_i とし、
活性化 g の Hermite 係数 c_k = s^k E[g^{(k)}(z)] (z~N(m_i, s_i²)) を使う。

  κ[g_1(z_i), g_2(z_j), ...] = Σ_{k,l,...} Π c / Π k! × (接続図の総和)

接続図は、各頂点(=関数適用ごと)の k 本の足を、
  - 2 本ブロック(別頂点の足同士のみ。重み ρ = 標準化した共分散)
  - 3 本ブロック(同一頂点内も可。重み τ = 標準化した 3 次キュムラント)
に分割したもの。τ については 1 次まで(3 本ブロックは高々1個、単一頂点の平均補正のみ 2 個まで)。

状態は z = (m, S, T) → ReLU → h = (μ, C, U) → 線形 z' = Wᵀh。
重みは local_engine と同じ向き(x @ W)。
"""

from __future__ import annotations

import itertools
import math

import numpy as np

_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def npdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def he(m: int, x: np.ndarray) -> np.ndarray:
    """確率論者の Hermite 多項式 He_m(x)。"""
    if m == 0:
        return np.ones_like(x)
    p, q = np.ones_like(x), x
    for j in range(1, m):
        p, q = q, x * q - j * p
    return q


def relu_coeffs(m: np.ndarray, s: np.ndarray, kmax: int) -> dict[int, np.ndarray]:
    """g_p(z) = ReLU(z)^p (p=1,2,3) の Hermite 係数 c_k^{(p)}(k=0..kmax)、形状 (kmax+1, n)。

    E[δ^{(m)}(z)] = s^{-m-1} He_m(-a) φ(a) (a=m/s) を使った閉形式。
    """
    a = m / s
    P, ph = ncdf(a), npdf(a)
    i0 = P
    i1 = m * P + s * ph
    i2 = m * i1 + s**2 * i0
    i3 = m * i2 + 2 * s**2 * i1
    out = {p: np.zeros((kmax + 1, m.size)) for p in (1, 2, 3)}
    for k in range(kmax + 1):
        out[1][k] = i1 if k == 0 else s * P if k == 1 else s * ph * he(k - 2, -a)
        if k == 0:
            out[2][k] = i2
            out[3][k] = i3
        elif k == 1:
            out[2][k] = 2 * s * i1
            out[3][k] = 3 * s * i2
        elif k == 2:
            out[2][k] = 2 * s**2 * P
            out[3][k] = 6 * s**2 * i1
        elif k == 3:
            out[2][k] = 2 * s**2 * ph * he(0, -a)
            out[3][k] = 6 * s**3 * P
        else:
            out[2][k] = 2 * s**2 * ph * he(k - 3, -a)
            out[3][k] = 6 * s**3 * ph * he(k - 4, -a)
    return out


def rho_pows(rho: np.ndarray, emax: int) -> list[np.ndarray]:
    """R[e] = rho^e / e! (要素ごと)。"""
    out = [np.ones_like(rho)]
    for e in range(1, emax + 1):
        out.append(out[-1] * rho / e)
    return out


def pair_cov(cA, cB, rho, t_iij, t_ijj, t_iii, t_jjj, kmax: int, use_tau: bool = True):
    """Cov[gA(z_i), gB(z_j)] (i≠j) の n×n 行列。cA, cB は (kmax+4, n)。

    τ=0 の項は Σ_k cA_k cB_k ρ^k / k!。τ 1 次の項は図(2 本辺 + 3 本ブロック 1 個)から:
      τ_iij: k 本の i と k-1 本の j → c_k(i) c_{k-1}(j) τ_iij ρ^{k-2} / (2 (k-2)!)
      τ_ijj: 対称
      τ_iii: l+3 本の i と l 本の j → c_{l+3}(i) c_l(j) τ_iii ρ^l / (6 l!)
      τ_jjj: 対称
    """
    R = rho_pows(rho, kmax)
    tot = np.zeros_like(rho)
    for k in range(1, kmax + 1):
        tot += np.outer(cA[k], cB[k]) * R[k]
    if not use_tau:
        return tot
    for k in range(2, kmax + 1):
        tot += np.outer(cA[k], cB[k - 1]) * t_iij * R[k - 2] / 2.0
        tot += np.outer(cA[k - 1], cB[k]) * t_ijj * R[k - 2] / 2.0
    for k in range(1, kmax + 1):
        tot += np.outer(cA[k + 3] * t_iii, cB[k]) * R[k] / 6.0
        tot += np.outer(cA[k], cB[k + 3] * t_jjj) * R[k] / 6.0
    return tot


def corrected_raw_mean(c, tau_diag, use_tau: bool = True):
    """E[g(Y_i)] = c_0 + c_3 τ/6 + c_6 τ²/72 (単一頂点: 3 本ブロックのみ)。"""
    out = c[0].copy()
    if use_tau:
        out += c[3] * tau_diag / 6.0 + c[6] * tau_diag**2 / 72.0
    return out


def _place1(v, a, n):
    sh = [1, 1, 1]
    sh[a] = n
    return v.reshape(sh)


def _place2(M, a, b, n):
    """result[idx] = M[idx[a], idx[b]] (a≠b)。"""
    if a > b:
        M, a, b = M.T, b, a
    sh = [1, 1, 1]
    sh[a] = n
    sh[b] = n
    return M.reshape(sh)


def third_distinct(c1, rho, tau3, emax: int, use_tau: bool = True):
    """κ[h_i, h_j, h_k] (i,j,k 相異)。ρ 辺の総数 ≤ emax、τ ブロックは高々 1 個。

    頂点 x,y,z の足の数を deg とすると、辺の数 e_xy,e_xz,e_yz から
      τ なし:  Π c_deg · Π R[e]         (非零の辺が 2 種以上)
      τ_xyz:   Π c_deg · τ_xyz · Π R[e]   (deg = 1 + 隣接辺)
      τ_xxy:   同上 / 2        (deg_x = 2 + ..., deg_y = 1 + ..., z は辺で接続)
      τ_xxx:   同上 / 6        (deg_x = 3 + ..., 全頂点が接続)
    """
    n = rho.shape[0]
    R = rho_pows(rho, emax)
    U = np.zeros((n, n, n))
    kc = c1.shape[0]

    def coef(deg):
        if max(deg) >= kc:
            return None
        return (
            _place1(c1[deg[0]], 0, n) * _place1(c1[deg[1]], 1, n) * _place1(c1[deg[2]], 2, n)
        )

    def edges(e01, e02, e12):
        return (
            _place2(R[e01], 0, 1, n) * _place2(R[e02], 0, 2, n) * _place2(R[e12], 1, 2, n)
        )

    for e01, e02, e12 in itertools.product(range(emax + 1), repeat=3):
        if e01 + e02 + e12 > emax:
            continue
        nz = sum(x > 0 for x in (e01, e02, e12))
        E = None
        if nz >= 2:
            c = coef((e01 + e02, e01 + e12, e02 + e12))
            if c is not None:
                E = edges(e01, e02, e12)
                U += c * E
        if not use_tau:
            continue
        # τ_ijk
        c = coef((1 + e01 + e02, 1 + e01 + e12, 1 + e02 + e12))
        if c is not None:
            E = edges(e01, e02, e12)
            U += c * tau3 * E
        # τ_xxy: (x, y, z) は軸の置換
        for x, y, z in itertools.permutations(range(3)):
            exy = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((x, y)))]
            exz = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((x, z)))]
            eyz = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((y, z)))]
            if exz + eyz < 1:
                continue
            deg = [0, 0, 0]
            deg[x], deg[y], deg[z] = 2 + exy + exz, 1 + exy + eyz, exz + eyz
            c = coef(tuple(deg))
            if c is None:
                continue
            txxy = np.einsum("iij->ij", tau3)
            U += c * _place2(txxy, x, y, n) * edges(e01, e02, e12) / 2.0
        # τ_xxx
        for x in range(3):
            y, z = [a for a in range(3) if a != x]
            exy = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((x, y)))]
            exz = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((x, z)))]
            eyz = {(0, 1): e01, (0, 2): e02, (1, 2): e12}[tuple(sorted((y, z)))]
            conn_y = exy > 0 or (eyz > 0 and exz > 0)
            conn_z = exz > 0 or (eyz > 0 and exy > 0)
            if not (conn_y and conn_z):
                continue
            deg = [0, 0, 0]
            deg[x], deg[y], deg[z] = 3 + exy + exz, exy + eyz, exz + eyz
            c = coef(tuple(deg))
            if c is None:
                continue
            txxx = np.einsum("iii->i", tau3)
            U += c * _place1(txxx, x, n) * edges(e01, e02, e12) / 6.0
    return U




def state_to_h(m, S, T, kmax: int, emax: int, use_tau: bool):
    """z = (m, S, T) から h = ReLU(z) の (μ, C, U) を返す。"""
    n = m.size
    s = np.sqrt(np.maximum(np.diag(S), 1e-12))
    rho = S / np.outer(s, s)
    kc = max(kmax + 3, 2 * emax + 3, 6) + 1
    cs = relu_coeffs(m, s, kc)
    c1, c2, c3 = cs[1], cs[2], cs[3]
    tau3 = T / (s[:, None, None] * s[None, :, None] * s[None, None, :]) if use_tau else None
    tdiag = np.einsum("iii->i", tau3) if use_tau else np.zeros(n)

    mu = corrected_raw_mean(c1, tdiag, use_tau)
    m2 = corrected_raw_mean(c2, tdiag, use_tau)
    m3 = corrected_raw_mean(c3, tdiag, use_tau)
    var = np.maximum(m2 - mu**2, 0.0)
    k3d = m3 - 3 * mu * m2 + 2 * mu**3

    zero = np.zeros((n, n))
    if use_tau:
        t_iij = np.einsum("iij->ij", tau3)
        t_ijj = np.einsum("ijj->ij", tau3)
    else:
        t_iij = t_ijj = zero
    C = pair_cov(c1, c1, rho, t_iij, t_ijj, tdiag, tdiag, kmax, use_tau)
    np.fill_diagonal(C, var)

    if not use_tau:
        return mu, C, None

    # Cov[h_i^2, h_k] → κ[h_i, h_i, h_k] = Cov(h_i²,h_k) - 2 μ_i C_ik
    cov_sq = pair_cov(c2, c1, rho, t_iij, t_ijj, tdiag, tdiag, kmax, True)
    u_iik = cov_sq - 2.0 * mu[:, None] * C  # [i,k] = κ[h_i,h_i,h_k]
    U = third_distinct(c1, rho, tau3, emax, True)
    for i in range(n):
        U[i, i, :] = u_iik[i]
        U[i, :, i] = u_iik[i]
        U[:, i, i] = u_iik[i]
        U[i, i, i] = k3d[i]
    return mu, C, U


def linear_step(W, mu, C, U):
    """z' = Wᵀ h の (m, S, T)。W は (in, out)。"""
    m = W.T @ mu
    S = W.T @ C @ W
    T = None
    if U is not None:
        T = np.einsum("ijk,ia->ajk", U, W)
        T = np.einsum("ajk,jb->abk", T, W)
        T = np.einsum("abk,kc->abc", T, W)
    return m, S, T


def propagate(weights, kmax: int = 6, emax: int = 3, use_tau: bool = True):
    """各層の h の平均 μ (depth, n) を返す。K=2 (use_tau=False) は Mehler 共分散伝播。"""
    n = weights[0].shape[0]
    mu = np.zeros(n)
    C = np.eye(n)
    U = np.zeros((n, n, n)) if use_tau else None
    rows = []
    for W in weights:
        m, S, T = linear_step(W.astype(np.float64), mu, C, U)
        mu, C, U = state_to_h(m, S, T, kmax, emax, use_tau)
        rows.append(mu)
    return np.stack(rows)
