"""因子化 K=3 キュムラント伝播(源を「家族」にまとめた版)。提出用、flopscope のみ。

前の版(k3_factored.py)から変えた点。いずれも ARC の公式参照実装(mlp_kprop, K=3 SIMPLE)と
RTX 5090 上の MC 真値との層別照合で見つけた抜けを埋めるもの(research/README.md)。

1. κ4 の放射部分を、平均・分散だけでなく共分散と「同じ添字が現れる 3 次の切り口」にも入れる。
   参照実装と同じく標準化した κ4 を κ̃4_iiii = ω、κ̃4_iijj = ω/3(i≠j)と置く。
2. 同一添字の 3 次切り口 κ[h_i,h_i,h_k](と κ3[h_i])を正しい値に直す補正 Δ を、
   各層で生まれる源と同じ「家族」に入れて運ぶ。Δ は
       Δ ≈ diag(e) (ρ ∘ 1 c1ᵀ) + Q B(低ランク) + 対角
   と分けられ(主要項は誕生源の脚 A と同じ形)、別の密な脚を運ばずに済む。

家族 f(層 b で誕生)が表す、層 l の事前活性化 z の 3 次キュムラント:
    T_abc = Σ_q c2_q Sym(A A M)_{abc,q} + Σ_q Sym(M M PΔ)_{abc,q}
    Sym(XXY)_{abc,q} = X_aq X_bq Y_cq + X_aq Y_bq X_cq + Y_aq X_bq X_cq
    PΔ = A diag(e) + M diag(dd) + Z Qᵀ
  M は伝播行列(誕生時 Wᵀ)、A = M diag(c1) ρ、Z は n×r の細い脚、Q・e・dd・c2 は誕生時に固定。
  輸送は X ← Wᵀ diag(Φ(α)) X(X = A, M, Z)。ReLU が読むのは
    D3_a  = T_aaa = 3 Σ_q (c2 A² M + M² PΔ)_aq
    D21   = T_aab = X_M Mᵀ + X_A Aᵀ + ((M∘M) Q) Zᵀ
            X_M = (A∘A) diag(c2) + (M∘M) diag(dd) + 2 (M∘PΔ)
            X_A = 2 (A∘M) diag(c2) + (M∘M) diag(e)
  家族 1 つ・1 層あたりの密な行列積は 4 回(輸送 2 + 読み出し 2)で、補正なしの源と同じ。
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# 事前活性化 z の超過尖度の層平均 ω_ℓ ≈ G[ℓ] / n(ℓ=0..15 は 1〜16 層目)。
# 幅1024・16層の MLP 3 個の MC 実測(results/layer_error.json の kurt × 1024)。
# 参照実装が解析的に出す層平均とも 2〜15% で一致する。16 層より深い場合は 約1.8/層で外挿。
_OMEGA_G = (
    0.05, 9.9, 16.94, 21.82, 26.71, 31.42, 35.48, 39.19,
    42.84, 45.89, 49.41, 52.42, 55.03, 58.67, 60.49, 62.31,
)


def _omega_g(layer: int) -> float:
    if layer < len(_OMEGA_G):
        return _OMEGA_G[layer]
    return _OMEGA_G[-1] + 1.8 * (layer - len(_OMEGA_G) + 1)


def _hermite_neg_alpha(alpha, upto: int):
    """He_0..He_upto(-alpha) のリスト。"""
    x = -alpha
    out = [fnp.ones_like(x), x]
    for j in range(1, upto):
        out.append(x * out[j] - float(j) * out[j - 1])
    return out[: upto + 1]


def _estimate_flops(n: int, depth: int, window: int, rank: int) -> float:
    """解析的な FLOP 見積り(行列積 1 回 = 2 n³)。少し多めに見積もる。

    層ごと: 共分散のサンドイッチ 3.22e9 (n/1024)³ + 誕生 2 n³ + 補正の低ランク化 ≈ 12 n² r
    家族 1 つ・1 層あたり: 8 n³(輸送 2 + 読み出し 2)+ 細い脚 ≈ 10 n² r
    """
    n3 = float(n) ** 3
    base = 3.22e9 * (n / 1024.0) ** 3 + 2.0 * n3 + 12.0 * n * n * rank
    total = 0.0
    for layer in range(depth):
        total += base + (8.0 * n3 + 10.0 * n * n * rank) * min(layer, window)
    return total


def _fit_window(n: int, depth: int, budget: int, window: int, rank: int,
                margin: float = 0.85) -> int:
    """予算を超えないよう窓を縮める(超えるとその MLP の予測はゼロになり、割引もない)。"""
    while window > 0 and _estimate_flops(n, depth, window, rank) > margin * budget:
        window -= 1
    return window


def _pair_terms(cA, cB, R, kmax, t_iij, t_ijj, tdiag, k4):
    """Cov[gA(z_i), gB(z_j)](i≠j)を Hermite 係数で展開した和。

    ρ の辺だけ: Σ_k cA_k cB_k ρ^k/k!
    3 次ブロック 1 個: (cA_k cB_{k-1} τ_iij + cA_{k-1} cB_k τ_ijj) ρ^{k-2}/(2 (k-2)!)
                      + (cA_{k+3} τ_iii cB_k + cA_k cB_{k+3} τ_jjj) ρ^k/(6 k!)
    κ4 放射ブロック 1 個: cA_{2+e} cB_{2+e} q_i q_j ρ^e/(4 e!) + (cA_{4+e} ω_i cB_e + 対称) ρ^e/(24 e!)
    """
    out = fnp.multiply(fnp.outer(cA[1], cB[1]), R[1])
    for k in range(2, kmax + 1):
        out = out + fnp.multiply(fnp.outer(cA[k], cB[k]), R[k])
    if t_iij is not None:
        for k in range(2, kmax + 1):
            Rk2 = R[0] if k == 2 else R[k - 2]
            out = out + fnp.multiply(
                fnp.outer(cA[k], cB[k - 1]) * t_iij + fnp.outer(cA[k - 1], cB[k]) * t_ijj, Rk2
            ) * 0.5
    for k in range(1, kmax + 1):
        out = out + (fnp.outer(cA[k + 3] * tdiag, cB[k]) + fnp.outer(cA[k], cB[k + 3] * tdiag)) * (
            R[k] / 6.0
        )
    if k4 is not None:
        om, k22 = k4
        out = out + fnp.multiply(fnp.outer(cA[2], cB[2]), k22) * 0.25
        for e in range(1, kmax + 1):
            out = out + fnp.multiply(fnp.outer(cA[2 + e], cB[2 + e]), k22 * R[e]) * 0.25
            out = out + (fnp.outer(cA[4 + e] * om, cB[e]) + fnp.outer(cA[e], cB[4 + e] * om)) * (
                R[e] / 24.0
            )
    return out


class Estimator(BaseEstimator):
    kmax = 3  # Mehler 展開の次数(共分散)
    window = 7  # 家族を何層分持つか(古い家族を捨てて FLOP を減らす)
    iik_rank = 16  # 補正 Δ の残差の低ランク近似の階数(None なら補正しない)
    use_k4 = True  # κ4 の放射部分を共分散・同一添字切り口に入れる
    omega_scale = 1.0  # ω の全体倍率(理論値 1)
    tau_scale = 1.0  # 3 次キュムラントの倍率(理論値 1)
    calib = 0.0  # 最終層の出力に掛ける縮小係数(正解 ≈ (1 - calib) × 予測)
    collect = False  # True なら層ごとの内部量を self.features に残す(解析用)

    def __init__(self, **kwargs) -> None:
        self._setup_rng = None
        for key, value in kwargs.items():
            if not hasattr(type(self), key):
                raise TypeError(f"unknown option {key}")
            setattr(self, key, value)
        self.features = []

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n, depth, kmax = mlp.width, mlp.depth, self.kmax
        rank = 0 if self.iik_rank is None else self.iik_rank
        window = _fit_window(n, depth, budget, depth if self.window is None else self.window, rank)
        f32 = fnp.float32

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        families: list[dict] = []
        rows = []

        for layer, w in enumerate(mlp.weights):
            last = layer == depth - 1

            # --- 線形層: z の平均・共分散 ---
            m = w.T @ mu
            S = fnp.einsum("ij,ia,jb->ab", C, w, w)
            var = fnp.maximum(fnp.diag(S), 1e-12)
            sigma = fnp.sqrt(var)
            alpha = m / sigma
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)

            # --- 3 次キュムラントの読み出し(D3 = T_aaa, D21 = T_aab) ---
            D3 = fnp.zeros(n, dtype=f32)
            D21 = None
            for f in families:
                A, M, c2 = f["A"], f["M"], f["c2"]
                AA = A * A
                d3 = AA * M * c2[None, :]
                if "e" in f:
                    PD = A * f["e"][None, :] + M * f["dd"][None, :]
                    if f["Z"] is not None:
                        PD = PD + fnp.matmul(f["Z"], f["Q"].T)
                    MM = M * M
                    d3 = d3 + MM * PD
                D3 = D3 + 3.0 * fnp.sum(d3, axis=1)
                if last:
                    continue
                XM = AA * c2[None, :]
                XA = 2.0 * (A * M) * c2[None, :]
                t = None
                if "e" in f:
                    XM = XM + MM * f["dd"][None, :] + 2.0 * (M * PD)
                    XA = XA + MM * f["e"][None, :]
                    if f["Z"] is not None:
                        t = fnp.matmul(fnp.matmul(MM, f["Q"]), f["Z"].T)
                t2 = fnp.matmul(XM, M.T) + fnp.matmul(XA, A.T)
                t = t2 if t is None else t + t2
                D21 = t if D21 is None else D21 + t
            D3_raw, D21_raw = D3, D21  # 家族が表す値そのもの(倍率を掛ける前)
            if self.tau_scale != 1.0:
                D3 = D3 * self.tau_scale
                if D21 is not None:
                    D21 = D21 * self.tau_scale
            tdiag = D3 / (sigma * var)  # τ_iii = T_iii / σ³
            omega = self.omega_scale * _omega_g(layer) / n

            # --- Hermite 係数 (p=1) ---
            top = max(kmax + 4, 7)
            he = _hermite_neg_alpha(alpha, top)
            c = [m * Phi + sigma * phi, sigma * Phi]
            for k in range(2, top + 1):
                c.append(sigma * phi * he[k - 2])

            # --- 平均と E[h²](単一頂点の 3 次・4 次ブロック補正) ---
            mu_new = c[0] + c[3] * tdiag / 6.0 + c[6] * tdiag * tdiag / 72.0 + c[4] * (omega / 24.0)
            e2_0 = (m * m + var) * Phi + m * sigma * phi
            ez2 = (
                e2_0 + 2.0 * var * phi * tdiag / 6.0
                + 2.0 * var * phi * he[3] * tdiag * tdiag / 72.0
                + 2.0 * var * phi * he[1] * (omega / 24.0)
            )
            var_post = fnp.maximum(ez2 - mu_new * mu_new, 0.0)
            if self.collect:
                self.features.append({
                    "m": m, "sigma": sigma, "alpha": alpha, "tdiag": tdiag, "mu": mu_new,
                    "mu_gauss": c[0], "var_post": var_post, "D3": D3_raw,
                    "D21": D21_raw if D21_raw is not None else fnp.zeros((n, n), dtype=f32),
                })
            mu = mu_new
            rows.append(mu)
            if last:
                break

            # --- ReLU 後の共分散 ---
            rho = S / fnp.outer(sigma, sigma)
            if D21 is not None:
                t_iij = D21 / fnp.outer(var, sigma)  # T_iij / (σ_i² σ_j)
                t_ijj = t_iij.T
            else:
                t_iij = t_ijj = None
            R = [fnp.ones_like(rho), rho]  # R[k] = ρ^k / k!
            rp = rho
            for k in range(2, kmax + 1):
                rp = rp * rho
                R.append(rp / float(math.factorial(k)))
            k4 = None
            if self.use_k4:
                om = omega * fnp.ones_like(sigma)
                q = fnp.sqrt(om / 3.0)
                k4 = (om, fnp.outer(q, q))
            Cn = _pair_terms(c, c, R, kmax, t_iij, t_ijj, tdiag, k4)
            fnp.fill_diagonal(Cn, var_post)
            C = flops.as_symmetric(Cn, symmetry=(0, 1))

            # --- 同一添字の 3 次切り口の補正 Δ ---
            #   u[i,k] = κ[h_i,h_i,h_k] = Cov(h_i², h_k) - 2 μ_i C_ik   (i≠k)
            #   κ3[h_i] = E[h³] - 3 μ E[h²] + 2 μ³
            # 家族が既に表している値(誕生: 2 c1_i c2_i c1_k ρ_ik + c1_i² c2_k ρ_ik²、
            # 輸送: Φ_i² Φ_k T_iik)との差を Δ として、新しい家族に入れる。
            w_next = mlp.weights[layer + 1]
            wT = w_next.T
            fam = {"c2": c[2], "age": 0}
            fam["A"] = fnp.matmul(wT, c[1][:, None] * rho)
            fam["M"] = wT
            if self.iik_rank is not None:
                c2p = [e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]
                for k in range(3, kmax + 5):
                    c2p.append(2.0 * var * phi * he[k - 3])
                u = _pair_terms(c2p, c, R, kmax, t_iij, t_ijj, tdiag, k4) - 2.0 * mu[:, None] * Cn
                ez3 = (
                    m * e2_0 + 2.0 * var * c[0]
                    + 6.0 * var * sigma * Phi * tdiag / 6.0
                    + 6.0 * var * sigma * phi * he[2] * tdiag * tdiag / 72.0
                )
                if k4 is not None:
                    ez3 = ez3 + 6.0 * var * sigma * phi * (omega / 24.0)
                k3d = ez3 - 3.0 * mu * ez2 + 2.0 * mu * mu * mu
                u_src = 2.0 * fnp.outer(c[1] * c[2], c[1]) * rho + fnp.outer(c[1] * c[1], c[2]) * (
                    rho * rho
                )
                k3_src = 3.0 * c[2] * c[1] * c[1]
                if D21_raw is not None:
                    u_src = u_src + fnp.outer(Phi * Phi, Phi) * D21_raw
                    k3_src = k3_src + Phi * Phi * Phi * D3_raw
                delta = u - u_src
                fnp.fill_diagonal(delta, 0.0)
                # 主要項: 行 i ごとに base_ik = c1_k ρ_ik の倍数 e_i で近似(最小二乗)
                base = rho * c[1][None, :]
                fnp.fill_diagonal(base, 0.0)
                e = fnp.sum(delta * base, axis=1) / fnp.maximum(fnp.sum(base * base, axis=1), 1e-30)
                fam["e"] = e
                fam["dd"] = (k3d - k3_src) / 3.0 - c[1] * e
                fam["Z"] = None
                if rank > 0:
                    # 残差の低ランク近似(乱択レンジファインダー、べき乗 1 回)。
                    # 試行行列は乱数でなく次層の重みの先頭 r 列(決定的)。
                    resid = delta - e[:, None] * base
                    Y = fnp.matmul(resid, w_next[:, :rank])
                    Y = fnp.matmul(resid, fnp.matmul(resid.T, Y))
                    Q, _ = fnp.linalg.qr(Y)
                    B = fnp.matmul(Q.T, resid)  # (r, n)
                    fam["Q"] = Q
                    fam["Z"] = fnp.matmul(wT, B.T)  # (n, r)

            # --- 家族の輸送(次層の重みと ReLU の傾き Φ)と誕生 ---
            moved = []
            for f in families:
                if f["age"] + 1 >= window:
                    continue  # 窓の外に出る家族は運ばずに捨てる
                g = dict(f)
                g["age"] = f["age"] + 1
                g["A"] = fnp.matmul(wT, Phi[:, None] * f["A"])
                g["M"] = fnp.matmul(wT, Phi[:, None] * f["M"])
                if f.get("Z") is not None:
                    g["Z"] = fnp.matmul(wT, Phi[:, None] * f["Z"])
                moved.append(g)
            if window > 0:
                moved.append(fam)
            families = moved

        if self.calib:
            rows[-1] = rows[-1] * (1.0 - self.calib)
        return fnp.stack(rows, axis=0)
