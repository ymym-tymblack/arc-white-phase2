"""因子化 K=3 キュムラント伝播(提出用 flopscope 実装)。

research/k3_dense.py(密テンソル版・n=128 で MC 検証済み)の最小構成
  {rho, tau1, pairtau, meantau}
を n=1024 向けに因子化したもの。密版で必要と確認した項だけを持つ。

状態:
  mu_h, C : 前層 ReLU 後の平均と共分散
  sources : 事前活性化 z の 3 次キュムラント T を、源 b ごとの和で持つ
        T_abc = Σ_b Σ_q w_b[q] ( A_b[a,q] A_b[b,q] P_b[c,q] + (P を他の位置に置いた 2 通り) )
    A, P は n×n。w は長さ n。
  読み出し:
        D3[a]    = T_aaa = 3 Σ_q w_q A_aq² P_aq
        D21[a,b] = T_aab = (A∘A·w) Pᵀ + 2 (A∘P·w) Aᵀ     (n³ の行列積 2 回)
    ReLU は D3(平均補正)と D21(共分散補正)だけを読む。
  輸送(ReLU の傾き gain=Φ(α) と次層の重み W を通す):
        A ← Wᵀ diag(gain) A,   P ← Wᵀ diag(gain) P
  誕生(ReLU 後の共分散 ρ から。密版の rho 項 c1(i) c1(j) c2(k) ρ_ik ρ_jk に対応):
        A = Wᵀ diag(c1) ρ,  P = Wᵀ,  w = c2
    ここで c1 = Hermite 係数 c_1 = σΦ(α)、c2 = c_2 = σφ(α)。

k 次 Hermite 係数 (p=1): c_0 = μΦ+σφ, c_1 = σΦ, c_k = σφ He_{k-2}(-α) (k≥2)
p=2 の係数(分散用): c_0 = E[z²;z>0], c_3 = 2σ²φ, c_6 = 2σ²φ He_3(-α)
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


def _hermite_neg_alpha(alpha, upto: int):
    """He_0..He_upto(-alpha) のリスト(float32 ベクトル)。"""
    x = -alpha
    out = [fnp.ones_like(x), x]
    for j in range(1, upto):
        out.append(x * out[j] - float(j) * out[j - 1])
    return out[: upto + 1]


# 事前活性化 z の超過尖度の層平均 ω_ℓ ≈ G[ℓ] / n (ℓ=0..15 は 1〜16 層目)。
# 値は幅1024・16層の MLP 3 個の MC 実測(results/layer_error.json の kurt × 1024)。
# 幅128でも G/n が実測と一致することを確認済み(ω は 1/n に比例)。
# 16 層より深い場合は最後の傾き 約1.8/層 で外挿する。
_OMEGA_G = (
    0.05, 9.9, 16.94, 21.82, 26.71, 31.42, 35.48, 39.19,
    42.84, 45.89, 49.41, 52.42, 55.03, 58.67, 60.49, 62.31,
)


def _omega_g(layer: int) -> float:
    if layer < len(_OMEGA_G):
        return _OMEGA_G[layer]
    return _OMEGA_G[-1] + 1.8 * (layer - len(_OMEGA_G) + 1)


def _estimate_flops(n: int, depth: int, win_rho: int, win_iik: int) -> float:
    """解析的な FLOP 見積り(行列積 1 回 = 2 n³)。少し多めに見積もる。

    層ごとの基本コスト ≈ 共分散のサンドイッチ 3.22e9 (n/1024)³ + 誕生の行列積 2 n³
                          (+ iik 源の誕生 2 n³)
    源 1 個・1 層あたり  ≈ 8 n³(輸送 2 回 + 読み出し 2 回)
    """
    n3 = float(n) ** 3
    base = 3.22e9 * (n / 1024.0) ** 3 + 2.0 * n3 + (2.0 * n3 if win_iik > 0 else 0.0)
    total = 0.0
    for layer in range(depth):
        live = min(layer, win_rho) + min(layer, win_iik)
        total += base + 8.0 * n3 * live
    return total


def _fit_windows(n: int, depth: int, budget: int, win_rho: int, win_iik: int,
                 margin: float = 0.85) -> tuple[int, int]:
    """予算を超えないよう、大きい方の窓から 1 ずつ縮める。

    FLOP が予算を超えると、その MLP の予測はゼロに置き換わる(割引もなし)。
    競技の形状(1024×16)では縮めないが、深い MLP(スモークテストなど)で効く。
    """
    while _estimate_flops(n, depth, win_rho, win_iik) > margin * budget:
        if win_rho == 0 and win_iik == 0:
            break
        if win_iik >= win_rho and win_iik > 0:
            win_iik -= 1
        else:
            win_rho -= 1
    return win_rho, win_iik


class Estimator(BaseEstimator):
    # 既定値は計画3改(research/search_k3.py、結果は results/search_k3*.json)で決めた。
    # 探索用 MLP は mini 0〜39 番(40 個)。縮小係数 calib は設定ごとに解析的に最適化し、
    # 残り 5 変数を焼きなまし法(25 反復)→ ビームサーチ(幅 2、収束まで)で探索した。
    #   探索用 40 MLP: 6.06e-8 → 5.79e-8(前半 20 個・後半 20 個とも改善)
    #   最終確認用 60 MLP(40〜99 番、探索に不使用): 5.88e-8 → 5.47e-8
    # τ を下げて calib を上げる方向に改善が続き、τ=0.94 で止まった(0.92 は悪化)。
    # 倍率は理論値 1 からの当てはめで、MLP の形状が変われば再調整が必要。
    kmax = 2  # Mehler 展開の次数(共分散)
    use_omega = True  # κ4 対角(超過尖度)の平均・分散補正
    omega_scale = 0.8  # ω の全体倍率(理論値 1)
    window = 7  # 誕生した源を何層分持つか(古い源を捨てて FLOP を減らす)
    # 計画1: 同じニューロンが 2 回現れる 3 次の切り口 κ[h_i,h_i,h_k] と κ[h_i,h_i,h_i] を
    # 正しい値にする補正源を、何層分持つか(0 なら使わない)。生 MSE は下がるが、
    # FLOP の増加分を上回らず、どの探索でも window_iik=0 が最良または同点だった。
    window_iik = 0
    tau_scale = 0.94  # 3 次キュムラント(D3, D21)の倍率(理論値 1)
    # 最終層の出力に掛ける縮小係数。正解 ≈ (1 - calib) × 予測。
    # 9 特徴量の ridge(0.816)・GBM(0.83〜0.85)は、この 1 パラメータ補正を超えなかった(段階5)。
    calib = 5.09e-4
    collect = False  # True なら各層の内部量を self.features に残す(解析用。提出では使わない)

    def __init__(
        self,
        kmax: int | None = None,
        use_omega: bool | None = None,
        omega_scale: float | None = None,
        window: int | None = None,
        tau_scale: float | None = None,
        collect: bool | None = None,
        calib: float | None = None,
        window_iik: int | None = None,
    ) -> None:
        self._setup_rng = None
        if kmax is not None:
            self.kmax = kmax
        if use_omega is not None:
            self.use_omega = use_omega
        if omega_scale is not None:
            self.omega_scale = omega_scale
        if window is not None:
            self.window = window
        if tau_scale is not None:
            self.tau_scale = tau_scale
        if collect is not None:
            self.collect = collect
        if calib is not None:
            self.calib = calib
        if window_iik is not None:
            self.window_iik = window_iik
        self.features = []

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = mlp.depth
        win_rho, win_iik = _fit_windows(
            n, depth, budget, depth if self.window is None else self.window, self.window_iik
        )
        kmax = self.kmax
        f32 = fnp.float32

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        sources: list[list] = []  # [A, P, w, 種類, 年齢]
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

            # --- 3 次キュムラントの読み出し(D3, D21) ---
            D3 = fnp.zeros(n, dtype=f32)
            D21 = None
            for A, P, wq, _kind, _age in sources:
                AA = A * A
                D3 = D3 + 3.0 * fnp.sum(AA * P * wq[None, :], axis=1)
                if not last:
                    AAw = AA * wq[None, :]
                    APw = (A * P) * wq[None, :]
                    t = fnp.matmul(AAw, P.T) + 2.0 * fnp.matmul(APw, A.T)
                    D21 = t if D21 is None else D21 + t
            D3_raw, D21_raw = D3, D21  # 源が表す値そのもの(倍率を掛ける前)
            if self.tau_scale != 1.0:
                D3 = D3 * self.tau_scale
                if D21 is not None:
                    D21 = D21 * self.tau_scale
            tdiag = D3 / (sigma * var)  # τ_iii = T_iii / σ³
            omega = (self.omega_scale * _omega_g(layer) / n) if self.use_omega else 0.0

            # --- Hermite 係数 (p=1) ---
            top = max(kmax + 3, 6)
            he = _hermite_neg_alpha(alpha, top)
            c = [m * Phi + sigma * phi, sigma * Phi]
            for k in range(2, top + 1):
                c.append(sigma * phi * he[k - 2])

            # --- 平均(単一頂点の 3 本ブロック補正) ---
            mu_new = (
                c[0] + c[3] * tdiag / 6.0 + c[6] * tdiag * tdiag / 72.0
                + c[4] * (omega / 24.0)
            )
            # --- 分散: E[h²] を p=2 係数で補正 ---
            e2_0 = (m * m + var) * Phi + m * sigma * phi
            e2_3 = 2.0 * var * phi
            e2_6 = 2.0 * var * phi * he[3]
            e2_4 = 2.0 * var * phi * he[1]
            ez2 = (
                e2_0 + e2_3 * tdiag / 6.0 + e2_6 * tdiag * tdiag / 72.0
                + e2_4 * (omega / 24.0)
            )
            var_post = fnp.maximum(ez2 - mu_new * mu_new, 0.0)
            if self.collect:
                self.features.append({
                    "m": m, "sigma": sigma, "alpha": alpha, "tdiag": tdiag,
                    "mu": mu_new, "mu_gauss": c[0], "var_post": var_post,
                })
            mu = mu_new
            rows.append(mu)
            if last:
                break

            # --- ReLU 後の共分散(Mehler + τ 1 次補正) ---
            rho = S / fnp.outer(sigma, sigma)
            if D21 is not None:
                t_iij = D21 / fnp.outer(var, sigma)  # T_iij / (σ_i² σ_j)
                t_ijj = t_iij.T
            else:
                t_iij = None
            R = [None, rho]  # R[k] = ρ^k / k!
            rp = rho
            for k in range(2, kmax + 1):
                rp = rp * rho
                R.append(rp / float(math.factorial(k)))
            Cn = fnp.multiply(fnp.outer(c[1], c[1]), R[1])
            for k in range(2, kmax + 1):
                Cn = Cn + fnp.multiply(fnp.outer(c[k], c[k]), R[k])
            if t_iij is not None:
                for k in range(2, kmax + 1):
                    Rk2 = fnp.ones_like(rho) if k == 2 else R[k - 2]
                    Cn = Cn + fnp.multiply(
                        fnp.outer(c[k], c[k - 1]) * t_iij
                        + fnp.outer(c[k - 1], c[k]) * t_ijj,
                        Rk2,
                    ) * 0.5
            # τ_iii による項(単一頂点ブロック、k=1..kmax)
            for k in range(1, kmax + 1):
                Cn = Cn + (
                    fnp.outer(c[k + 3] * tdiag, c[k])
                    + fnp.outer(c[k], c[k + 3] * tdiag)
                ) * R[k] / 6.0
            fnp.fill_diagonal(Cn, var_post)
            C = flops.as_symmetric(Cn, symmetry=(0, 1))

            # --- 計画1: 同一添字の 3 次切り口を正しい値にする補正 Δ ---
            #   u[i,k] = κ[h_i,h_i,h_k] = Cov(h_i², h_k) - 2 μ_i C_ik   (i≠k)
            #   u[i,i] = κ3[h_i] = E[h³] - 3 μ E[h²] + 2 μ³
            # 源が既に表している値 U_src(誕生項 + 輸送項)との差 Δ = u - U_src を、
            # A = I, P = Δᵀ(対角は 1/3)の源として次の層へ運ぶ。
            delta = None
            if win_iik > 0:
                # p=2(h²)の Hermite 係数: k=0..kmax+3
                c2p = [e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]
                for k in range(3, kmax + 4):
                    c2p.append(2.0 * var * phi * he[k - 3])
                cov_sq = fnp.multiply(fnp.outer(c2p[1], c[1]), R[1])
                for k in range(2, kmax + 1):
                    cov_sq = cov_sq + fnp.multiply(fnp.outer(c2p[k], c[k]), R[k])
                if t_iij is not None:
                    for k in range(2, kmax + 1):
                        Rk2 = fnp.ones_like(rho) if k == 2 else R[k - 2]
                        cov_sq = cov_sq + fnp.multiply(
                            fnp.outer(c2p[k], c[k - 1]) * t_iij
                            + fnp.outer(c2p[k - 1], c[k]) * t_ijj,
                            Rk2,
                        ) * 0.5
                for k in range(1, kmax + 1):
                    cov_sq = cov_sq + (
                        fnp.outer(c2p[k + 3] * tdiag, c[k])
                        + fnp.outer(c2p[k], c[k + 3] * tdiag)
                    ) * R[k] / 6.0
                u = cov_sq - 2.0 * mu[:, None] * Cn
                # E[h³](τ 補正込み)と 3 次キュムラントの対角
                e3_0 = m * e2_0 + 2.0 * var * c[0]
                e3_3 = 6.0 * var * sigma * Phi
                e3_6 = 6.0 * var * sigma * phi * he[2]
                ez3 = e3_0 + e3_3 * tdiag / 6.0 + e3_6 * tdiag * tdiag / 72.0
                k3d = ez3 - 3.0 * mu * ez2 + 2.0 * mu * mu * mu
                # 源が表している値(誕生: 2 c1_i c2_i c1_k ρ_ik + c1_i² c2_k ρ_ik²、
                # 輸送: Φ_i² Φ_k T_iik)
                c1c2 = c[1] * c[2]
                u_src = (
                    2.0 * fnp.outer(c1c2, c[1]) * rho
                    + fnp.outer(c[1] * c[1], c[2]) * (rho * rho)
                )
                k3_src = 3.0 * c[2] * c[1] * c[1]
                if D21_raw is not None:
                    u_src = u_src + fnp.outer(Phi * Phi, Phi) * D21_raw
                    k3_src = k3_src + Phi * Phi * Phi * D3_raw
                delta = u - u_src
                fnp.fill_diagonal(delta, (k3d - k3_src) / 3.0)

            # --- 源の輸送(次層の重みと ReLU の傾き gain=Φ) ---
            w_next = mlp.weights[layer + 1]
            wT = w_next.T
            moved = []
            for A, P, wq, kind, age in sources:
                win = win_rho if kind == "rho" else win_iik
                if age + 1 >= win:
                    continue  # 窓の外に出る源は運ばずに捨てる
                moved.append([
                    fnp.matmul(wT, Phi[:, None] * A),
                    fnp.matmul(wT, Phi[:, None] * P),
                    wq,
                    kind,
                    age + 1,
                ])
            # --- 源の誕生: A = Wᵀ diag(c1) ρ, P = Wᵀ, w = c_2 ---
            if win_rho > 0:
                moved.append([fnp.matmul(wT, c[1][:, None] * rho), wT, c[2], "rho", 0])
            # --- 計画1 の補正源: A = Wᵀ, P = Wᵀ Δᵀ, w = 1 ---
            if delta is not None:
                moved.append([wT, fnp.matmul(wT, delta.T), fnp.ones_like(c[2]), "iik", 0])
            sources = moved

        if self.calib:
            rows[-1] = rows[-1] * (1.0 - self.calib)
        return fnp.stack(rows, axis=0)
