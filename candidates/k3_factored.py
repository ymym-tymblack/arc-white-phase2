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


def _max_live_sources(n: int, depth: int, budget: int, margin: float = 0.85) -> int:
    """予算内に収まる「同時に保持する源の数」の上限を解析的な FLOP 見積りから決める。

    FLOP が予算を超えると、その MLP の予測はゼロに置き換わる(割引もなし)。
    競技の形状(1024×16)では上限に達しないが、深い MLP(スモークテストなど)では
    古い源から捨てて予算内に収める。
    見積り(行列積 1 回 = 2 n³):
      層ごとの基本コスト   ≈ 3.22e9 (n/1024)³ + 2 n³ (誕生)
      源 1 個・1 層あたり  ≈ 8 n³ (輸送 2 + 読み出し 2)
    """
    n3 = float(n) ** 3
    base = 3.22e9 * (n / 1024.0) ** 3 + 2.0 * n3
    for cap in range(depth, -1, -1):
        total = 0.0
        for layer in range(depth):
            live = min(layer, cap)
            total += base + 8.0 * n3 * live
        if total <= margin * budget:
            return cap
    return 0


class Estimator(BaseEstimator):
    kmax = 3  # Mehler 展開の次数(共分散)
    use_omega = True  # κ4 対角(超過尖度)の平均・分散補正
    # 段階4: Optuna(TPE)で mini の 12 MLP 上を 30 試行探索し、別の 28 MLP で検証した値。
    #   検証 28 MLP の調整後スコア: 既定(window=16, 倍率1.0) 1.34e-7 → 下記 9.3e-8
    # 倍率は理論値が 1 だが、τ は 1.14 付近が最良だった(古い源を捨てた分の補償を含む
    # 可能性があり、MLP 形状が変われば再調整が必要)。
    omega_scale = 1.05  # ω の全体倍率(理論値 1)
    window = 7  # 同時に保持する源の上限(古い源を捨てて FLOP を減らす)
    tau_scale = 1.14  # 3 次キュムラント(D3, D21)の倍率(理論値 1)
    # 段階5: 最終層の出力に掛ける縮小係数。mini 100 MLP で 正解 ≈ (1 - 2.37e-4) × 予測
    # (予測値と残差の相関 -0.31、1 パラメータ補正で 5 分割 CV の MSE が 0.825 倍)。
    # 9 特徴量の ridge(0.816)・GBM(0.83〜0.85)は 1 パラメータ補正を超えなかった。
    calib = 2.37e-4
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
        self.features = []

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = mlp.depth
        max_live = _max_live_sources(n, depth, budget)
        if self.window is not None:
            max_live = min(max_live, self.window)
        kmax = self.kmax
        f32 = fnp.float32

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        sources: list[list] = []  # [A, P, w]
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
            for A, P, wq in sources:
                AA = A * A
                D3 = D3 + 3.0 * fnp.sum(AA * P * wq[None, :], axis=1)
                if not last:
                    AAw = AA * wq[None, :]
                    APw = (A * P) * wq[None, :]
                    t = fnp.matmul(AAw, P.T) + 2.0 * fnp.matmul(APw, A.T)
                    D21 = t if D21 is None else D21 + t
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

            # --- 源の輸送(次層の重みと ReLU の傾き gain=Φ) ---
            w_next = mlp.weights[layer + 1]
            wT = w_next.T
            moved = []
            for A, P, wq in sources:
                moved.append([
                    fnp.matmul(wT, Phi[:, None] * A),
                    fnp.matmul(wT, Phi[:, None] * P),
                    wq,
                ])
            # --- 源の誕生: A = Wᵀ diag(c1) ρ, P = Wᵀ, w = c_2 ---
            moved.append([fnp.matmul(wT, c[1][:, None] * rho), wT, c[2]])
            sources = moved[-max_live:] if max_live else []

        if self.calib:
            rows[-1] = rows[-1] * (1.0 - self.calib)
        return fnp.stack(rows, axis=0)
