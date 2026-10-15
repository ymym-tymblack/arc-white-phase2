"""因子化 K=3 キュムラント伝播(源を「家族」にまとめた版)。提出用、flopscope のみ。

前の版(research/archive/k3_factored_v1.py)から変えた点。いずれも ARC の公式参照実装(mlp_kprop, K=3 SIMPLE)と
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

directional_k4=True のときは G=diag(g)+λ C_off、κ4=(1/6)Sym6(G,I) を使う。
g は raw postactivation の K4/K22 行和から求め、次層の対角を正確に輸送する。
再生するのは非対角だけで、postactivation 分散の逆数による射影はしない。
κ4→κ3 の Sym3(K⊗u) は同じ A/M の読み出しへ折り込む(古い共有基底の家族も同様)。
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


def _estimate_flops(n: int, depth: int, window: int, rank: int,
                    directional: bool = False) -> float:
    """解析的な FLOP 見積り(行列積 1 回 = 2 n³)。少し多めに見積もる。

    層ごと: 共分散のサンドイッチ 3.22e9 (n/1024)³ + 誕生 2 n³ + 補正の低ランク化 ≈ 12 n² r
    家族 1 つ・1 層あたり: 8 n³(輸送 2 + 読み出し 2)+ 細い脚 ≈ 10 n² r
    """
    n3 = float(n) ** 3
    base = 3.22e9 * (n / 1024.0) ** 3 + 2.0 * n3 + 12.0 * n * n * rank
    total = 0.0
    for layer in range(depth):
        total += base + (8.0 * n3 + 10.0 * n * n * rank) * min(layer, window)
        if directional:
            # 追加の (2,2)/(3,1) ペア展開、raw power、家族ごとの要素積を余裕を持って数える。
            total += (1000.0 + 50.0 * layer) * n * n
    return total


def _fit_window(n: int, depth: int, budget: int, window: int, rank: int,
                margin: float = 0.85, directional: bool = False) -> int:
    """予算を超えないよう窓を縮める(超えるとその MLP の予測はゼロになり、割引もない)。"""
    while window > 0 and _estimate_flops(n, depth, window, rank, directional) > margin * budget:
        window -= 1
    return window


def _pair_terms(cA, cB, R, kmax, t_iij, t_ijj, tdiag, k4, k31=None):
    """Cov[gA(z_i), gB(z_j)](i≠j)を Hermite 係数で展開した和。

    ρ の辺だけ: Σ_k cA_k cB_k ρ^k/k!
    3 次ブロック 1 個: (cA_k cB_{k-1} τ_iij + cA_{k-1} cB_k τ_ijj) ρ^{k-2}/(2 (k-2)!)
                      + (cA_{k+3} τ_iii cB_k + cA_k cB_{k+3} τ_jjj) ρ^k/(6 k!)
    κ4 放射ブロック 1 個: cA_{2+e} cB_{2+e} q_i q_j ρ^e/(4 e!) + (cA_{4+e} ω_i cB_e + 対称) ρ^e/(24 e!)
    任意の k31[i,j]=κ̃_iiij は (cA[e+3] cB[e+1] k31 + 転置側) ρ^e/(6 e!) を加える。
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
    if k31 is not None:
        for e in range(kmax + 1):
            out = out + (
                fnp.outer(cA[e + 3], cB[e + 1]) * k31
                + fnp.outer(cA[e + 1], cB[e + 3]) * k31.T
            ) * (R[e] / 6.0)
    return out


def _relu_power_coefficients(m, var, sigma, Phi, phi, he, power, top):
    """ReLU の raw power の Hermite 係数。相殺しやすいベクトルだけ倍精度にする。"""
    m, var, sigma, Phi, phi = (
        x.astype(fnp.float64) for x in (m, var, sigma, Phi, phi)
    )
    moments = [Phi, m * Phi + sigma * phi]
    for p in range(2, power + 1):
        moments.append(m * moments[-1] + float(p - 1) * var * moments[-2])
    coeff = [moments[power]]
    for k in range(1, top + 1):
        if k <= power:
            coeff.append(
                float(math.factorial(power) // math.factorial(power - k))
                * sigma ** k * moments[power - k]
            )
        else:
            coeff.append(float(math.factorial(power)) * sigma ** power * phi * he[k - power - 1])
    return coeff


class Estimator(BaseEstimator):
    kmax = 3  # Mehler 展開の次数(共分散)
    window = 16  # 家族を何層分持つか(古い家族を捨てて FLOP を減らす)
    iik_rank = 16  # 補正 Δ の残差の低ランク近似の階数(None なら補正しない)
    use_k4 = True  # κ4 の放射部分を共分散・同一添字切り口に入れる
    omega_scale = 1.0  # ω の全体倍率(理論値 1)
    tau_scale = 1.0  # 3 次キュムラントの倍率(理論値 1)
    calib = 0.0  # 最終層の出力に掛ける縮小係数(正解 ≈ (1 - calib) × 予測)
    collect = False  # True なら層ごとの内部量を self.features に残す(解析用)
    # 古い家族の共有基底化: 年齢 age_old に達した家族は、全員で共有する n×rank_old の基底 U に
    # 射影して持つ(M ≈ U F)。輸送は U だけを運べばよく(X = U F なら T X = (T U) F で厳密)、
    # 家族ごとの n³ の行列積が n² r に減る。None なら使わない。
    # mini で探索(0〜11 番)し検証(12〜39 番): 基底化なし 1.540e-8 → age 6・階数 256・作り直しなし 1.382e-8。
    # 作り直す版より、最初の基底を運び続ける版の方が安く(C/B -0.006)誤差も小さかった。
    age_old = 6
    rank_old = 256
    rebuild_old = False  # True: 家族が入るたびに基底を作り直す / False: 最初の基底を運び続ける
    # κ4 → 3 次の誕生。どちらも T_ijk = K_ij c2_k + K_ik c2_j + K_jk c2_i(K は対称、対角 0)の形:
    #  (a) (2,1,1) 型 κ̃4[z_i,z_j,z_k,z_k] ≈ λ ρ_ij、λ = beta_k4 · ω: K += (λ/2) c1 ρ_off c1
    #  (b) 放射部分 κ̃4_iijj = ω/3 と ρ の辺 1 本の経路: K += (ω/12)(c3 ρ_off c1 + c1 ρ_off c3)
    beta_k4 = 0.0
    k4_birth_rad = False
    directional_k4 = False  # raw harmonic core の対角輸送と共分散方向の再生(旧誕生項とは排他的)
    regen_p0 = regen_p1 = regen_p2 = regen_p3 = 0.8
    regen_feed = 1.0
    regen_use31 = 1.0
    regen_start = 1
    regen_end = 16
    # 3 次 (2,1) 切り口 τ と ρ の辺 1 本の経路による誕生(参照実装 AUGMENT の k3 脚)。
    # 中心 q の脚 X = M B、B_iq = (c2_i c2_q τ_iiq + c1_i c3_q τ_iqq)/2。
    # B は低ランクに近い(上位 64 方向で 70〜97%)ので B ≈ Q R(階数 k3leg_rank)と持ち、
    # 運ぶのは n×r の Xq = M Q だけにする(X = Xq R)。
    # 検証用 28 MLP: なし 1.382e-8 → 階数 16 で 1.313e-8(生 MSE 3.5e-8 → 3.1e-8、C/B 0.394 → 0.425)。
    k3leg = True
    k3leg_rank = 16

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
        k3leg = self.k3leg
        directional = self.directional_k4
        regen = directional and self.use_k4
        if depth > 24:
            # とても深い MLP では residual 時間(上限 0.4 s)が層数に比例して増えるので、
            # 3 次キュムラントを持たない共分散伝播(+ κ4 放射)だけにする。
            k3leg = False
            window = 0
        elif depth > 16:
            # 競技の形状(1024×16)以外で深い MLP(採点前のスモークテストなど)では、
            # FLOP と residual 時間の上限から離れるよう、軽い設定に落とす。
            k3leg = False
            window = _fit_window(n, depth, budget, min(8, depth), rank, margin=0.6,
                                 directional=regen)
        else:
            window = _fit_window(n, depth, budget, depth if self.window is None else self.window,
                                 rank, directional=regen)
        f32 = fnp.float32

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        core_diag = fnp.zeros(n, dtype=f32) if regen else None
        var_prev = fnp.ones(n, dtype=f32) if regen else None
        # 家族は「横並び」の行列で持つ: 家族 f の脚が列ブロック [f n, (f+1) n) を占める。
        # こうすると、輸送も読み出しも家族の数によらず 1 回の演算で済む(Python 側の時間を抑える)。
        nf = 0  # 生きている家族の数(古い順に左から並ぶ)
        A_w = M_w = c2_w = e_w = dd_w = Z_w = Q_s = None
        c1_w = g_w = None  # κ4 誕生項: 家族ごとの c1 と X=(A-M*c1)*g の固定列スケール
        u_w = None  # directional 誕生の固定脚 d*c2/var(旧経路は c2 を使う)
        c1_o = g_o = u_o = None  # 共有基底でも X を Ao/Mo と同じ列スケールから再構成
        X3q_w = R3_s = None  # k3 脚の誕生項: X = Xq R(家族ごと)
        FX3_o = None  # 古い層: X = U FX3
        r3 = self.k3leg_rank
        FX_o = FU_o = None  # 古い層の κ4 誕生項: X = U FX、u = U FU
        # 古い層(共有基底): U は n×r、FA/FM/FP は r×(家族数·n)、A = U FA, M = U FM, PΔ = U FP
        U = FA_o = FM_o = FP_o = c2_o = None
        r_old = self.rank_old
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
            lam = 0.0
            dG = metric = None
            if regen:
                WW = w * w
                metric = fnp.sum(WW, axis=0)
                tg = fnp.matmul(WW.T, core_diag)
                tv = fnp.diag(S) - fnp.matmul(WW.T, var_prev)
                strength = 0.0
                if self.regen_start <= layer <= self.regen_end:
                    position = 3.0 * layer / max(depth - 1, 1)
                    knot = min(int(position), 2)
                    fraction = position - knot
                    controls = (self.regen_p0, self.regen_p1, self.regen_p2, self.regen_p3)
                    strength = controls[knot] * (1.0 - fraction) + controls[knot + 1] * fraction
                denominator = fnp.mean(var) - strength * fnp.mean(tv)
                if not float(denominator) > 0.0:
                    raise ValueError("directional_k4 regeneration denominator must be positive")
                lam = strength * fnp.mean(tg) / denominator
                dG = tg + lam * tv

            # --- 3 次キュムラントの読み出し(D3 = T_aaa, D21 = T_aab) ---
            D3 = fnp.zeros(n, dtype=f32)
            D21 = None
            if nf > 0:
                AA = A_w * A_w
                d3 = AA * M_w * c2_w[None, :]
                X3_w = None
                if X3q_w is not None:
                    X3_w = fnp.reshape(
                        fnp.einsum("nkr,krm->nkm", fnp.reshape(X3q_w, (n, nf, r3)), R3_s), (n, nf * n)
                    )
                    d3 = d3 + 2.0 * (X3_w * M_w * A_w)
                if e_w is not None:
                    MM = M_w * M_w
                    PD = A_w * e_w[None, :] + M_w * dd_w[None, :]
                    if Z_w is not None:
                        ZQ = fnp.einsum("nkr,kmr->nkm", fnp.reshape(Z_w, (n, nf, rank)), Q_s)
                        PD = PD + fnp.reshape(ZQ, (n, nf * n))
                    d3 = d3 + MM * PD
                D3 = 3.0 * fnp.sum(d3, axis=1)
                if regen and g_w is not None:
                    X_w = (A_w - M_w * c1_w[None, :]) * g_w[None, :]
                    M3 = fnp.reshape(M_w, (n, nf, n))
                    kd = 2.0 * fnp.sum(fnp.reshape(X_w * M_w, (n, nf, n)), axis=2)
                    uu = fnp.einsum("nfm,fm->nf", M3, fnp.reshape(u_w, (nf, n)))
                    D3 = D3 + 3.0 * fnp.sum(kd * uu, axis=1)
                if not last:
                    XM = AA * c2_w[None, :]
                    XA = 2.0 * (A_w * M_w) * c2_w[None, :]
                    if e_w is not None:
                        XM = XM + MM * dd_w[None, :] + 2.0 * (M_w * PD)
                        XA = XA + MM * e_w[None, :]
                    if X3_w is not None:
                        XM = XM + 2.0 * (X3_w * A_w)
                        XA = XA + 2.0 * (X3_w * M_w)
                    if regen and g_w is not None:
                        # X の転置側も A/M の既存 2 積へ折り込み、家族ごとの n³ 積を増やさない。
                        fold_m = A_w * g_w[None, :] - 2.0 * M_w * (c1_w * g_w)[None, :]
                        XM = XM + fnp.reshape(
                            2.0 * uu[:, :, None] * fnp.reshape(fold_m, (n, nf, n)), (n, nf * n)
                        )
                        XA = XA + fnp.reshape(
                            2.0 * uu[:, :, None] * fnp.reshape(M_w * g_w[None, :], (n, nf, n)),
                            (n, nf * n),
                        )
                    D21 = fnp.matmul(XM, M_w.T) + fnp.matmul(XA, A_w.T)
                    if regen and g_w is not None:
                        D21 = D21 + fnp.matmul(kd, uu.T)
                    if X3_w is not None:
                        Y3 = fnp.einsum("nkm,krm->nkr", fnp.reshape(M_w * A_w, (n, nf, n)), R3_s)
                        D21 = D21 + 2.0 * fnp.matmul(fnp.reshape(Y3, (n, nf * r3)), X3q_w.T)
                    if Z_w is not None:
                        Y = fnp.einsum("nkm,kmr->nkr", fnp.reshape(MM, (n, nf, n)), Q_s)
                        D21 = D21 + fnp.matmul(fnp.reshape(Y, (n, nf * rank)), Z_w.T)
                if g_w is not None and not directional:
                    # κ4 誕生項: T = Sym3(K ⊗ u)、K = X Mᵀ(X = (A - M c1) (λ/2) c1)、u = M c2
                    # 輸送後 K = Y + Yᵀ、Y = X Mᵀ、X = (A - M c1) g、u = M c2
                    X_w = (A_w - M_w * c1_w[None, :]) * g_w[None, :]
                    X3 = fnp.reshape(X_w, (n, nf, n))
                    M3 = fnp.reshape(M_w, (n, nf, n))
                    kd = 2.0 * fnp.sum(X3 * M3, axis=2)  # (n, nf): K の対角
                    uu = fnp.einsum("anq,nq->an", M3, fnp.reshape(c2_w, (nf, n)))  # (n, nf)
                    D3 = D3 + 3.0 * fnp.sum(kd * uu, axis=1)
                    if not last:
                        Xu = fnp.reshape(X3 * uu[:, :, None], (n, nf * n))
                        Mu = fnp.reshape(M3 * uu[:, :, None], (n, nf * n))
                        D21 = D21 + fnp.matmul(kd, uu.T) + 2.0 * (
                            fnp.matmul(Xu, M_w.T) + fnp.matmul(Mu, X_w.T)
                        )
            if U is not None:
                # 古い層の読み出し: 密な脚を U から作り、D21 は r 次元で足し合わせてから U ᵀ を掛ける
                Ao = fnp.matmul(U, FA_o)
                Mo = fnp.matmul(U, FM_o)
                Po = fnp.matmul(U, FP_o)
                AAo = Ao * Ao
                MMo = Mo * Mo
                D3 = D3 + 3.0 * fnp.sum(AAo * Mo * c2_o[None, :] + MMo * Po, axis=1)
                X3o = None
                if FX3_o is not None:
                    X3o = fnp.matmul(U, FX3_o)
                    D3 = D3 + 6.0 * fnp.sum(X3o * Mo * Ao, axis=1)
                if regen and g_o is not None:
                    ko = g_o.shape[0] // n
                    Xo = (Ao - Mo * c1_o[None, :]) * g_o[None, :]
                    Mo3 = fnp.reshape(Mo, (n, ko, n))
                    kdo = 2.0 * fnp.sum(fnp.reshape(Xo * Mo, (n, ko, n)), axis=2)
                    uo = fnp.einsum("nfm,fm->nf", Mo3, fnp.reshape(u_o, (ko, n)))
                    D3 = D3 + 3.0 * fnp.sum(kdo * uo, axis=1)
                if FX_o is not None and last:
                    ko = FX_o.shape[1] // n
                    Xo3 = fnp.reshape(fnp.matmul(U, FX_o), (n, ko, n))
                    uo = fnp.matmul(U, FU_o)
                    kdo = 2.0 * fnp.sum(Xo3 * fnp.reshape(Mo, (n, ko, n)), axis=2)
                    D3 = D3 + 3.0 * fnp.sum(kdo * uo, axis=1)
                if not last:
                    XMo = AAo * c2_o[None, :] + 2.0 * (Mo * Po)
                    XAo = 2.0 * (Ao * Mo) * c2_o[None, :]
                    if X3o is not None:
                        XMo = XMo + 2.0 * (X3o * Ao)
                        XAo = XAo + 2.0 * (X3o * Mo)
                    if regen and g_o is not None:
                        fold_m = Ao * g_o[None, :] - 2.0 * Mo * (c1_o * g_o)[None, :]
                        XMo = XMo + fnp.reshape(
                            2.0 * uo[:, :, None] * fnp.reshape(fold_m, (n, ko, n)), (n, ko * n)
                        )
                        XAo = XAo + fnp.reshape(
                            2.0 * uo[:, :, None] * fnp.reshape(Mo * g_o[None, :], (n, ko, n)),
                            (n, ko * n),
                        )
                        thin = fnp.matmul(kdo, uo.T)
                        D21 = thin if D21 is None else D21 + thin
                    Racc = (
                        fnp.matmul(XMo, FM_o.T)
                        + fnp.matmul(XAo, FA_o.T)
                        + fnp.matmul(MMo, FP_o.T)
                    )
                    if X3o is not None:
                        Racc = Racc + 2.0 * fnp.matmul(Mo * Ao, FX3_o.T)
                    if FX_o is not None:
                        ko = FX_o.shape[1] // n
                        Xo3 = fnp.reshape(fnp.matmul(U, FX_o), (n, ko, n))
                        uo = fnp.matmul(U, FU_o)  # (n, ko)
                        Mo3 = fnp.reshape(Mo, (n, ko, n))
                        kdo = 2.0 * fnp.sum(Xo3 * Mo3, axis=2)
                        D3 = D3 + 3.0 * fnp.sum(kdo * uo, axis=1)
                        D21 = (fnp.matmul(kdo, uo.T) if D21 is None else D21 + fnp.matmul(kdo, uo.T))
                        Racc = Racc + 2.0 * (
                            fnp.matmul(fnp.reshape(Xo3 * uo[:, :, None], (n, ko * n)), FM_o.T)
                            + fnp.matmul(fnp.reshape(Mo3 * uo[:, :, None], (n, ko * n)), FX_o.T)
                        )
                    t_old = fnp.matmul(Racc, U.T)
                    D21 = t_old if D21 is None else D21 + t_old
            D3_raw, D21_raw = D3, D21  # 家族が表す値そのもの(倍率を掛ける前)
            if self.tau_scale != 1.0:
                D3 = D3 * self.tau_scale
                if D21 is not None:
                    D21 = D21 * self.tau_scale
            tdiag = D3 / (sigma * var)  # τ_iii = T_iii / σ³
            if directional:
                omega = metric * dG / (var * var) if regen else 0.0
            else:
                omega = self.omega_scale * _omega_g(layer) / n

            # --- Hermite 係数 (p=1) ---
            top = max(kmax + 4, 7)
            he = _hermite_neg_alpha(alpha, top)
            c = [m * Phi + sigma * phi, sigma * Phi]
            for k in range(2, top + 1):
                c.append(sigma * phi * he[k - 2])

            # --- 平均と E[h²](単一頂点の 3 次・4 次ブロック補正) ---
            e2_0 = (m * m + var) * Phi + m * sigma * phi
            raw_mu = raw_ez2 = raw_ez3 = raw_ez4 = raw_var = None
            if regen:
                raw_powers = []
                for power in range(1, 3 if last else 5):
                    coeff = _relu_power_coefficients(m, var, sigma, Phi, phi, he, power, 6)
                    raw_powers.append(
                        coeff[0] + coeff[3] * tdiag / 6.0
                        + coeff[6] * tdiag * tdiag / 72.0 + coeff[4] * omega / 24.0
                    )
                raw_mu, raw_ez2 = raw_powers[:2]
                mu_new = raw_mu.astype(f32)
                ez2 = raw_ez2.astype(f32)
                if not last:
                    raw_ez3, raw_ez4 = raw_powers[2:]
                raw_var = fnp.maximum(raw_ez2 - raw_mu * raw_mu, 0.0)
                var_post = raw_var.astype(f32)
            else:
                mu_new = c[0] + c[3] * tdiag / 6.0 + c[6] * tdiag * tdiag / 72.0 + c[4] * (omega / 24.0)
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
                if regen:
                    self.features[-1].update({
                        "k4_lambda": lam, "k4_gdiag": dG, "k4_metric": metric, "k4_omega": omega,
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
            k4 = k31 = None
            if regen:
                dv, gv = metric / var, dG / var
                k22 = (fnp.outer(dv, gv) + fnp.outer(gv, dv)) / 6.0
                fnp.fill_diagonal(k22, 0.0)
                k4 = (omega, k22)
                k31 = (self.regen_use31 * 0.5 * lam) * dv[:, None] * rho
                fnp.fill_diagonal(k31, 0.0)
            elif self.use_k4 and not directional:
                om = omega * fnp.ones_like(sigma)
                q = fnp.sqrt(om / 3.0)
                k4 = (om, fnp.outer(q, q))
            Cn = _pair_terms(c, c, R, kmax, t_iij, t_ijj, tdiag, k4, k31)
            fnp.fill_diagonal(Cn, var_post)
            C = flops.as_symmetric(Cn, symmetry=(0, 1))
            c2p = cov21 = None
            if regen:
                c2p = [e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]
                for kk in range(3, kmax + 5):
                    c2p.append(2.0 * var * phi * he[kk - 3])
                cov21 = _pair_terms(c2p, c, R, kmax, t_iij, t_ijj, tdiag, k4, k31)
                cov22 = _pair_terms(c2p, c2p, R, kmax, t_iij, t_ijj, tdiag, k4, k31)
                post_k22 = (
                    cov22 - 2.0 * mu[:, None] * cov21.T - 2.0 * mu[None, :] * cov21
                    + 4.0 * fnp.outer(mu, mu) * Cn - 2.0 * Cn * Cn
                )
                fnp.fill_diagonal(post_k22, 0.0)
                post_k4 = (
                    raw_ez4 - 4.0 * raw_mu * raw_ez3 + 6.0 * raw_mu * raw_mu * raw_ez2
                    - 3.0 * raw_mu ** 4 - 3.0 * raw_var ** 2
                )
                H = post_k4 + fnp.sum(post_k22, axis=1)
                core_diag = (
                    (6.0 / (n + 4)) * H
                    - (3.0 / ((n + 2) * (n + 4))) * fnp.sum(H)
                ).astype(f32)
                var_prev = var_post
                if self.collect:
                    self.features[-1].update({
                        "k4_post_core_diag": core_diag, "k4_post_K22": post_k22,
                        "k4_post_K4": post_k4,
                    })

            # --- 同一添字の 3 次切り口の補正 Δ ---
            #   u[i,k] = κ[h_i,h_i,h_k] = Cov(h_i², h_k) - 2 μ_i C_ik   (i≠k)
            #   κ3[h_i] = E[h³] - 3 μ E[h²] + 2 μ³
            # 家族が既に表している値(誕生: 2 c1_i c2_i c1_k ρ_ik + c1_i² c2_k ρ_ik²、
            # 輸送: Φ_i² Φ_k T_iik)との差を Δ として、新しい家族に入れる。
            w_next = mlp.weights[layer + 1]
            wT = w_next.T
            A_new = fnp.matmul(wT, c[1][:, None] * rho)
            B3 = X3q_new = R3_new = None
            if k3leg:
                if t_iij is not None:
                    B3 = 0.5 * (fnp.outer(c[2], c[2]) * t_iij + fnp.outer(c[1], c[3]) * t_ijj)
                    fnp.fill_diagonal(B3, 0.0)
                    Yq = fnp.matmul(B3, w_next[:, :r3])
                    Yq = fnp.matmul(B3, fnp.matmul(B3.T, Yq))
                    Q3, _ = fnp.linalg.qr(Yq)
                    R3_new = fnp.matmul(Q3.T, B3)  # (r3, n)
                    B3 = fnp.matmul(Q3, R3_new)
                    X3q_new = fnp.matmul(wT, Q3)
                else:
                    X3q_new = fnp.zeros((n, r3), dtype=f32)
                    R3_new = fnp.zeros((r3, n), dtype=f32)
            # κ4 誕生項の列スケール g(K = (c1 ρ_off g + 転置) の形)
            g_new = u_new = None
            if regen:
                g_new = (lam * self.regen_feed / 24.0) * c[1]
                u_new = metric * c[2] / var
            elif not directional and (self.beta_k4 or self.k4_birth_rad):
                g_new = (0.25 * self.beta_k4 * omega) * c[1]
                if self.k4_birth_rad:
                    g_new = g_new + (omega / 12.0) * c[3]
            e_new = dd_new = Z_new = Q_new = None
            if self.iik_rank is not None and window > 0:
                if c2p is None:
                    c2p = [e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]
                    for kk in range(3, kmax + 5):
                        c2p.append(2.0 * var * phi * he[kk - 3])
                if cov21 is None:
                    cov21 = _pair_terms(c2p, c, R, kmax, t_iij, t_ijj, tdiag, k4, k31)
                u = cov21 - 2.0 * mu[:, None] * Cn
                if regen:
                    k3d = (raw_ez3 - 3.0 * raw_mu * raw_ez2 + 2.0 * raw_mu ** 3).astype(f32)
                else:
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
                if B3 is not None:
                    # k3 脚の誕生項の同一添字切り口: T_iik = 2 B_ik E_ik + 2 c1_i B_ki(E = c1 ρ)
                    # (B は低ランク近似したもの = 源が実際に表している値)
                    u_src = u_src + 2.0 * B3 * (c[1][:, None] * rho) + 2.0 * c[1][:, None] * B3.T
                if g_new is not None:
                    # κ4 誕生項の同一添字切り口: T_iik = 2 K_ik u_i(K_ii = 0)
                    Kb = fnp.outer(c[1], g_new) * rho
                    Kb = Kb + Kb.T
                    fnp.fill_diagonal(Kb, 0.0)
                    u_src = u_src + 2.0 * (u_new if regen else c[2])[:, None] * Kb
                if D21_raw is not None:
                    u_src = u_src + fnp.outer(Phi * Phi, Phi) * D21_raw
                    k3_src = k3_src + Phi * Phi * Phi * D3_raw
                delta = u - u_src
                fnp.fill_diagonal(delta, 0.0)
                # 主要項: 行 i ごとに base_ik = c1_k ρ_ik の倍数 e_i で近似(最小二乗)
                base = rho * c[1][None, :]
                fnp.fill_diagonal(base, 0.0)
                e_new = fnp.sum(delta * base, axis=1) / fnp.maximum(fnp.sum(base * base, axis=1), 1e-30)
                dd_new = (k3d - k3_src) / 3.0 - c[1] * e_new
                if rank > 0:
                    # 残差の低ランク近似(乱択レンジファインダー、べき乗 1 回)。
                    # 試行行列は乱数でなく次層の重みの先頭 r 列(決定的)。
                    resid = delta - e_new[:, None] * base
                    Yr = fnp.matmul(resid, w_next[:, :rank])
                    Yr = fnp.matmul(resid, fnp.matmul(resid.T, Yr))
                    Q_new, _ = fnp.linalg.qr(Yr)
                    Z_new = fnp.matmul(wT, fnp.matmul(Q_new.T, resid).T)  # (n, r)

            # --- 家族の輸送(次層の重みと ReLU の傾き Φ)と誕生 ---
            if nf > 0 and nf >= window:
                # 最も古い家族(左端のブロック)は窓の外に出るので運ばずに捨てる
                A_w, M_w, c2_w = A_w[:, n:], M_w[:, n:], c2_w[n:]
                if X3q_w is not None:
                    X3q_w, R3_s = X3q_w[:, r3:], R3_s[1:]
                if e_w is not None:
                    e_w, dd_w = e_w[n:], dd_w[n:]
                if Z_w is not None:
                    Z_w, Q_s = Z_w[:, rank:], Q_s[1:]
                if g_w is not None:
                    c1_w, g_w = c1_w[n:], g_w[n:]
                    if regen:
                        u_w = u_w[n:]
                nf -= 1
            if U is not None:
                U = fnp.matmul(wT, Phi[:, None] * U)
            if nf > 0:
                A_w = fnp.matmul(wT, Phi[:, None] * A_w)
                if X3q_w is not None:
                    X3q_w = fnp.matmul(wT, Phi[:, None] * X3q_w)
                M_w = fnp.matmul(wT, Phi[:, None] * M_w)
                if Z_w is not None:
                    Z_w = fnp.matmul(wT, Phi[:, None] * Z_w)
            if window > 0:
                if g_new is not None:
                    if nf == 0 or g_w is None:
                        c1_w, g_w = c[1], g_new
                        if regen:
                            u_w = u_new
                    else:
                        c1_w = fnp.concatenate([c1_w, c[1]])
                        g_w = fnp.concatenate([g_w, g_new])
                        if regen:
                            u_w = fnp.concatenate([u_w, u_new])
                if nf == 0:
                    A_w, M_w, c2_w = A_new, fnp.asarray(wT, dtype=f32), c[2]
                    X3q_w = X3q_new
                    R3_s = None if R3_new is None else fnp.reshape(R3_new, (1, r3, n))
                    e_w, dd_w = e_new, dd_new
                    Z_w = Z_new
                    Q_s = None if Q_new is None else fnp.reshape(Q_new, (1, n, rank))
                else:
                    A_w = fnp.concatenate([A_w, A_new], axis=1)
                    if X3q_new is not None:
                        X3q_w = fnp.concatenate([X3q_w, X3q_new], axis=1)
                        R3_s = fnp.concatenate([R3_s, fnp.reshape(R3_new, (1, r3, n))], axis=0)
                    M_w = fnp.concatenate([M_w, wT], axis=1)
                    c2_w = fnp.concatenate([c2_w, c[2]])
                    if e_new is not None:
                        e_w = fnp.concatenate([e_w, e_new])
                        dd_w = fnp.concatenate([dd_w, dd_new])
                    if Z_new is not None:
                        Z_w = fnp.concatenate([Z_w, Z_new], axis=1)
                        Q_s = fnp.concatenate([Q_s, fnp.reshape(Q_new, (1, n, rank))], axis=0)
                nf += 1
            if self.age_old is not None and nf > self.age_old:
                # 最も古い若い家族(年齢 age_old)を共有基底へ移す
                A_f, M_f, c2_f = A_w[:, :n], M_w[:, :n], c2_w[:n]
                X3_f = None
                if X3q_w is not None:
                    X3_f = fnp.matmul(X3q_w[:, :r3], R3_s[0])
                    X3q_w, R3_s = X3q_w[:, r3:], R3_s[1:]
                X_f = None
                if g_w is not None:
                    if regen:
                        c1_f, g_f, u_f = c1_w[:n], g_w[:n], u_w[:n]
                        if g_o is None:
                            c1_o, g_o, u_o = c1_f, g_f, u_f
                        else:
                            c1_o = fnp.concatenate([c1_o, c1_f])
                            g_o = fnp.concatenate([g_o, g_f])
                            u_o = fnp.concatenate([u_o, u_f])
                        u_w = u_w[n:]
                    else:
                        X_f = (A_f - M_f * c1_w[:n][None, :]) * g_w[:n][None, :]
                    c1_w, g_w = c1_w[n:], g_w[n:]
                PD_f = M_f * dd_w[:n][None, :] if dd_w is not None else fnp.zeros((n, n), dtype=f32)
                if e_w is not None:
                    PD_f = PD_f + A_f * e_w[:n][None, :]
                if Z_w is not None:
                    PD_f = PD_f + fnp.matmul(Z_w[:, :rank], Q_s[0].T)
                A_w, M_w, c2_w = A_w[:, n:], M_w[:, n:], c2_w[n:]
                if e_w is not None:
                    e_w, dd_w = e_w[n:], dd_w[n:]
                if Z_w is not None:
                    Z_w, Q_s = Z_w[:, rank:], Q_s[1:]
                nf -= 1
                # 新しい基底: 移す家族の伝播脚 M の主要な列空間(乱択レンジファインダー、べき乗 1 回)
                if U is None or self.rebuild_old:
                    Yb = fnp.matmul(M_f, w_next[:, n - r_old:])
                    Yb = fnp.matmul(M_f, fnp.matmul(M_f.T, Yb))
                    U_new, _ = fnp.linalg.qr(Yb)
                else:
                    U_new, _ = fnp.linalg.qr(U)  # 運んだ基底を正規直交に戻すだけ
                if U is not None:
                    Tm = fnp.matmul(U_new.T, U)  # 既存の古い家族を新しい基底で表し直す
                    FA_o = fnp.matmul(Tm, FA_o)
                    FM_o = fnp.matmul(Tm, FM_o)
                    FP_o = fnp.matmul(Tm, FP_o)
                    if FX_o is not None:
                        FX_o = fnp.matmul(Tm, FX_o)
                        FU_o = fnp.matmul(Tm, FU_o)
                    if FX3_o is not None:
                        FX3_o = fnp.matmul(Tm, FX3_o)
                fa, fm, fp = (fnp.matmul(U_new.T, X) for X in (A_f, M_f, PD_f))
                if U is None:
                    FA_o, FM_o, FP_o, c2_o = fa, fm, fp, c2_f
                else:
                    FA_o = fnp.concatenate([FA_o, fa], axis=1)
                    FM_o = fnp.concatenate([FM_o, fm], axis=1)
                    FP_o = fnp.concatenate([FP_o, fp], axis=1)
                    c2_o = fnp.concatenate([c2_o, c2_f])
                if X3_f is not None:
                    fx3 = fnp.matmul(U_new.T, X3_f)
                    FX3_o = fx3 if FX3_o is None else fnp.concatenate([FX3_o, fx3], axis=1)
                if X_f is not None:
                    fx = fnp.matmul(U_new.T, X_f)
                    fu = fnp.matmul(U_new.T, fnp.matmul(M_f, c2_f))[:, None]
                    if FX_o is None:
                        FX_o, FU_o = fx, fu
                    else:
                        FX_o = fnp.concatenate([FX_o, fx], axis=1)
                        FU_o = fnp.concatenate([FU_o, fu], axis=1)
                U = U_new

        if self.calib:
            rows[-1] = rows[-1] * (1.0 - self.calib)
        return fnp.stack(rows, axis=0)
