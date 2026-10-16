"""因子化κ3家族と方向依存κ4コアによる平均推定。提出用、flopscopeのみ、入力サンプリングなし。

ARCのキュムラント伝播と公開再生式を参照し、rawコアと正値周辺密度で閉包する。
数式・独立検算・実測比較は research/README.md に記録する。

1. κ4の対角コアを運び、非対角は共分散の方向に再生する。
   単変量モーメントは正値密度で計算し、κ4の追加誕生を既存のκ3家族に折り込む。
2. 同一添字の 3 次切り口 κ[h_i,h_i,h_k](と κ3[h_i])を正しい値に直す補正 Δ を、
   各層で生まれる源と同じ「家族」に入れて運ぶ。Δ は
       Δ ≈ diag(e) (ρ ∘ 1 c1ᵀ) + Q B(低ランク) + 対角
   と分けられ(主要項は誕生源の脚 A と同じ形)、別の密な脚を運ばずに済む。

基本の家族 f(層 b で誕生)の3次キュムラント(追加脚は別項として重ねる):
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

方向依存モードは G=diag(g)+λ C_off、κ4=(1/6)Sym6(G,I) を使う。
g は raw postactivation の K4/K22 行和から求め、次層の対角を正確に輸送する。
再生するのは非対角だけで、postactivation 分散の逆数による射影はしない。
κ4→κ3 の Sym3(K⊗u) は同じ A/M の読み出しへ折り込む(古い共有基底の家族も同様)。
単変量の raw moment は正の 12 次 Hermite 密度を解析積分し、中心化・標準化で入力平均と分散を保つ。
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
    # 旧1次元呼び出しは outer の演算・課金順をそのまま維持する。
    outer = fnp.outer if cA[1].ndim == cB[1].ndim == 1 else (
        lambda a, b: a[..., :, None] * b[..., None, :]
    )
    out = fnp.multiply(outer(cA[1], cB[1]), R[1])
    for k in range(2, kmax + 1):
        out = out + fnp.multiply(outer(cA[k], cB[k]), R[k])
    if t_iij is not None:
        for k in range(2, kmax + 1):
            Rk2 = R[0] if k == 2 else R[k - 2]
            out = out + fnp.multiply(
                outer(cA[k], cB[k - 1]) * t_iij + outer(cA[k - 1], cB[k]) * t_ijj, Rk2
            ) * 0.5
    if tdiag is not None:
        for k in range(1, kmax + 1):
            out = out + (outer(cA[k + 3] * tdiag, cB[k]) + outer(cA[k], cB[k + 3] * tdiag)) * (
                R[k] / 6.0
            )
    if k4 is not None:
        om, k22 = k4
        out = out + fnp.multiply(outer(cA[2], cB[2]), k22) * 0.25
        for e in range(1, kmax + 1):
            out = out + fnp.multiply(outer(cA[2 + e], cB[2 + e]), k22 * R[e]) * 0.25
            out = out + (outer(cA[4 + e] * om, cB[e]) + outer(cA[e], cB[4 + e] * om)) * (
                R[e] / 24.0
            )
    if k31 is not None:
        for e in range(kmax + 1):
            out = out + (
                outer(cA[e + 3], cB[e + 1]) * k31
                + outer(cA[e + 1], cB[e + 3]) * k31.T
            ) * (R[e] / 6.0)
    return out


def _positive_hermite_map(degrees):
    """固定の [1, a_i, a_i a_j] → He_0..He_12 線形写像。"""
    pairs = tuple((i, j) for i in range(len(degrees)) for j in range(i, len(degrees)))
    weights = [[0.0] * (1 + len(degrees) + len(pairs)) for _ in range(13)]
    weights[0][0] = 1.0
    for i, degree in enumerate(degrees):
        weights[degree][1 + i] = 1.0
    for column, (i, j) in enumerate(pairs, 1 + len(degrees)):
        di, dj = degrees[i], degrees[j]
        for r in range(min(di, dj) + 1):
            weights[di + dj - 2 * r][column] = (
                (0.5 if i == j else 1.0)
                * math.factorial(r) * math.comb(di, r) * math.comb(dj, r)
            )
    return tuple(i for i, _ in pairs), tuple(j for _, j in pairs), fnp.asarray(weights, dtype=fnp.float64)


_POSITIVE_HERMITE_MAP = _positive_hermite_map((3, 4, 5, 6))
_POSITIVE_EVEN_HERMITE_MAP = _positive_hermite_map((4, 6))


def _mills_polynomials(levels):
    """同じ Laplace 連分数を t=1/x² の正係数多項式比へ変換する。"""
    p0 = q0 = (1.0,)
    p1, q1 = (1.0,), (1.0, 1.0)
    for level in range(2, levels + 1):
        def advance(previous, previous2):
            return tuple(
                (previous[k] if k < len(previous) else 0.0)
                + (float(level) * previous2[k - 1] if 0 < k <= len(previous2) else 0.0)
                for k in range(max(len(previous), len(previous2) + 1))
            )
        p0, p1 = p1, advance(p1, p0)
        q0, q1 = q1, advance(q1, q0)
    return p1[::-1], q1[::-1]


_MILLS_P32, _MILLS_Q32 = (
    fnp.asarray(p, dtype=fnp.float64) for p in _mills_polynomials(32)
)


def _hermite_power_matrix(top):
    """He_k(x)を単項式基底で一括評価する固定係数。動的な計算は予測内で行う。"""
    rows = [[1.0] + [0.0] * top]
    for degree in range(1, top + 1):
        row = [0.0] + rows[-1][:-1]
        if degree > 1:
            row = [x - float(degree - 1) * y for x, y in zip(row, rows[-2])]
        rows.append(row)
    return fnp.asarray(rows, dtype=fnp.float64)


_POSITIVE_HE10 = _hermite_power_matrix(10)
_POSITIVE_LOWER_INDICES = fnp.asarray([
    [max(p - k, 0) for k in range(5)] for p in range(1, 5)
], dtype=fnp.int64)
_POSITIVE_LOWER_FACTORS = fnp.asarray([
    [float(math.factorial(p) // math.factorial(p - k)) if k <= p else 0.0 for k in range(5)]
    for p in range(1, 5)
], dtype=fnp.float64)
_POSITIVE_TAIL_INDICES = fnp.asarray([
    [max(k - p, 0) for k in range(13)] for p in range(1, 5)
], dtype=fnp.int64)
_RELU_FACTORIALS = fnp.asarray([1.0, 1.0, 2.0, 6.0, 24.0], dtype=fnp.float64)
_FP64_VARIANCE_ROUNDOFF = 64.0 * 2.0 ** -52
_FP32_VARIANCE_ROUNDOFF = 64.0 * 2.0 ** -23


def _positive_relu_powers(m, var, sigma, tdiag, omega, through):
    """正の D=1+u+u²/2 を正規化・標準化した分布の E[ReLU(X)^p] (FP64)。

    u の 2 次補正は中心化・標準化後の H3..H8 を
    τ/6, ω/24, 0, τ²/72, τω/144, ω²/1152 にする。
    var は呼び出し規約のために残し、σ から有効分散を一度だけ作る。
    tdiag=None は厳密な零歪度、さらに omega=None は厳密な Gaussian を表す。
    """
    m, sigma = (fnp.asarray(x, dtype=fnp.float64) for x in (m, sigma))
    gaussian = tdiag is None and omega is None
    if gaussian:
        m_eff, s_eff = m, sigma
    else:
        o = fnp.broadcast_to(fnp.asarray(0.0 if omega is None else omega, dtype=fnp.float64), m.shape)
        oo = o * o
        if tdiag is None:
            a = fnp.stack((o / 24.0 - oo / 16.0, -oo / 72.0), axis=0)
            left, right, weights = _POSITIVE_EVEN_HERMITE_MAP
        else:
            t = fnp.broadcast_to(fnp.asarray(tdiag, dtype=fnp.float64), m.shape)
            tt, to = t * t, t * o
            a = fnp.stack((
                t / 6.0 - to / 4.0,
                o / 24.0 - tt / 8.0 - oo / 16.0,
                -to / 12.0, -oo / 72.0,
            ), axis=0)
            left, right, weights = _POSITIVE_HERMITE_MAP
        products = a[list(left)] * a[list(right)]
        features = fnp.concatenate((fnp.ones_like(m)[None, :], a, products), axis=0)
        b = fnp.matmul(weights, features)
        Z = b[0]
        mq = None if tdiag is None else b[1] / Z
        vq = 1.0 + 2.0 * b[2] / Z
        if mq is not None:
            vq = vq - mq * mq
        if not float(fnp.min(vq)) > 0.0:
            raise ValueError("positive Hermite marginal variance must be positive")
        s_eff = sigma / fnp.sqrt(vq)
        m_eff = m if mq is None else m - s_eff * mq
    alpha_eff = m_eff / s_eff
    Phi = fnp.asarray(flops.stats.norm.cdf(alpha_eff), dtype=fnp.float64)
    phi = fnp.asarray(flops.stats.norm.pdf(alpha_eff), dtype=fnp.float64)
    # flopscope の cdf は (1+erf)/2 で、深い負の裾では桁落ちする。
    # Φ(-x)/φ(x) の Laplace 連分数を裾だけに使い、微小な raw moment の符号を保つ。
    tail_mask = alpha_eff < -4.0
    if bool(fnp.any(tail_mask)):
        x = fnp.maximum(-alpha_eff, 4.0)
        t_tail = 1.0 / (x * x)
        mills = fnp.polyval(_MILLS_P32, t_tail) / fnp.polyval(_MILLS_Q32, t_tail) / x
        Phi = fnp.where(tail_mask, phi * mills, Phi)
    s_powers = [None, s_eff, s_eff * s_eff]
    for p in range(3, through + 1):
        s_powers.append(s_powers[-1] * s_eff)
    moments = [Phi, m_eff * Phi + s_eff * phi]
    for p in range(2, through + 1):
        moments.append(m_eff * moments[-1] + float(p - 1) * s_powers[2] * moments[-2])
    if gaussian:
        return moments[1:]
    he = fnp.matmul(
        _POSITIVE_HE10, fnp.vander(-alpha_eff, N=11, increasing=True).T
    )
    sigma_powers = fnp.stack(s_powers[1:through + 1], axis=0)
    scaled_b = fnp.concatenate((b[:1], b[1:through + 1] * sigma_powers), axis=0)
    moment_table = fnp.stack(moments, axis=0)
    lower = fnp.einsum(
        "pkn,kn,pk->pn",
        moment_table[_POSITIVE_LOWER_INDICES[:through, :through + 1]],
        scaled_b, _POSITIVE_LOWER_FACTORS[:through, :through + 1],
        optimize=False,
    )
    he_padded = fnp.concatenate((fnp.zeros_like(m)[None, :], he), axis=0)
    tail = fnp.einsum(
        "kn,pkn->pn", b, he_padded[_POSITIVE_TAIL_INDICES[:through]], optimize=False
    )
    scales = _RELU_FACTORIALS[1:through + 1, None] * sigma_powers * phi
    raw_powers = (lower + scales * tail) / Z
    return [raw_powers[p] for p in range(through)]


def _compact_family_columns(value, keep, count, columns):
    """横並びの家族ブロックを一括で間引く。None の脚は作らない。"""
    if value is None:
        return None
    if value.ndim == 1:
        return fnp.reshape(fnp.reshape(value, (count, columns))[keep], (-1,))
    return fnp.reshape(
        fnp.reshape(value, (value.shape[0], count, columns))[:, keep, :],
        (value.shape[0], len(keep) * columns),
    )


def _old_basis(candidate, energy, minimum):
    """小さい QR の R のエネルギーで共有基底を選ぶ。energy=1 は SVD 不要。"""
    basis, rotation = fnp.linalg.qr(candidate)
    if energy < 1.0:
        left, singular, _ = fnp.linalg.svd(rotation, full_matrices=False)
        cumulative = fnp.cumsum(singular * singular)
        used = int(fnp.searchsorted(cumulative, energy * cumulative[-1])) + 1
        used = min(basis.shape[1], max(minimum, used))
        left = left[:, :used]
        basis = fnp.matmul(basis, left)
        rotation = fnp.matmul(left.T, rotation)
    return basis, rotation


class Estimator(BaseEstimator):
    kmax = 3  # Mehler 展開の次数(共分散)
    window = 16  # 家族を何層分持つか(古い家族を捨てて FLOP を減らす)
    iik_rank = 16  # 補正 Δ の残差の低ランク近似の階数(None なら補正しない)
    use_k4 = True  # κ4補正。方向依存モードではコア・切り口・追加誕生をまとめて有効化
    omega_scale = 1.0  # ω の全体倍率(理論値 1)
    tau_scale = 1.0  # 3 次キュムラントの倍率(理論値 1)
    calib = 0.0  # 最終層の出力に掛ける縮小係数(正解 ≈ (1 - calib) × 予測)
    collect = False  # True なら層ごとの内部量を self.features に残す(解析用)
    # 古い家族の共有基底化: 年齢 age_old に達した家族は、全員で共有する n×rank_old の基底 U に
    # 射影して持つ(M ≈ U F)。輸送は U だけを運べばよく(X = U F なら T X = (T U) F で厳密)、
    # 家族ごとの n³ の行列積が n² r に減る。None なら使わない。
    # 旧放射モードの探索・検証: 基底化なし1.540e-8 → age6・階数256・作り直しなし1.382e-8。
    # 作り直す版より、最初の基底を運び続ける版の方が安く(C/B -0.006)誤差も小さかった。
    age_old = 7
    rank_old = 256
    family_prune_rel = 0.0  # 家族別の対角寄与ヒューリスティック。0 は評価・間引きとも無効
    rank_old_energy = 1.0  # 1 は固定階数(小行列 SVD も実行しない)
    rank_old_min = 32
    rebuild_old = False  # True: 家族が入るたびに基底を作り直す / False: 最初の基底を運び続ける
    # 旧放射モード用のκ4→3次誕生。T_ijk=K_ij c2_k+K_ik c2_j+K_jk c2_i(Kは対称、対角0):
    #  (a) (2,1,1) 型 κ̃4[z_i,z_j,z_k,z_k] ≈ λ ρ_ij、λ = beta_k4 · ω: K += (λ/2) c1 ρ_off c1
    #  (b) 放射部分 κ̃4_iijj = ω/3 と ρ の辺 1 本の経路: K += (ω/12)(c3 ρ_off c1 + c1 ρ_off c3)
    beta_k4 = 0.0
    k4_birth_rad = False
    directional_k4 = True  # Falseは旧放射モードとの対照用。旧誕生項とは排他的
    regen_p0 = 1.6
    regen_p1 = regen_p2 = 1.2
    regen_p3 = 1.6
    regen_feed = 0.5
    regen_use31 = 1.0
    regen_start = 1
    regen_end = 16  # 0始まり、終端を含む
    # 3 次 (2,1) 切り口 τ と ρ の辺 1 本の経路による誕生(参照実装 AUGMENT の k3 脚)。
    # 中心 q の脚 X = M B、B_iq = (c2_i c2_q τ_iiq + c1_i c3_q τ_iqq)/2。
    # B は低ランクに近い(上位 64 方向で 70〜97%)ので B ≈ Q R(階数 k3leg_rank)と持ち、
    # 運ぶのは n×r の Xq = M Q だけにする(X = Xq R)。
    # 旧放射モードの検証28MLP: なし1.382e-8 → 階数16で1.313e-8(MSE3.5e-8→3.1e-8)。
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
        rank = 0 if self.iik_rank is None else min(self.iik_rank, n)
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
        C = None  # 最初の入力共分散 I は密に作らず WᵀW として縮約する
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
        FX3q_o = R3_o = None  # 古い家族も X3 = (U FX3q) R3 の細い脚で持つ
        r3 = min(self.k3leg_rank, n)
        # 固定列スケールと細い残差を保存し、密な PΔ/κ3 脚は射影・回転しない。
        U = FA_o = FM_o = FZ_o = c2_o = e_o = dd_o = Q_o = None
        r_old = None if self.rank_old is None else min(self.rank_old, n)
        old_min = 0 if r_old is None else min(r_old, max(1, self.rank_old_min))
        prune = self.family_prune_rel > 0.0
        birth_w = fnp.empty(0, dtype=fnp.int64) if prune else None
        rows = []

        for layer, w in enumerate(mlp.weights):
            last = layer == depth - 1

            # --- 線形層: z の平均・共分散 ---
            m = w.T @ mu
            if layer == 0:
                if last:
                    raw_pre_var = fnp.sum(w * w, axis=0)
                else:
                    S = fnp.matmul(w.T, w)
                    raw_pre_var = fnp.diag(S)
            elif last:
                raw_pre_var = fnp.sum(w * fnp.matmul(C, w), axis=0)
            else:
                S = fnp.einsum("ij,ia,jb->ab", C, w, w)
                raw_pre_var = fnp.diag(S)
            if regen:
                # 共分散の対角スケールに対する FP32 丸め誤差だけを許す。
                pre_roundoff = _FP32_VARIANCE_ROUNDOFF * fnp.max(fnp.abs(raw_pre_var))
                if not float(fnp.min(raw_pre_var) + pre_roundoff) >= 0.0:
                    raise ValueError("directional_k4 preactivation variance exceeds negative roundoff allowance")
            var = fnp.maximum(raw_pre_var, 1e-12)
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
                tv = raw_pre_var - fnp.matmul(WW.T, var_prev)
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
            absent_third = regen and nf == 0 and U is None
            D3 = None if absent_third else fnp.zeros(n, dtype=f32)
            D21 = None
            score_w = score_o = None
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
                if prune:
                    score_w = fnp.sum(fnp.reshape(fnp.abs(d3), (n, nf, n)), axis=(0, 2))
                if g_w is not None:
                    X_w = (A_w - M_w * c1_w[None, :]) * g_w[None, :]
                    M3 = fnp.reshape(M_w, (n, nf, n))
                    kd = 2.0 * fnp.sum(fnp.reshape(X_w * M_w, (n, nf, n)), axis=2)
                    uu = fnp.einsum(
                        "nfm,fm->nf", M3, fnp.reshape(u_w if regen else c2_w, (nf, n))
                    )
                    D3 = D3 + 3.0 * fnp.sum(kd * uu, axis=1)
                    if prune:
                        score_w = score_w + fnp.sum(fnp.abs(kd * uu), axis=0)
                if not last:
                    XM = AA * c2_w[None, :]
                    XA = 2.0 * (A_w * M_w) * c2_w[None, :]
                    if e_w is not None:
                        XM = XM + MM * dd_w[None, :] + 2.0 * (M_w * PD)
                        XA = XA + MM * e_w[None, :]
                    if X3_w is not None:
                        XM = XM + 2.0 * (X3_w * A_w)
                        XA = XA + 2.0 * (X3_w * M_w)
                    if g_w is not None:
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
                    if g_w is not None:
                        D21 = D21 + fnp.matmul(kd, uu.T)
                    if X3_w is not None:
                        Y3 = fnp.einsum("nkm,krm->nkr", fnp.reshape(M_w * A_w, (n, nf, n)), R3_s)
                        D21 = D21 + 2.0 * fnp.matmul(fnp.reshape(Y3, (n, nf * r3)), X3q_w.T)
                    if Z_w is not None:
                        Y = fnp.einsum("nkm,kmr->nkr", fnp.reshape(MM, (n, nf, n)), Q_s)
                        D21 = D21 + fnp.matmul(fnp.reshape(Y, (n, nf * rank)), Z_w.T)
            if U is not None:
                # 基本脚だけを復元。残差・κ3脚の転置読み出しは細い因子へ畳み込む。
                ko = c2_o.shape[0] // n
                Ao = fnp.matmul(U, FA_o)
                Mo = fnp.matmul(U, FM_o)
                AAo = Ao * Ao
                d3o = AAo * Mo * c2_o[None, :]
                Po = MMo = None
                if e_o is not None:
                    MMo = Mo * Mo
                    Po = Ao * e_o[None, :] + Mo * dd_o[None, :]
                    if FZ_o is not None:
                        Zo = fnp.matmul(U, FZ_o)
                        Po = Po + fnp.reshape(
                            fnp.einsum("nkr,kmr->nkm", fnp.reshape(Zo, (n, ko, rank)), Q_o),
                            (n, ko * n),
                        )
                    d3o = d3o + MMo * Po
                X3o = None
                if FX3q_o is not None:
                    X3qo = fnp.matmul(U, FX3q_o)
                    X3o = fnp.reshape(
                        fnp.einsum("nkr,krm->nkm", fnp.reshape(X3qo, (n, ko, r3)), R3_o),
                        (n, ko * n),
                    )
                    d3o = d3o + 2.0 * X3o * Mo * Ao
                D3 = D3 + 3.0 * fnp.sum(d3o, axis=1)
                if prune:
                    score_o = fnp.sum(fnp.reshape(fnp.abs(d3o), (n, ko, n)), axis=(0, 2))
                if g_o is not None:
                    Xo = (Ao - Mo * c1_o[None, :]) * g_o[None, :]
                    Mo3 = fnp.reshape(Mo, (n, ko, n))
                    kdo = 2.0 * fnp.sum(fnp.reshape(Xo * Mo, (n, ko, n)), axis=2)
                    uo = fnp.einsum(
                        "nfm,fm->nf", Mo3, fnp.reshape(u_o if regen else c2_o, (ko, n))
                    )
                    D3 = D3 + 3.0 * fnp.sum(kdo * uo, axis=1)
                    if prune:
                        score_o = score_o + fnp.sum(fnp.abs(kdo * uo), axis=0)
                if not last:
                    XMo = AAo * c2_o[None, :]
                    XAo = 2.0 * (Ao * Mo) * c2_o[None, :]
                    if Po is not None:
                        XMo = XMo + 2.0 * Mo * Po + MMo * dd_o[None, :]
                        XAo = XAo + MMo * e_o[None, :]
                    if X3o is not None:
                        XMo = XMo + 2.0 * (X3o * Ao)
                        XAo = XAo + 2.0 * (X3o * Mo)
                    if g_o is not None:
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
                    Racc = fnp.matmul(XMo, FM_o.T) + fnp.matmul(XAo, FA_o.T)
                    if FZ_o is not None:
                        Y = fnp.einsum("nkm,kmr->nkr", fnp.reshape(MMo, (n, ko, n)), Q_o)
                        Racc = Racc + fnp.matmul(fnp.reshape(Y, (n, ko * rank)), FZ_o.T)
                    if X3o is not None:
                        Y3 = fnp.einsum("nkm,krm->nkr", fnp.reshape(Mo * Ao, (n, ko, n)), R3_o)
                        Racc = Racc + 2.0 * fnp.matmul(
                            fnp.reshape(Y3, (n, ko * r3)), FX3q_o.T
                        )
                    t_old = fnp.matmul(Racc, U.T)
                    D21 = t_old if D21 is None else D21 + t_old
            D3_raw, D21_raw = D3, D21  # 家族が表す値そのもの(倍率を掛ける前)
            # 読み出した層の寄与を指標に、次の輸送前に家族を間引く。
            # 対角寄与の L1 は安価なヒューリスティックで、全テンソルの誤差上界ではない。
            removed_w = removed_o = 0
            young_read = nf
            old_read = 0 if U is None else c2_o.shape[0] // n
            old_rank_read = 0 if U is None else U.shape[1]
            if prune and not last and (score_w is not None or score_o is not None):
                scores = [score for score in (score_w, score_o) if score is not None]
                cutoff = self.family_prune_rel * fnp.max(fnp.concatenate(scores))
                if score_w is not None:
                    keep = fnp.flatnonzero(score_w > cutoff)
                    removed_w = nf - len(keep)
                    if removed_w:
                        A_w = _compact_family_columns(A_w, keep, nf, n)
                        M_w = _compact_family_columns(M_w, keep, nf, n)
                        c2_w = _compact_family_columns(c2_w, keep, nf, n)
                        e_w = _compact_family_columns(e_w, keep, nf, n)
                        dd_w = _compact_family_columns(dd_w, keep, nf, n)
                        c1_w = _compact_family_columns(c1_w, keep, nf, n)
                        g_w = _compact_family_columns(g_w, keep, nf, n)
                        u_w = _compact_family_columns(u_w, keep, nf, n)
                        Z_w = _compact_family_columns(Z_w, keep, nf, rank)
                        Q_s = None if Q_s is None else Q_s[keep]
                        X3q_w = _compact_family_columns(X3q_w, keep, nf, r3)
                        R3_s = None if R3_s is None else R3_s[keep]
                        birth_w = birth_w[keep]
                        nf = len(keep)
                if score_o is not None:
                    keep = fnp.flatnonzero(score_o > cutoff)
                    removed_o = old_read - len(keep)
                    if removed_o:
                        FA_o = _compact_family_columns(FA_o, keep, old_read, n)
                        FM_o = _compact_family_columns(FM_o, keep, old_read, n)
                        c2_o = _compact_family_columns(c2_o, keep, old_read, n)
                        e_o = _compact_family_columns(e_o, keep, old_read, n)
                        dd_o = _compact_family_columns(dd_o, keep, old_read, n)
                        c1_o = _compact_family_columns(c1_o, keep, old_read, n)
                        g_o = _compact_family_columns(g_o, keep, old_read, n)
                        u_o = _compact_family_columns(u_o, keep, old_read, n)
                        FZ_o = _compact_family_columns(FZ_o, keep, old_read, rank)
                        Q_o = None if Q_o is None else Q_o[keep]
                        FX3q_o = _compact_family_columns(FX3q_o, keep, old_read, r3)
                        R3_o = None if R3_o is None else R3_o[keep]
                        if not len(keep):
                            U = FA_o = FM_o = FZ_o = c2_o = e_o = dd_o = Q_o = None
                            c1_o = g_o = u_o = FX3q_o = R3_o = None
            if self.tau_scale != 1.0:
                if D3 is not None:
                    D3 = D3 * self.tau_scale
                if D21 is not None:
                    D21 = D21 * self.tau_scale
            tdiag = None if absent_third else D3 / (sigma * var)  # τ_iii = T_iii / σ³
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
                raw_powers = _positive_relu_powers(
                    m, var, sigma, tdiag, None if layer == 0 else omega, 2 if last else 4
                )
                raw_mu, raw_ez2 = raw_powers[:2]
                mu_new = raw_mu.astype(f32)
                if not last:
                    raw_ez3, raw_ez4 = raw_powers[2:]
                raw_mean_sq = raw_mu * raw_mu
                raw_var = raw_ez2 - raw_mean_sq
                roundoff = _FP64_VARIANCE_ROUNDOFF * fnp.maximum(fnp.abs(raw_ez2), raw_mean_sq)
                if not float(fnp.min(raw_var + roundoff)) >= 0.0:
                    raise ValueError("positive Hermite ReLU variance exceeds negative roundoff allowance")
                var_post = fnp.maximum(raw_var, 0.0).astype(f32)
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
                    "m": m, "sigma": sigma, "alpha": alpha,
                    "tdiag": tdiag if tdiag is not None else fnp.zeros_like(m), "mu": mu_new,
                    "mu_gauss": c[0], "var_post": var_post,
                    "D3": D3_raw if D3_raw is not None else fnp.zeros_like(m),
                    "D21": D21_raw if D21_raw is not None else fnp.zeros((n, n), dtype=f32),
                    "compute": {
                        "young_families_read": young_read, "old_families_read": old_read,
                        "young_families_removed": removed_w, "old_families_removed": removed_o,
                        "old_rank": old_rank_read,
                    },
                })
                if regen:
                    self.features[-1].update({
                        "k4_lambda": lam, "k4_gdiag": dG, "k4_metric": metric, "k4_omega": omega,
                        "k4_raw_mean": raw_mu, "k4_raw_second": raw_ez2,
                        "k4_raw_variance": raw_var,
                        "k4_raw_pre_variance": raw_pre_var,
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
            if regen and layer > 0:
                dv, gv = metric / var, dG / var
                k22 = (fnp.outer(dv, gv) + fnp.outer(gv, dv)) / 6.0
                fnp.fill_diagonal(k22, 0.0)
                k4 = (omega, k22)
                if self.regen_use31 != 0.0 and strength != 0.0:
                    k31 = (self.regen_use31 * 0.5 * lam) * dv[:, None] * rho
                    fnp.fill_diagonal(k31, 0.0)
            elif self.use_k4 and not directional:
                om = omega * fnp.ones_like(sigma)
                q = fnp.sqrt(om / 3.0)
                k4 = (om, fnp.outer(q, q))
            c2p = cov21 = None
            if regen:
                c2p = [e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]
                for kk in range(3, kmax + 5):
                    c2p.append(2.0 * var * phi * he[kk - 3])
                # 軸は [Hermite次数, 出力関数, 頂点]。3表面を同じ演算列で運ぶ。
                coefficients = fnp.stack((fnp.stack(c[:kmax + 5]), fnp.stack(c2p)), axis=1)
                pair = _pair_terms(
                    coefficients[:, [0, 1, 1], :], coefficients[:, [0, 0, 1], :],
                    R, kmax, t_iij, t_ijj, tdiag, k4, k31,
                )
                Cn, cov21, cov22 = pair[0], pair[1], pair[2]
            else:
                Cn = _pair_terms(c, c, R, kmax, t_iij, t_ijj, tdiag, k4, k31)
            fnp.fill_diagonal(Cn, var_post)
            C = flops.as_symmetric(Cn, symmetry=(0, 1))
            if regen:
                post_k22 = (
                    cov22 - 2.0 * mu[:, None] * cov21.T - 2.0 * mu[None, :] * cov21
                    + 4.0 * fnp.outer(mu, mu) * Cn - 2.0 * Cn * Cn
                )
                fnp.fill_diagonal(post_k22, 0.0)
                post_k4 = (
                    raw_ez4 - 4.0 * raw_mu * raw_ez3 + 6.0 * raw_mean_sq * raw_ez2
                    - 3.0 * raw_mean_sq * raw_mean_sq - 3.0 * raw_var ** 2
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
            expires = nf > 0 and (
                layer - int(birth_w[0]) >= window if prune else nf >= window
            )
            if expires:
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
                if prune:
                    birth_w = birth_w[1:]
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
                if prune:
                    birth_w = fnp.concatenate((birth_w, fnp.asarray([layer], dtype=fnp.int64)))
                nf += 1
            transfer = self.age_old is not None and r_old is not None and nf > 0
            if transfer:
                transfer = (
                    layer - int(birth_w[0]) >= self.age_old if prune else nf > self.age_old
                )
            if transfer:
                # 最も古い若い家族を共有基底へ移す。固定列スケールは射影しない。
                A_f, M_f, c2_f = A_w[:, :n], M_w[:, :n], c2_w[:n]
                A_w, M_w, c2_w = A_w[:, n:], M_w[:, n:], c2_w[n:]
                nf -= 1
                if prune:
                    birth_w = birth_w[1:]
                # 再直交化だけなら QR の R が既存基底の座標変換そのもの。
                if U is None or self.rebuild_old:
                    Yb = fnp.matmul(M_f, w_next[:, n - r_old:])
                    Yb = fnp.matmul(M_f, fnp.matmul(M_f.T, Yb))
                    U_new, _ = _old_basis(Yb, self.rank_old_energy, old_min)
                    Tm = None if U is None else fnp.matmul(U_new.T, U)
                else:
                    U_new, Tm = _old_basis(U, self.rank_old_energy, old_min)
                if U is not None:
                    FA_o = fnp.matmul(Tm, FA_o)
                    FM_o = fnp.matmul(Tm, FM_o)
                    if FZ_o is not None:
                        FZ_o = fnp.matmul(Tm, FZ_o)
                    if FX3q_o is not None:
                        FX3q_o = fnp.matmul(Tm, FX3q_o)
                fa, fm = fnp.matmul(U_new.T, A_f), fnp.matmul(U_new.T, M_f)
                if U is None:
                    FA_o, FM_o, c2_o = fa, fm, c2_f
                else:
                    FA_o = fnp.concatenate([FA_o, fa], axis=1)
                    FM_o = fnp.concatenate([FM_o, fm], axis=1)
                    c2_o = fnp.concatenate([c2_o, c2_f])
                if e_w is not None:
                    ef, ddf = e_w[:n], dd_w[:n]
                    e_o = ef if e_o is None else fnp.concatenate([e_o, ef])
                    dd_o = ddf if dd_o is None else fnp.concatenate([dd_o, ddf])
                    e_w, dd_w = e_w[n:], dd_w[n:]
                if Z_w is not None:
                    fz = fnp.matmul(U_new.T, Z_w[:, :rank])
                    FZ_o = fz if FZ_o is None else fnp.concatenate([FZ_o, fz], axis=1)
                    Q_o = Q_s[:1] if Q_o is None else fnp.concatenate([Q_o, Q_s[:1]], axis=0)
                    Z_w, Q_s = Z_w[:, rank:], Q_s[1:]
                if X3q_w is not None:
                    fx3q = fnp.matmul(U_new.T, X3q_w[:, :r3])
                    FX3q_o = fx3q if FX3q_o is None else fnp.concatenate([FX3q_o, fx3q], axis=1)
                    R3_o = R3_s[:1] if R3_o is None else fnp.concatenate([R3_o, R3_s[:1]], axis=0)
                    X3q_w, R3_s = X3q_w[:, r3:], R3_s[1:]
                if g_w is not None:
                    c1f, gf = c1_w[:n], g_w[:n]
                    c1_o = c1f if c1_o is None else fnp.concatenate([c1_o, c1f])
                    g_o = gf if g_o is None else fnp.concatenate([g_o, gf])
                    c1_w, g_w = c1_w[n:], g_w[n:]
                    if regen:
                        uf = u_w[:n]
                        u_o = uf if u_o is None else fnp.concatenate([u_o, uf])
                        u_w = u_w[n:]
                U = U_new
            if self.collect:
                self.features[-1]["compute"].update({
                    "young_families_next": nf,
                    "old_families_next": 0 if U is None else c2_o.shape[0] // n,
                    "old_rank_next": 0 if U is None else U.shape[1],
                })

        if self.calib:
            rows[-1] = rows[-1] * (1.0 - self.calib)
        return fnp.stack(rows, axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """単独実行時の比較用に examples の推定器を読む。提出時には呼ばない。"""
    import importlib.util
    from pathlib import Path
    import sys

    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[candidate.stem] = module
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(f"Unknown baseline {name!r}; see examples/.")


if __name__ == "__main__":
    import argparse
    from local_engine import build_mlp, compare_against_monte_carlo

    parser = argparse.ArgumentParser(description="Run the submission estimator locally.")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print("--- Submission estimator ---")
    compare_against_monte_carlo(Estimator(), mlp, seed=args.seed)
    if args.baseline:
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(_load_baseline(args.baseline)(), mlp, seed=args.seed)
