"""root estimator.py の Estimator.predict(既定設定)の PyTorch 移植。学習用(提出物ではない)。

対象の設定(幅 n・深さ 16 以下、窓 = 深さ): directional κ4 + 再生、球面入力、(3,1) 二項則 b2、
学習補正器 Δω、家族(若い家族 ≤ age_old=6、古い家族は共有基底 rank_old=256 を QR で更新)、
補正 Δ の低ランク脚(iik_rank=4)、k3 脚(k3leg_rank=4)。Strassen は使わず密な積で計算する。
dtype は flopscope 版に合わせる(密な状態は FP32、正値 Hermite 密度・Hermite 表・κ4 中心化は FP64)。

先頭に MLP のバッチ軸 B を持つ: ベクトル (B, n)、行列 (B, n, n)、家族の横並び (B, n, nf·n)。
学習するパラメータ: b2(k31_two_term)、regen_p0..p3、regen_feed、regen_use31、補正器の全要素
(mean, inv_std, W, b, yscale)。_OMEGA_G は directional モードの鎖に入らないので持たない。
"""
from __future__ import annotations

import base64
import math
import re
import struct
from pathlib import Path

import torch
from torch import nn

F32, F64 = torch.float32, torch.float64
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------------------------
# 固定の係数表(estimator.py と同じ構成)
def _hermite_power_matrix(top):
    rows = [[1.0] + [0.0] * top]
    for degree in range(1, top + 1):
        row = [0.0] + rows[-1][:-1]
        if degree > 1:
            row = [x - float(degree - 1) * y for x, y in zip(row, rows[-2])]
        rows.append(row)
    return rows


def _positive_hermite_map(degrees):
    pairs = tuple((i, j) for i in range(len(degrees)) for j in range(i, len(degrees)))
    weights = [[0.0] * (1 + len(degrees) + len(pairs)) for _ in range(13)]
    weights[0][0] = 1.0
    for i, degree in enumerate(degrees):
        weights[degree][1 + i] = 1.0
    for column, (i, j) in enumerate(pairs, 1 + len(degrees)):
        di, dj = degrees[i], degrees[j]
        for r in range(min(di, dj) + 1):
            weights[di + dj - 2 * r][column] = (
                (0.5 if i == j else 1.0) * math.factorial(r) * math.comb(di, r) * math.comb(dj, r)
            )
    return [i for i, _ in pairs], [j for _, j in pairs], weights


def _mills_polynomials(levels):
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


class _Tables:
    """デバイス上の固定表(デバイスごとに 1 回作る)。"""

    def __init__(self, device):
        t = lambda x, dt=F64: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
        self.he = {top: t(_hermite_power_matrix(top)) for top in (7, 10)}
        l, r, w = _positive_hermite_map((3, 4, 5, 6))
        self.map_full = (l, r, t(w))
        l, r, w = _positive_hermite_map((4, 6))
        self.map_even = (l, r, t(w))
        p, q = _mills_polynomials(32)
        self.mills_p, self.mills_q = list(p), list(q)
        self.lower_idx = t([[max(p - k, 0) for k in range(5)] for p in range(1, 5)], torch.long)
        self.lower_fac = t([[float(math.factorial(p) // math.factorial(p - k)) if k <= p else 0.0
                             for k in range(5)] for p in range(1, 5)])
        self.tail_idx = t([[max(k - p, 0) for k in range(13)] for p in range(1, 5)], torch.long)
        self.relu_fact = t([1.0, 1.0, 2.0, 6.0, 24.0])


_TABLES = {}


def tables(device):
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = _Tables(device)
    return _TABLES[key]


# ----------------------------------------------------------------------------------------------
def norm_pdf64(x):
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def norm_cdf64(x):
    # flopscope と同じ (1 + erf)/2(FP64)
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def vander_increasing(x, N):
    """numpy.vander(x, N, increasing=True) を (N, *x.shape) で返す(累積積)。"""
    cols = [torch.ones_like(x), x]
    for _ in range(2, N):
        cols.append(cols[-1] * x)
    return torch.stack(cols[:N], dim=0)


def hermite_neg_alpha(T, alpha, upto):
    """He_0..He_upto(-α) を (upto+1, B, n) の FP32 で(FP64 で評価)。"""
    powers = vander_increasing(-alpha.to(F64), upto + 1)  # (k, B, n)
    return torch.einsum("hk,kbn->hbn", T.he[upto], powers).to(F32)


def polyval(coeffs, x):
    y = torch.zeros_like(x)
    for c in coeffs:
        y = y * x + c
    return y


def positive_relu_powers(T, m, sigma, tdiag, omega, through):
    """estimator._positive_relu_powers の移植(omega は常に与えられる)。返り値は FP64 の list。"""
    m, sigma = m.to(F64), sigma.to(F64)
    o = omega.to(F64)
    oo = o * o
    if tdiag is None:
        a = torch.stack((o / 24.0 - oo / 16.0, -oo / 72.0), dim=0)
        left, right, weights = T.map_even
    else:
        t = tdiag.to(F64)
        tt, to = t * t, t * o
        a = torch.stack((
            t / 6.0 - to / 4.0,
            o / 24.0 - tt / 8.0 - oo / 16.0,
            -to / 12.0, -oo / 72.0,
        ), dim=0)
        left, right, weights = T.map_full
    products = a[left] * a[right]
    features = torch.cat((torch.ones_like(m)[None], a, products), dim=0)
    b = torch.einsum("kf,fbn->kbn", weights, features)
    Z = b[0]
    mq = None if tdiag is None else b[1] / Z
    vq = 1.0 + 2.0 * b[2] / Z
    if mq is not None:
        vq = vq - mq * mq
    s_eff = sigma / torch.sqrt(vq)
    m_eff = m if mq is None else m - s_eff * mq
    alpha_eff = m_eff / s_eff
    Phi = norm_cdf64(alpha_eff)
    phi = norm_pdf64(alpha_eff)
    tail_mask = alpha_eff < -4.0
    x = torch.clamp(-alpha_eff, min=4.0)
    t_tail = 1.0 / (x * x)
    mills = polyval(T.mills_p, t_tail) / polyval(T.mills_q, t_tail) / x
    Phi = torch.where(tail_mask, phi * mills, Phi)
    s_powers = [None, s_eff, s_eff * s_eff]
    for _ in range(3, through + 1):
        s_powers.append(s_powers[-1] * s_eff)
    moments = [Phi, m_eff * Phi + s_eff * phi]
    for p in range(2, through + 1):
        moments.append(m_eff * moments[-1] + float(p - 1) * s_powers[2] * moments[-2])
    he = torch.einsum("hk,kbn->hbn", T.he[10], vander_increasing(-alpha_eff, 11))
    sigma_powers = torch.stack(s_powers[1:through + 1], dim=0)
    scaled_b = torch.cat((b[:1], b[1:through + 1] * sigma_powers), dim=0)
    moment_table = torch.stack(moments, dim=0)
    idx = T.lower_idx[:through, :through + 1]
    fac = T.lower_fac[:through, :through + 1]
    lower = torch.einsum("pkbn,kbn,pk->pbn", moment_table[idx], scaled_b, fac)
    he_padded = torch.cat((torch.zeros_like(m)[None], he), dim=0)
    tail = torch.einsum("kbn,pkbn->pbn", b, he_padded[T.tail_idx[:through]])
    scales = T.relu_fact[1:through + 1, None, None] * sigma_powers * phi
    raw = (lower + scales * tail) / Z
    return [raw[p] for p in range(through)]


# ----------------------------------------------------------------------------------------------
def decode_corrector_blob(source_path):
    """estimator.py の _OMEGA_CORRECTOR_DIMS / _OMEGA_CORRECTOR_B64 を読む(Python の list で返す)。"""
    text = Path(source_path).read_text()
    dims = tuple(int(x) for x in re.search(
        r"_OMEGA_CORRECTOR_DIMS = \(([^)]*)\)", text).group(1).split(",") if x.strip())
    block = re.search(r"_OMEGA_CORRECTOR_B64 = \((.*?)\n\)", text, re.S).group(1)
    blob = "".join(re.findall(r'"([^"]*)"', block))
    raw = base64.b64decode(blob)
    values = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    pos = 0

    def take(count):
        nonlocal pos
        part = values[pos:pos + count]
        pos += count
        return part

    mean = take(dims[0])
    inv_std = take(dims[0])
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        W = take(d_in * d_out)
        layers.append(([W[i * d_out:(i + 1) * d_out] for i in range(d_in)], take(d_out)))
    yscale = values[pos]
    if pos + 1 != len(values):
        raise ValueError("unexpected blob length")
    return dims, mean, inv_std, layers, yscale


def encode_corrector_blob(mean, inv_std, layers, yscale):
    """decode と同じ並び(mean, inv_std, 層ごとに W 行優先 (in, out), b、yscale)の base64。"""
    values = list(mean) + list(inv_std)
    for W, b in layers:
        values += [x for row in W for x in row] + list(b)
    values.append(float(yscale))
    return base64.b64encode(struct.pack("<%df" % len(values), *values)).decode("ascii")


class ChainParams(nn.Module):
    """学習対象のパラメータ(初期値は estimator.py の既定値)。"""

    def __init__(self, source_path, b2=3.0, regen_p=(1.6, 1.2, 1.2, 1.6), feed=0.5, use31=1.0):
        super().__init__()
        self.b2 = nn.Parameter(torch.tensor(float(b2), dtype=F32))
        self.regen_p = nn.Parameter(torch.tensor(list(regen_p), dtype=F32))
        self.regen_feed = nn.Parameter(torch.tensor(float(feed), dtype=F32))
        self.regen_use31 = nn.Parameter(torch.tensor(float(use31), dtype=F32))
        dims, mean, inv_std, layers, yscale = decode_corrector_blob(source_path)
        self.dims = dims
        # 特徴の平均・1/std と yscale は尺度が特徴ごとに大きく違うので、Adam の 1 歩が尺度に
        # 比例するよう相対量で持つ: mean = mean0 + std0·dmean、inv_std = inv_std0·exp(dlog_is)、
        # yscale = yscale0·exp(dlog_y)(初期値はすべて 0 で、元の値と完全に一致する)。
        self.register_buffer("corr_mean0", torch.tensor(mean, dtype=F32))
        self.register_buffer("corr_inv_std0", torch.tensor(inv_std, dtype=F32))
        self.register_buffer("corr_yscale0", torch.tensor(float(yscale), dtype=F32))
        self.corr_dmean = nn.Parameter(torch.zeros(dims[0], dtype=F32))
        self.corr_dlog_is = nn.Parameter(torch.zeros(dims[0], dtype=F32))
        self.corr_dlog_y = nn.Parameter(torch.zeros((), dtype=F32))
        self.corr_W = nn.ParameterList([nn.Parameter(torch.tensor(W, dtype=F32)) for W, _ in layers])
        self.corr_b = nn.ParameterList([nn.Parameter(torch.tensor(b, dtype=F32)) for _, b in layers])

    @property
    def corr_mean(self):
        return self.corr_mean0 + self.corr_dmean / self.corr_inv_std0

    @property
    def corr_inv_std(self):
        return self.corr_inv_std0 * torch.exp(self.corr_dlog_is)

    @property
    def corr_yscale(self):
        return self.corr_yscale0 * torch.exp(self.corr_dlog_y)

    def scalar_groups(self):
        return [self.b2, self.regen_p, self.regen_feed, self.regen_use31]

    def corrector_groups(self):
        return [self.corr_dmean, self.corr_dlog_is, self.corr_dlog_y, *self.corr_W, *self.corr_b]

    def export(self):
        """(estimator 属性の dict, base64 blob) を返す。"""
        p = self.regen_p.detach().double().cpu().tolist()
        attrs = dict(
            k31_two_term=float(self.b2.detach()), regen_p0=p[0], regen_p1=p[1], regen_p2=p[2],
            regen_p3=p[3], regen_feed=float(self.regen_feed.detach()),
            regen_use31=float(self.regen_use31.detach()),
        )
        layers = [(W.detach().cpu().tolist(), b.detach().cpu().tolist())
                  for W, b in zip(self.corr_W, self.corr_b)]
        with torch.no_grad():
            blob = encode_corrector_blob(self.corr_mean.cpu().tolist(),
                                         self.corr_inv_std.cpu().tolist(), layers,
                                         float(self.corr_yscale))
        return attrs, blob


# ----------------------------------------------------------------------------------------------
def _fill_diag(X, value):
    """fill_diagonal の非破壊版。value はスカラーか (B, n)。"""
    n = X.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=X.device)
    if not torch.is_tensor(value):
        return X.masked_fill(eye, value)
    return torch.where(eye, torch.diag_embed(value.to(X.dtype)), X)


def _outer(a, b):
    return a[:, :, None] * b[:, None, :]


def _tr(X):
    return X.transpose(-1, -2)


def _col(v):  # 行 i のスケール v_i(numpy の v[:, None])
    return v[:, :, None]


def _row(v):  # 列のスケール(numpy の v[None, :])
    return v[:, None, :]


def pair_terms_batched(A, Bc, R, kmax, t_iij, t_ijj, tdiag, k4, k31):
    """estimator._pair_terms_batched の移植。A, Bc: (K, s, B, n)、R: list of (B, n, n)。返り値 (s, B, n, n)。"""
    extra_a = extra_b = None
    if tdiag is not None:
        extra_a = A[4:kmax + 4] * (tdiag / 6.0)
        extra_b = Bc[4:kmax + 4] * (tdiag / 6.0)
    if k4 is not None:
        om, k22 = k4
        add_a = A[5:kmax + 5] * (om / 24.0)
        add_b = Bc[5:kmax + 5] * (om / 24.0)
        extra_a = add_a if extra_a is None else extra_a + add_a
        extra_b = add_b if extra_b is None else extra_b + add_b

    def outer_s(u, v):  # (s, B, n) ⊗ (s, B, n) -> (s, B, n, n)
        return u[..., :, None] * v[..., None, :]

    base_a = A[1:kmax + 1]
    out = None
    for e in range(kmax):
        if extra_a is None:
            term = outer_s(base_a[e], Bc[1 + e])
        else:
            term = outer_s(base_a[e] + extra_a[e], Bc[1 + e]) + outer_s(base_a[e], extra_b[e])
        term = term * R[1 + e]
        out = term if out is None else out + term
    acc = None
    if t_iij is not None:
        P0 = P1 = None
        for k in range(2, kmax + 1):
            Rk = R[k - 2]
            p0 = outer_s(A[k], Bc[k - 1] * 0.5) * Rk
            p1 = outer_s(A[k - 1], Bc[k] * 0.5) * Rk
            P0 = p0 if P0 is None else P0 + p0
            P1 = p1 if P1 is None else P1 + p1
        acc = t_iij * P0 + t_ijj * P1
    if k4 is not None or k31 is not None:
        Pk22 = Pk31 = Pk31t = None
        for e in range(kmax + 1):
            Re = R[e]
            if k4 is not None:
                v = outer_s(A[2 + e] * 0.25, Bc[2 + e]) * Re
                Pk22 = v if Pk22 is None else Pk22 + v
            if k31 is not None:
                v = outer_s(A[e + 3] / 6.0, Bc[e + 1]) * Re
                Pk31 = v if Pk31 is None else Pk31 + v
                v = outer_s(A[e + 1] / 6.0, Bc[e + 3]) * Re
                Pk31t = v if Pk31t is None else Pk31t + v
        if k4 is not None:
            v = k4[1] * Pk22
            acc = v if acc is None else acc + v
        if k31 is not None:
            v = k31 * Pk31 + _tr(k31) * Pk31t
            acc = v if acc is None else acc + v
    if acc is not None:
        out = out + acc
    return out


def omega_shift(P, layer, omega, tdiag, alpha, var, sigma, metric, tg, tv, lam,
                w, WW, A_prev, c_prev, post_k4_prev, var_post_prev, k3d_prev):
    """estimator._omega_shift の移植(バッチ)。"""
    vv = var * var
    v15 = var * sigma
    c1, c2, c3 = c_prev[1], c_prev[2], c_prev[3]
    Y = _tr(A_prev) - _col(c1) * w
    Y2 = Y * Y
    WWw = WW * w
    sums = torch.stack((
        torch.einsum("ba,baj,baj->bj", c3, w, Y2 * Y),
        torch.einsum("ba,baj,baj->bj", c2 * c2, WW, Y2),
        torch.einsum("ba,baj,baj->bj", c1, w, Y),
        torch.einsum("ba,baj,baj->bj", c1 * c2, WWw, Y),
        torch.einsum("ba,baj,baj->bj", c2, w, Y2),
    )) / torch.stack((vv, vv, var, vv, v15))
    diag = torch.einsum("kba,baj->kbj",
                        torch.stack((post_k4_prev, (var_post_prev * var_post_prev).to(F64))),
                        (WW * WW).to(F64)) / vv.to(F64)
    diag3 = torch.einsum("ba,baj->bj", k3d_prev, WWw) / v15
    od = metric * tg / vv
    lamv = lam[:, None].expand_as(omega)
    X = torch.stack((
        omega.to(F64), od.to(F64), (omega - od).to(F64), (metric / var).to(F64), (tv / var).to(F64),
        alpha.to(F64), tdiag.to(F64), torch.full_like(omega, layer / 15.0).to(F64), lamv.to(F64),
        sums[0].to(F64), sums[1].to(F64), sums[2].to(F64), sums[3].to(F64), diag[0], diag[1],
        sums[4].to(F64), diag3.to(F64),
    ), dim=-1).to(F32)  # (B, n, 17)
    h = (X - P.corr_mean) * P.corr_inv_std
    nl = len(P.corr_W)
    for k in range(nl):
        h = h @ P.corr_W[k] + P.corr_b[k]
        if k < nl - 1:
            h = torch.clamp(h, min=0.0)
    shift = h[..., 0] * P.corr_yscale
    return shift - shift.mean(dim=1, keepdim=True)


# ----------------------------------------------------------------------------------------------
class ChainConfig:
    kmax = 3
    iik_rank = 4
    age_old = 6
    rank_old = 256
    regen_start = 1
    regen_end = 16
    k3leg_rank = 4
    learned_omega = True
    symmetrize_gram = True


_STATE_KEYS = (
    "mu", "C", "core_diag", "var_prev", "nf",
    "A_w", "M_w", "c2_w", "e_w", "dd_w", "Z_w", "Q_s", "c1_w", "g_w", "u_w", "X3q_w", "R3_s",
    "U", "FA_o", "FM_o", "FZ_o", "c2_o", "e_o", "dd_o", "Q_o", "c1_o", "g_o", "u_o",
    "FX3q_o", "R3_o", "A_born", "c", "post_k4", "k3d",
)


def _layer(P, cfg, weights, layer, s):
    """1 層分の計算(estimator.predict のループ本体)。s は状態の dict で、新しい dict を返す。

    最後の層では s["row"] だけを返す。それ以外は s["row"] と次層の状態。
    """
    T = tables(weights.device)
    Bsz, depth, n, _ = weights.shape
    dev = weights.device
    kmax = cfg.kmax
    rank = cfg.iik_rank
    r3 = cfg.k3leg_rank
    r_old = min(cfg.rank_old, n)
    window = depth  # 1024×16 の予算では窓 = 深さ(_fit_window の結果)
    sphere_g = -6.0 / (n + 2.0)
    (mu, C, core_diag, var_prev, nf,
     A_w, M_w, c2_w, e_w, dd_w, Z_w, Q_s, c1_w, g_w, u_w, X3q_w, R3_s,
     U, FA_o, FM_o, FZ_o, c2_o, e_o, dd_o, Q_o, c1_o, g_o, u_o,
     FX3q_o, R3_o, A_born, c, post_k4, k3d) = (s[k] for k in _STATE_KEYS)
    regen_p = P.regen_p

    w = weights[:, layer]
    last = layer == depth - 1
    m = torch.einsum("bia,bi->ba", w, mu)
    if layer == 0:
        if last:
            raw_pre_var = (w * w).sum(1)
        else:
            S = _tr(w) @ w
    elif last:
        raw_pre_var = (w * (C @ w)).sum(1)
    else:
        S = _tr(w) @ (C @ w)
    if not last:
        if cfg.symmetrize_gram:
            S = 0.5 * (S + _tr(S))
        raw_pre_var = torch.diagonal(S, dim1=-2, dim2=-1)
    var = torch.clamp(raw_pre_var, min=1e-12)
    sigma = torch.sqrt(var)
    alpha = m / sigma
    a64 = alpha.to(F64)
    phi = norm_pdf64(a64).to(F32)
    Phi = norm_cdf64(a64).to(F32)
    WW = w * w
    metric = WW.sum(1)
    tg = torch.einsum("bia,bi->ba", WW, core_diag)
    tv = raw_pre_var - torch.einsum("bia,bi->ba", WW, var_prev)
    if cfg.regen_start <= layer <= cfg.regen_end:
        position = 3.0 * layer / max(depth - 1, 1)
        knot = min(int(position), 2)
        fraction = position - knot
        strength = regen_p[knot] * (1.0 - fraction) + regen_p[knot + 1] * fraction
    else:
        strength = torch.zeros((), dtype=F32, device=dev)
    denominator = var.mean(1) - strength * tv.mean(1)
    lam = strength * tg.mean(1) / denominator
    if layer == 0:
        lam = torch.full((Bsz,), sphere_g, dtype=F32, device=dev)
    dG = tg + lam[:, None] * tv

    # --- 3 次キュムラントの読み出し ---
    absent_third = nf == 0 and U is None
    D3 = None if absent_third else torch.zeros(Bsz, n, dtype=F32, device=dev)
    D21 = None
    if nf > 0:
        AA = A_w * A_w
        d3 = AA * M_w * _row(c2_w)
        X3_w = torch.einsum("bnkr,bkrm->bnkm", X3q_w.reshape(Bsz, n, nf, r3), R3_s).reshape(
            Bsz, n, nf * n)
        d3 = d3 + 2.0 * (X3_w * M_w * A_w)
        MM = M_w * M_w
        PD = A_w * _row(e_w) + M_w * _row(dd_w)
        ZQ = torch.einsum("bnkr,bkmr->bnkm", Z_w.reshape(Bsz, n, nf, rank), Q_s)
        PD = PD + ZQ.reshape(Bsz, n, nf * n)
        d3 = d3 + MM * PD
        D3 = 3.0 * d3.sum(2)
        X_w = (A_w - M_w * _row(c1_w)) * _row(g_w)
        M3 = M_w.reshape(Bsz, n, nf, n)
        kd = 2.0 * (X_w * M_w).reshape(Bsz, n, nf, n).sum(3)
        uu = torch.einsum("bnfm,bfm->bnf", M3, u_w.reshape(Bsz, nf, n))
        D3 = D3 + 3.0 * (kd * uu).sum(2)
        if not last:
            XM = AA * _row(c2_w)
            XA = 2.0 * (A_w * M_w) * _row(c2_w)
            XM = XM + (MM * _row(dd_w) + 2.0 * (M_w * PD))
            XA = XA + MM * _row(e_w)
            XM = XM + 2.0 * (X3_w * A_w)
            XA = XA + 2.0 * (X3_w * M_w)
            fold_m = A_w * _row(g_w) - 2.0 * M_w * _row(c1_w * g_w)
            XM = XM + (2.0 * uu[..., None] * fold_m.reshape(Bsz, n, nf, n)).reshape(Bsz, n, nf * n)
            XA = XA + (2.0 * uu[..., None] * (M_w * _row(g_w)).reshape(Bsz, n, nf, n)).reshape(
                Bsz, n, nf * n)
            D21 = torch.cat((XA, XM), dim=2) @ _tr(torch.cat((A_w, M_w), dim=2))
            D21 = D21 + kd @ _tr(uu)
            Y3 = torch.einsum("bnkm,bkrm->bnkr", (M_w * A_w).reshape(Bsz, n, nf, n), R3_s)
            D21 = D21 + 2.0 * (Y3.reshape(Bsz, n, nf * r3) @ _tr(X3q_w))
            Yz = torch.einsum("bnkm,bkmr->bnkr", MM.reshape(Bsz, n, nf, n), Q_s)
            D21 = D21 + Yz.reshape(Bsz, n, nf * rank) @ _tr(Z_w)
    if U is not None:
        ko = c2_o.shape[1] // n
        cols_o = ko * n
        factors = torch.cat((FA_o, FM_o), dim=2)
        old_legs = U @ factors
        Ao, Mo = old_legs[..., :cols_o], old_legs[..., cols_o:]
        AAo = Ao * Ao
        d3o = AAo * Mo * _row(c2_o)
        MMo = Mo * Mo
        Po = Ao * _row(e_o) + Mo * _row(dd_o)
        Zo = U @ FZ_o
        Po = Po + torch.einsum("bnkr,bkmr->bnkm", Zo.reshape(Bsz, n, ko, rank), Q_o).reshape(
            Bsz, n, cols_o)
        d3o = d3o + MMo * Po
        X3qo = U @ FX3q_o
        X3o = torch.einsum("bnkr,bkrm->bnkm", X3qo.reshape(Bsz, n, ko, r3), R3_o).reshape(
            Bsz, n, cols_o)
        d3o = d3o + 2.0 * X3o * Mo * Ao
        D3 = D3 + 3.0 * d3o.sum(2)
        Xo = (Ao - Mo * _row(c1_o)) * _row(g_o)
        Mo3 = Mo.reshape(Bsz, n, ko, n)
        kdo = 2.0 * (Xo * Mo).reshape(Bsz, n, ko, n).sum(3)
        uo = torch.einsum("bnfm,bfm->bnf", Mo3, u_o.reshape(Bsz, ko, n))
        D3 = D3 + 3.0 * (kdo * uo).sum(2)
        if not last:
            XMo = AAo * _row(c2_o)
            XAo = 2.0 * (Ao * Mo) * _row(c2_o)
            XMo = XMo + (2.0 * Mo * Po + MMo * _row(dd_o))
            XAo = XAo + MMo * _row(e_o)
            XMo = XMo + 2.0 * (X3o * Ao)
            XAo = XAo + 2.0 * (X3o * Mo)
            fold_m = Ao * _row(g_o) - 2.0 * Mo * _row(c1_o * g_o)
            XMo = XMo + (2.0 * uo[..., None] * fold_m.reshape(Bsz, n, ko, n)).reshape(Bsz, n, cols_o)
            XAo = XAo + (2.0 * uo[..., None] * (Mo * _row(g_o)).reshape(Bsz, n, ko, n)).reshape(
                Bsz, n, cols_o)
            thin = kdo @ _tr(uo)
            D21 = thin if D21 is None else D21 + thin
            Racc = torch.cat((XAo, XMo), dim=2) @ _tr(factors)
            Yz = torch.einsum("bnkm,bkmr->bnkr", MMo.reshape(Bsz, n, ko, n), Q_o)
            Racc = Racc + Yz.reshape(Bsz, n, ko * rank) @ _tr(FZ_o)
            Y3 = torch.einsum("bnkm,bkrm->bnkr", (Mo * Ao).reshape(Bsz, n, ko, n), R3_o)
            Racc = Racc + 2.0 * (Y3.reshape(Bsz, n, ko * r3) @ _tr(FX3q_o))
            D21 = D21 + Racc @ _tr(U)
    D3_raw, D21_raw = D3, D21
    tdiag = None if absent_third else D3 / (sigma * var)
    omega = metric * dG / (var * var)
    if cfg.learned_omega and layer > 0 and A_born is not None:
        omega = omega + omega_shift(P, layer, omega, tdiag, alpha, var, sigma, metric, tg, tv,
                                    lam, w, WW, A_born, c, post_k4, var_prev, k3d)

    # --- Hermite 係数 ---
    top = max(kmax + 4, 7)
    he = hermite_neg_alpha(T, alpha, top)
    he_stack = he[:top - 1]
    sp = sigma * phi
    c_tail = sp[None] * he_stack
    c = [m * Phi + sp, sigma * Phi] + [c_tail[k] for k in range(top - 1)]

    e2_0 = (m * m + var) * Phi + m * sigma * phi
    raw_powers = positive_relu_powers(T, m, sigma, tdiag, omega, 2 if last else 4)
    raw_mu, raw_ez2 = raw_powers[:2]
    mu = raw_mu.to(F32)
    if last:
        return {"row": mu}
    raw_ez3, raw_ez4 = raw_powers[2:]
    raw_mean_sq = raw_mu * raw_mu
    raw_var = raw_ez2 - raw_mean_sq
    var_post = torch.clamp(raw_var, min=0.0).to(F32)

    # --- ReLU 後の共分散 ---
    rho = S / _outer(sigma, sigma)
    if D21 is not None:
        t_iij = D21 / _outer(var, sigma)
        t_ijj = _tr(t_iij)
    else:
        t_iij = t_ijj = None
    R = [torch.ones_like(rho), rho]
    rp = rho
    for k in range(2, kmax + 1):
        rp = rp * rho
        R.append(rp / float(math.factorial(k)))
    dv, gv = metric / var, dG / var
    k22 = _fill_diag((_outer(dv, gv) + _outer(gv, dv)) / 6.0, 0.0)
    k4 = (omega, k22)
    k31 = _fill_diag((P.regen_use31 * 0.5 * lam)[:, None, None] * _col(dv) * rho, 0.0)
    if t_iij is not None:
        k2 = _fill_diag(_col(P.b2 * tdiag) * t_iij, 0.0)
        k31 = k31 + k2
    c2p_tail = (2.0 * var * phi)[None] * he_stack[:kmax + 2]
    c2p = torch.cat([torch.stack([e2_0, 2.0 * sigma * c[0], 2.0 * var * Phi]), c2p_tail])
    c_head = torch.cat([torch.stack(c[:2]), c_tail[:kmax + 3]])
    coefficients = torch.stack((c_head, c2p), dim=1)  # (K, 2, B, n)
    pair = pair_terms_batched(coefficients[:, [0, 1, 1]], coefficients[:, [0, 0, 1]],
                              R, kmax, t_iij, t_ijj, tdiag, k4, k31)
    Cn, cov21, cov22 = pair[0], pair[1], pair[2]
    Cn = _fill_diag(Cn, var_post)
    C = Cn
    post_k22 = _fill_diag(
        cov22 - 2.0 * _col(mu) * _tr(cov21) - 2.0 * _row(mu) * cov21
        + 4.0 * _outer(mu, mu) * Cn - 2.0 * Cn * Cn, 0.0)
    post_k4 = (raw_ez4 - 4.0 * raw_mu * raw_ez3 + 6.0 * raw_mean_sq * raw_ez2
               - 3.0 * raw_mean_sq * raw_mean_sq - 3.0 * raw_var ** 2)
    H = post_k4 + post_k22.sum(2).to(F64)
    core_diag = ((6.0 / (n + 4)) * H
                 - (3.0 / ((n + 2) * (n + 4))) * H.sum(1, keepdim=True)).to(F32)
    var_prev = var_post

    # --- 同一添字の 3 次切り口の補正 Δ と k3 脚 ---
    w_next = weights[:, layer + 1]
    wT = _tr(w_next)
    B3 = None
    if t_iij is not None:
        B3 = _fill_diag(0.5 * (_outer(c[2], c[2]) * t_iij + _outer(c[1], c[3]) * t_ijj), 0.0)
        Yq = B3 @ w_next[..., :r3]
        Yq = B3 @ (_tr(B3) @ Yq)
        Q3, _ = torch.linalg.qr(Yq)
        R3_new = _tr(Q3) @ B3
        B3 = Q3 @ R3_new
        X3q_new = wT @ Q3
    else:
        X3q_new = torch.zeros(Bsz, n, r3, dtype=F32, device=dev)
        R3_new = torch.zeros(Bsz, r3, n, dtype=F32, device=dev)
    g_new = (lam * P.regen_feed / 24.0)[:, None] * c[1]
    u_new = metric * c[2] / var
    u = cov21 - 2.0 * _col(mu) * Cn
    k3d = (raw_ez3 - 3.0 * raw_mu * raw_ez2 + 2.0 * raw_mu ** 3).to(F32)
    u_src = 2.0 * _outer(c[1] * c[2], c[1]) * rho + _outer(c[1] * c[1], c[2]) * (rho * rho)
    k3_src = 3.0 * c[2] * c[1] * c[1]
    if B3 is not None:
        u_src = u_src + 2.0 * B3 * (_col(c[1]) * rho) + 2.0 * _col(c[1]) * _tr(B3)
    Kb = _outer(c[1], g_new) * rho
    Kb = _fill_diag(Kb + _tr(Kb), 0.0)
    u_src = u_src + 2.0 * _col(u_new) * Kb
    if D21_raw is not None:
        u_src = u_src + _outer(Phi * Phi, Phi) * D21_raw
        k3_src = k3_src + Phi * Phi * Phi * D3_raw
    delta = _fill_diag(u - u_src, 0.0)
    base = _fill_diag(rho * _row(c[1]), 0.0)
    e_new = (delta * base).sum(2) / torch.clamp((base * base).sum(2), min=1e-30)
    dd_new = (k3d - k3_src) / 3.0 - c[1] * e_new
    resid = delta - _col(e_new) * base
    Yr = resid @ w_next[..., :rank]
    Yr = resid @ (_tr(resid) @ Yr)
    Q_new, _ = torch.linalg.qr(Yr)
    Z_new = wT @ _tr(_tr(Q_new) @ resid)

    # --- 家族の輸送と誕生 ---
    if nf >= window:
        raise NotImplementedError("window expiry is not reachable for depth <= window")
    if nf > 0:
        X3q_w = wT @ (_col(Phi) * X3q_w)
        Z_w = wT @ (_col(Phi) * Z_w)
    ru = 0 if U is None else U.shape[2]
    cols = nf * n
    parts = []
    if ru:
        parts.append(_col(Phi) * U)
    if cols:
        parts.append(_col(Phi) * A_w)
    parts.append(_col(c[1]) * rho)
    if cols:
        parts.append(_col(Phi) * M_w)
    legs = torch.cat((wT @ torch.cat(parts, dim=2), wT), dim=2)
    if ru:
        U = legs[..., :ru]
    A_w = legs[..., ru:ru + cols + n]
    M_w = legs[..., ru + cols + n:]
    A_born = legs[..., ru + cols:ru + cols + n]
    if nf == 0:
        c1_w, g_w, u_w = c[1], g_new, u_new
        c2_w = c[2]
        X3q_w, R3_s = X3q_new, R3_new[:, None]
        e_w, dd_w = e_new, dd_new
        Z_w, Q_s = Z_new, Q_new[:, None]
    else:
        c1_w = torch.cat([c1_w, c[1]], 1)
        g_w = torch.cat([g_w, g_new], 1)
        u_w = torch.cat([u_w, u_new], 1)
        X3q_w = torch.cat([X3q_w, X3q_new], 2)
        R3_s = torch.cat([R3_s, R3_new[:, None]], 1)
        c2_w = torch.cat([c2_w, c[2]], 1)
        e_w = torch.cat([e_w, e_new], 1)
        dd_w = torch.cat([dd_w, dd_new], 1)
        Z_w = torch.cat([Z_w, Z_new], 2)
        Q_s = torch.cat([Q_s, Q_new[:, None]], 1)
    nf += 1
    if nf > cfg.age_old:
        A_f, M_f, c2_f = A_w[..., :n], M_w[..., :n], c2_w[:, :n]
        A_w, M_w, c2_w = A_w[..., n:], M_w[..., n:], c2_w[:, n:]
        nf -= 1
        if U is None:
            Yb = M_f @ w_next[..., n - r_old:]
            Yb = M_f @ (_tr(M_f) @ Yb)
            U_new, _ = torch.linalg.qr(Yb)
        else:
            U_new, Tm = torch.linalg.qr(U)
            FA_o, FM_o = Tm @ FA_o, Tm @ FM_o
            FZ_o, FX3q_o = Tm @ FZ_o, Tm @ FX3q_o
        UT = _tr(U_new)
        fa, fm = UT @ A_f, UT @ M_f
        fz = UT @ Z_w[..., :rank]
        fx3q = UT @ X3q_w[..., :r3]
        if U is None:
            FA_o, FM_o, c2_o = fa, fm, c2_f
            e_o, dd_o = e_w[:, :n], dd_w[:, :n]
            FZ_o, Q_o = fz, Q_s[:, :1]
            FX3q_o, R3_o = fx3q, R3_s[:, :1]
            c1_o, g_o, u_o = c1_w[:, :n], g_w[:, :n], u_w[:, :n]
        else:
            FA_o = torch.cat([FA_o, fa], 2)
            FM_o = torch.cat([FM_o, fm], 2)
            c2_o = torch.cat([c2_o, c2_f], 1)
            e_o = torch.cat([e_o, e_w[:, :n]], 1)
            dd_o = torch.cat([dd_o, dd_w[:, :n]], 1)
            FZ_o = torch.cat([FZ_o, fz], 2)
            Q_o = torch.cat([Q_o, Q_s[:, :1]], 1)
            FX3q_o = torch.cat([FX3q_o, fx3q], 2)
            R3_o = torch.cat([R3_o, R3_s[:, :1]], 1)
            c1_o = torch.cat([c1_o, c1_w[:, :n]], 1)
            g_o = torch.cat([g_o, g_w[:, :n]], 1)
            u_o = torch.cat([u_o, u_w[:, :n]], 1)
        e_w, dd_w = e_w[:, n:], dd_w[:, n:]
        Z_w, Q_s = Z_w[..., rank:], Q_s[:, 1:]
        X3q_w, R3_s = X3q_w[..., r3:], R3_s[:, 1:]
        c1_w, g_w, u_w = c1_w[:, n:], g_w[:, n:], u_w[:, n:]
        U = U_new
    out = {"row": mu}
    for key, value in zip(_STATE_KEYS, (
            mu, C, core_diag, var_prev, nf,
            A_w, M_w, c2_w, e_w, dd_w, Z_w, Q_s, c1_w, g_w, u_w, X3q_w, R3_s,
            U, FA_o, FM_o, FZ_o, c2_o, e_o, dd_o, Q_o, c1_o, g_o, u_o,
            FX3q_o, R3_o, A_born, c, post_k4, k3d)):
        out[key] = value
    return out


def predict(P: ChainParams, weights, cfg=ChainConfig, checkpoint=False):
    """weights: (B, depth, n, n) FP32(mlp.weights の向き)。返り値 (B, depth, n) FP32。

    checkpoint=True(かつ勾配計算中)なら層ごとに torch.utils.checkpoint で再計算し、保持する
    活性を層の状態だけにする。
    """
    Bsz, depth, n, _ = weights.shape
    dev = weights.device
    s = {key: None for key in _STATE_KEYS}
    s.update(
        mu=torch.zeros(Bsz, n, dtype=F32, device=dev),
        core_diag=torch.full((Bsz, n), -6.0 / (n + 2.0), dtype=F32, device=dev),
        var_prev=torch.ones(Bsz, n, dtype=F32, device=dev),
        nf=0,
    )
    rows = []
    use_ckpt = checkpoint and torch.is_grad_enabled()
    for layer in range(depth):
        if use_ckpt:
            s = torch.utils.checkpoint.checkpoint(_layer, P, cfg, weights, layer, s,
                                                  use_reentrant=False)
        else:
            s = _layer(P, cfg, weights, layer, s)
        rows.append(s["row"])
    c_n = math.sqrt(2.0 / n) * math.exp(math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0))
    return torch.stack(rows, dim=1) * c_n
