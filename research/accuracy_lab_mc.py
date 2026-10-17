"""球面入力分解 (accuracy_lab の sphere_input) を GPU の Monte Carlo で検算する。

A. 恒等式 E[h_l(x)] = c_n E[h_l(√n u)](x~N(0,I)、u=x/|x|、同じ乱数で比較)。
B. 層0の事前活性化 z=Wᵀ√n u の標準化κ4: κ̃_iiii=g、κ̃_iijj=g/3+(2g/3)ρ²、κ̃_iiij=gρ。
C. 層0の ReLU 後の 3 次キュムラント(添字すべて異なる)の球面−Gauss 差を
   P1=Σ c1_i c1_j c2_k ρ_ij/2、P2=Σ c1_i c1_j c2_k ρ_ik ρ_jk/2 へ回帰し、係数 g/3, 2g/3 を確かめる。
   推定器の誕生項は λ·feed/6 · P1 なので、λ=g なら厳密な feed は 2(λ=2g なら 1)。
例: research/gpu_env.sh research/accuracy_lab_mc.py --samples 100000000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

DEV = "cuda"


def c_n(n):
    return math.sqrt(2.0 / n) * math.exp(math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0))


def draw(n, chunk, gen):
    x = torch.randn(chunk, n, device=DEV, generator=gen, dtype=torch.float64)
    u = x * (math.sqrt(n) / x.norm(dim=1, keepdim=True))
    return x, u


def check_identity(n, depth, samples, chunk, seed):
    rng = np.random.default_rng(seed)
    W = [torch.tensor(rng.standard_normal((n, n)) * math.sqrt(2.0 / n), device=DEV) for _ in range(depth)]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    sg = torch.zeros(depth, n, device=DEV, dtype=torch.float64)
    ss = torch.zeros_like(sg)
    sd = torch.zeros_like(sg)  # 差の二乗和(同じ乱数の差の標準誤差)
    cn = c_n(n)
    done = 0
    while done < samples:
        x, u = draw(n, chunk, gen)
        for layer, w in enumerate(W):
            x = torch.relu(x @ w)
            u = torch.relu(u @ w)
            sg[layer] += x.sum(0)
            ss[layer] += u.sum(0)
            sd[layer] += ((x - cn * u) ** 2).sum(0)
        done += chunk
    mg, ms = sg / done, cn * ss / done
    se = torch.sqrt((sd / done - (mg - ms) ** 2) / done)
    z = ((mg - ms) / se).abs()
    # 深さ1は解析的に E = |w|/√(2π)
    exact1 = W[0].norm(dim=0) / math.sqrt(2 * math.pi)
    return dict(n=n, depth=depth, samples=done, c_n=cn,
                max_abs_diff=[float((mg[layer] - ms[layer]).abs().max()) for layer in range(depth)],
                max_z=[float(z[layer].max()) for layer in range(depth)],
                mean_abs_z=[float(z[layer].mean()) for layer in range(depth)],
                layer0_gauss_vs_exact=float((mg[0] - exact1).abs().max()),
                layer0_sphere_vs_exact=float((ms[0] - exact1).abs().max()))


def correlated_weights(n, seed, share=0.5):
    """ρ が O(0.3) 程度になるよう共通成分を混ぜた列正規化重み。"""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)) + share * rng.standard_normal((n, 1)) * np.sqrt(n) / 2
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return torch.tensor(W, device=DEV)


def check_cumulants(n, samples, chunk, seed):
    W = correlated_weights(n, seed)
    S = W.T @ W
    sig = S.diag().sqrt()
    rho = S / torch.outer(sig, sig)
    g = -6.0 / (n + 2)
    gen = torch.Generator(device=DEV).manual_seed(seed + 7)
    E22 = torch.zeros(n, n, device=DEV, dtype=torch.float64)
    E31 = torch.zeros_like(E22)
    # ReLU 後の 1〜3 次モーメント(Gauss と球面)
    m1 = {k: torch.zeros(n, device=DEV, dtype=torch.float64) for k in "gs"}
    m2 = {k: torch.zeros(n, n, device=DEV, dtype=torch.float64) for k in "gs"}
    m3 = {k: torch.zeros(n, n, n, device=DEV, dtype=torch.float64) for k in "gs"}
    done = 0
    while done < samples:
        x, u = draw(n, chunk, gen)
        zs = u @ W
        z2 = zs * zs
        E22 += z2.T @ z2
        E31 += (z2 * zs).T @ zs
        for key, z in (("g", x @ W), ("s", zs)):
            h = torch.relu(z)
            m1[key] += h.sum(0)
            m2[key] += h.T @ h
            m3[key] += torch.einsum("bi,bj,bk->ijk", h, h, h)
        done += chunk
    E22 /= done
    E31 /= done
    k22 = E22 - (torch.outer(S.diag(), S.diag()) + 2 * S * S)
    k31 = E31 - 3 * S.diag()[:, None] * S
    vv = torch.outer(sig * sig, sig * sig)
    k22s = k22 / vv
    k31s = k31 / (sig[:, None] ** 3 * sig[None, :])
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    pred22 = g / 3 + (2 * g / 3) * rho * rho
    pred31 = g * rho
    kurt = k22s.diag()

    def cum3(key):
        a, b, c = m1[key] / done, m2[key] / done, m3[key] / done
        sym = (torch.einsum("ij,k->ijk", b, a) + torch.einsum("ik,j->ijk", b, a)
               + torch.einsum("jk,i->ijk", b, a))
        return c - sym + 2 * torch.einsum("i,j,k->ijk", a, a, a)

    diff = cum3("s") - cum3("g")
    # Hermite 係数(α=0、σ=sig)
    c1 = sig * 0.5
    c2 = sig / math.sqrt(2 * math.pi)
    p1 = 0.5 * (torch.einsum("i,j,k,ij->ijk", c1, c1, c2, rho)
                + torch.einsum("i,k,j,ik->ijk", c1, c1, c2, rho)
                + torch.einsum("j,k,i,jk->ijk", c1, c1, c2, rho))
    p2 = 0.5 * (torch.einsum("i,j,k,ik,jk->ijk", c1, c1, c2, rho, rho)
                + torch.einsum("i,k,j,ij,jk->ijk", c1, c1, c2, rho, rho)
                + torch.einsum("j,k,i,ij,ik->ijk", c1, c1, c2, rho, rho))
    idx = torch.arange(n, device=DEV)
    mask = (idx[:, None, None] != idx[None, :, None]) & (idx[:, None, None] != idx[None, None, :]) & (
        idx[None, :, None] != idx[None, None, :])
    X = torch.stack((p1[mask], p2[mask]), 1)
    y = diff[mask]
    coef = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    resid = y - X @ coef
    r2 = 1 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    a1 = float(coef[0])
    return dict(
        n=n, samples=done, g=g, rho_offdiag_abs_mean=float(rho[off].abs().mean()),
        kurt_mean=float(kurt.mean()), kurt_max_abs_err=float((kurt - g).abs().max()),
        k22_max_abs_err=float((k22s - pred22)[off].abs().max()),
        k22_rms=float(k22s[off].pow(2).mean().sqrt()),
        k22_err_if_only_g3=float((k22s - g / 3)[off].abs().max()),
        k31_max_abs_err=float((k31s - pred31)[off].abs().max()),
        k31_slope_vs_g_rho=float((k31s[off] * rho[off]).sum() / (rho[off] ** 2).sum() / g),
        kappa3_fit_coef_p1_over_g=a1 / g, kappa3_fit_coef_p2_over_g=float(coef[1]) / g,
        kappa3_fit_r2=r2,
        exact_feed_at_lambda_g=6.0 * a1 / g,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100_000_000)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--output", default="results/accuracy_lab_mc.json")
    args = ap.parse_args()
    out = dict(
        identity=[check_identity(64, 2, args.samples, args.chunk, 1),
                  check_identity(64, 4, args.samples, args.chunk, 2)],
        cumulants=[check_cumulants(16, args.samples, args.chunk // 4, 3),
                   check_cumulants(32, args.samples, args.chunk // 8, 4)],
    )
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
