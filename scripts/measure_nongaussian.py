"""非ガウス性(κ3, κ4)の伝播が成り立つか、効果の上限はどれくらいかを測る。

検証1: ニューロン間の連動(交差キュムラント)を無視した独立和の近似
         κ3(z'_a) = sum_k W[k,a]^3 κ3(h_k),  κ4(z'_a) = sum_k W[k,a]^4 κ4(h_k)
       が、MC で測った z' の κ3, κ4 と合うか(層ごと、ニューロン間の相関と平均比)。
       h の κ は (A) MC 実測、(B) 実測 (mu_z, sigma_z) のガウスから閉形式、の2通りで入れる。

検証2: 効果の上限。各ニューロンの E[ReLU(z)] を次の4通りで計算し、MC 平均との MSE を比べる。
         E0: 推定器の出力(参照ノイズ控除済み)
         E1: 実測 (mu_z, sigma_z) をガウスに入れた閉形式
         E2: E1 + Edgeworth 補正(実測 γ1, γ2)       ← κ を完全に当てたときの上限
         E3: E1 + Edgeworth 補正(独立和で予測した γ1, γ2) ← 提案方式
       E1-E3 は MC 平均と同じ標本から作るので、標本ノイズが大きく相殺される(層1で確認)。
       E1 が E0 より十分小さければ、誤差の主因は周辺分布の非ガウス性ではなく (mu_z, sigma_z) の伝播側にある。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from local_engine import build_mlp  # noqa: E402
from measure_layer_error import load_estimator, run_estimator  # noqa: E402

_erfc = np.frompyfunc(math.erfc, 1, 1)


def ncdf(x):
    return 0.5 * _erfc(-x / math.sqrt(2.0)).astype(np.float64)


def npdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def accumulate(z, h, sz, sh):
    """z, h: (m, n) float32。べき和 k=1..4 を float64 で sz, sh (4, n) に加算。"""
    for arr, out in ((z, sz), (h, sh)):
        a2 = arr * arr
        out[0] += arr.sum(0, dtype=np.float64)
        out[1] += a2.sum(0, dtype=np.float64)
        out[2] += (a2 * arr).sum(0, dtype=np.float64)
        out[3] += (a2 * a2).sum(0, dtype=np.float64)


def mc_moments(weights, n: int, seed: int, chunk: int = 10_000):
    depth, width = len(weights), weights[0].shape[0]
    rng = np.random.default_rng(seed)
    sz = np.zeros((depth, 4, width))
    sh = np.zeros((depth, 4, width))
    done = 0
    with ThreadPoolExecutor(max_workers=depth) as ex:
        while done < n:
            m = min(chunk, n - done)
            x = rng.standard_normal((m, width), dtype=np.float32)
            zs, hs = [], []
            for w in weights:
                z = x @ w
                x = np.maximum(z, 0.0)
                zs.append(z)
                hs.append(x)
            list(ex.map(lambda l: accumulate(zs[l], hs[l], sz[l], sh[l]), range(depth)))
            done += m
    return sz / n, sh / n


def central(raw):
    """raw: (4, n) の 1..4 次の生モーメント → (mean, var, k3, k4)。"""
    m1, m2, m3, m4 = raw
    var = m2 - m1**2
    k3 = m3 - 3 * m1 * m2 + 2 * m1**3
    k4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4 - 3 * var**2
    return m1, var, k3, k4


def relu_gauss_cumulants(mu, sigma):
    """z~N(mu, sigma^2) のとき ReLU(z) の (mean, var, k3, k4)。I_n=E[z^n; z>0] の漸化式。"""
    a = mu / sigma
    P, p = ncdf(a), npdf(a)
    i = [P, mu * P + sigma * p]
    for n in range(2, 5):
        i.append(mu * i[n - 1] + sigma**2 * (n - 1) * i[n - 2])
    return central(np.stack(i[1:5]))


def edgeworth_mean(mu, sigma, g1, g2):
    a = mu / sigma
    P, p = ncdf(a), npdf(a)
    base = mu * P + sigma * p
    c3 = sigma * p * (-a)
    c4 = sigma * p * (a * a - 1)
    c6 = sigma * p * (a**4 - 6 * a * a + 3)
    return base, base + g1 / 6 * c3 + g2 / 24 * c4 + g1**2 / 72 * c6


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("-n", type=int, default=1_000_000)
    args = ap.parse_args()

    for seed in args.seeds:
        mlp = build_mlp(args.width, args.depth, seed)
        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        pred, _ = run_estimator(load_estimator(ROOT / "examples/03_covariance_propagation.py"), mlp)

        t0 = time.time()
        rz, rh = mc_moments(weights, args.n, seed + 1000)
        print(f"\n=== seed={seed} N={args.n:,} mc={time.time() - t0:.0f}s")
        print(
            f"{'L':>2} | {'E0 est':>9} {'E1 gauss':>9} {'E2 edge(実測κ)':>14} {'E3 edge(予測κ)':>14} |"
            f" {'g1 実測':>7} {'g1 予測A':>8} {'相関':>5} | {'g2 実測':>7} {'g2 予測A':>8} {'予測B':>7} {'相関A':>5}"
        )
        prev_k = None  # 前層の h の (k3, k4) 実測 / ガウス閉形式
        for l in range(args.depth):
            mu, var, k3, k4 = central(rz[l])
            sig = np.sqrt(var)
            g1, g2 = k3 / var**1.5, k4 / var**2
            hm, hv, hk3, hk4 = central(rh[l])
            noise = float(hv.mean() / args.n)

            if l == 0:
                pg1 = pg2 = pg2b = np.zeros_like(g1)
            else:
                w3 = weights[l].astype(np.float64) ** 3
                w4 = weights[l].astype(np.float64) ** 4
                pg1 = (prev_k["meas"][0] @ w3) / var**1.5
                pg2 = (prev_k["meas"][1] @ w4) / var**2
                pg2b = (prev_k["gauss"][1] @ w4) / var**2
            prev_k = {
                "meas": (hk3, hk4),
                "gauss": relu_gauss_cumulants(mu, sig)[2:],
            }
            base, e2 = edgeworth_mean(mu, sig, g1, g2)
            _, e3 = edgeworth_mean(mu, sig, pg1, pg2)
            mse = lambda x: float(np.mean((x - hm) ** 2))  # noqa: E731
            e0 = float(np.mean((pred[l] - hm) ** 2)) - noise
            print(
                f"{l + 1:>2} | {e0:>9.2e} {mse(base):>9.2e} {mse(e2):>14.2e} {mse(e3):>14.2e} |"
                f" {g1.mean():>7.3f} {pg1.mean():>8.3f} {corr(g1, pg1) if l else float('nan'):>5.2f} |"
                f" {g2.mean():>7.3f} {pg2.mean():>8.3f} {pg2b.mean():>7.3f} {corr(g2, pg2) if l else float('nan'):>5.2f}"
            )


if __name__ == "__main__":
    main()
