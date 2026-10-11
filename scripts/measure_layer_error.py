"""層ごとの推定誤差と、共分散固有値の減衰・非ガウス性を測る計測スクリプト。

`examples/03_covariance_propagation.py` の Estimator を flopscope 上でそのまま実行し、
同じ MLP を NumPy(float32, BLAS 全スレッド)で多数サンプルして層ごとに比較する。
ここでの MC は評価用の参照値なので、課金対象外(提出物には含まれない)。

列の意味(1 MLP, 層ごと):
  mse      : 推定器と MC 平均の MSE
  noise    : MC 参照の標準誤差² = mean(Var[h]) / N_mean (参照ノイズ)
  debiased : mse - noise (推定器自身の誤差の見積り。負なら参照ノイズ以下)
  sig2     : mean(MC平均²) (値の大きさ。rel = debiased / sig2)
  skew     : 事前活性化 z の |歪み| の neuron 平均 (ガウスなら 0)
  kurt     : 事前活性化 z の超過尖度の neuron 平均 (ガウスなら 0)
  topR     : ReLU 後の中心化共分散の固有値のうち上位 R 個が占める分散の割合
  pr       : 参与率 (tr C)^2 / tr(C^2) (実効ランクの目安)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import flopscope as flops  # noqa: E402
from whestbench import SetupContext  # noqa: E402

from local_engine import build_mlp  # noqa: E402

RANKS = (8, 32, 64, 128, 256)


def load_estimator(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module.Estimator()


def run_estimator(est, mlp, budget: int = 2**41):
    est.setup(
        SetupContext(
            width=mlp.width,
            depth=mlp.depth,
            flop_budget=budget,
            api_version="1.0",
            submission_dir=str(ROOT),
            seed=0,
        )
    )
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
        pred = est.predict(mlp, budget)
    est.teardown()
    return np.asarray(pred, dtype=np.float64), int(ctx.flops_used)


def mc_reference(weights, n_mean: int, n_cov: int, seed: int, chunk: int = 25_000):
    """全 n_mean 個で層ごとの平均・分散を、先頭 n_cov 個で共分散と z の 1〜4 次モーメントを集める。"""
    depth, width = len(weights), weights[0].shape[0]
    rng = np.random.default_rng(seed)
    s1 = np.zeros((depth, width))  # sum h        (n_mean 個)
    s2 = np.zeros((depth, width))  # sum h^2      (n_mean 個)
    c1 = np.zeros((depth, width))  # sum h        (先頭 n_cov 個)
    hth = np.zeros((depth, width, width))  # sum h h^T (先頭 n_cov 個)
    zm = np.zeros((depth, 4, width))  # sum z^k, k=1..4 (先頭 n_cov 個)
    done = 0
    while done < n_mean:
        m = min(chunk, n_mean - done)
        use_cov = done + m <= n_cov
        x = rng.standard_normal((m, width), dtype=np.float32)
        for l, w in enumerate(weights):
            z = x @ w
            x = np.maximum(z, 0.0)
            s1[l] += x.sum(0, dtype=np.float64)
            s2[l] += np.einsum("ij,ij->j", x, x, dtype=np.float64)
            if use_cov:
                c1[l] += x.sum(0, dtype=np.float64)
                hth[l] += (x.T @ x).astype(np.float64)
                z64 = z.astype(np.float64)
                p = z64.copy()
                for k in range(4):
                    zm[l, k] += p.sum(0)
                    p *= z64
        done += m
    mean = s1 / n_mean
    var = s2 / n_mean - mean**2
    return mean, var, c1, hth, zm


def measure(seed: int, args) -> list[dict]:
    mlp = build_mlp(args.width, args.depth, seed)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]

    t0 = time.time()
    pred, est_flops = run_estimator(load_estimator(ROOT / args.estimator), mlp)
    t1 = time.time()
    mean, var, c1, hth, zm = mc_reference(weights, args.n_mean, args.n_cov, seed + 1000)
    t2 = time.time()
    print(
        f"\n=== seed={seed} width={args.width} depth={args.depth} "
        f"estimator_flops={est_flops:,} ({est_flops / 2**41:.2%} of 2^41) "
        f"est={t1 - t0:.0f}s mc={t2 - t1:.0f}s N_mean={args.n_mean:,} N_cov={args.n_cov:,}"
    )
    print(
        f"{'L':>2} {'mse':>10} {'noise':>10} {'debiased':>10} {'sig2':>8} {'rel':>9} "
        f"{'skew':>6} {'kurt':>6} " + " ".join(f"top{r:<4}" for r in RANKS) + f" {'pr':>6}"
    )
    out = []
    for l in range(args.depth):
        mse = float(np.mean((pred[l] - mean[l]) ** 2))
        noise = float(np.mean(var[l]) / args.n_mean)
        deb = mse - noise
        sig2 = float(np.mean(mean[l] ** 2))

        m1, m2, m3, m4 = (zm[l, k] / args.n_cov for k in range(4))
        c2 = m2 - m1**2
        c3 = m3 - 3 * m1 * m2 + 2 * m1**3
        c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
        skew = float(np.mean(np.abs(c3 / c2**1.5)))
        kurt = float(np.mean(c4 / c2**2 - 3))

        mu = c1[l] / args.n_cov
        cov = hth[l] / args.n_cov - np.outer(mu, mu)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(min=0)
        tr = ev.sum()
        top = {r: float(ev[:r].sum() / tr) for r in RANKS}
        pr = float(tr**2 / (ev**2).sum())

        print(
            f"{l + 1:>2} {mse:>10.3e} {noise:>10.3e} {deb:>10.3e} {sig2:>8.4f} {deb / sig2:>9.2e} "
            f"{skew:>6.3f} {kurt:>6.3f} "
            + " ".join(f"{top[r]:>7.3f}" for r in RANKS)
            + f" {pr:>6.1f}"
        )
        out.append(
            dict(seed=seed, layer=l + 1, mse=mse, noise=noise, debiased=deb, sig2=sig2,
                 skew=skew, kurt=kurt, pr=pr, **{f"top{r}": v for r, v in top.items()})
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--n-mean", type=int, default=2_000_000)
    ap.add_argument("--n-cov", type=int, default=200_000)
    ap.add_argument("--estimator", default="examples/03_covariance_propagation.py")
    ap.add_argument("--out", default=None, help="JSON 出力先(省略可)")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        results += measure(seed, args)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
