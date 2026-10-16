"""mini データセット(N=1e9 の正解付き)で推定器を並列評価するハーネス。

whest run を介さず、MLP をプロセスごとにキャッシュして素早く多数の設定を試す。
スコアは公式と同じ: final_layer_mse × max(0.1, C/B)。C は flopscope の課金 FLOP。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_STATE = {}


def _init(threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))


def _load_mlp(idx: int):
    if idx in _STATE:
        return _STATE[idx]
    import numpy as np
    import whestbench
    from whestbench import load_dataset

    if "ds" not in _STATE:
        _STATE["ds"] = load_dataset(
            "aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini"
        )
    ds = _STATE["ds"]
    mlp = whestbench.mlp_at(ds, idx)
    truth = np.asarray(ds[idx]["all_layer_means"], dtype=np.float64)
    _STATE[idx] = (mlp, truth)
    return _STATE[idx]


def _run(args):
    path, kwargs, idx = args
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    mlp, truth = _load_mlp(idx)
    spec = importlib.util.spec_from_file_location("cand_" + Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator(**kwargs)
    budget = 2**41
    est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=budget,
                           api_version="1.0", submission_dir=str(ROOT), seed=0))
    t = time.time()
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
        pred = est.predict(mlp, budget)
    dt = time.time() - t
    pred = np.asarray(pred, dtype=np.float64)
    mse = float(((pred[-1] - truth[-1]) ** 2).mean())
    all_mse = float(((pred - truth) ** 2).mean())
    used = int(ctx.flops_used)
    p, r = pred[-1], truth[-1] - pred[-1]
    return dict(idx=idx, mse=mse, all_mse=all_mse, flops=used, sec=dt,
                resid=float(ctx.residual_wall_time_s),
                layers=((pred - truth) ** 2).mean(1).tolist(),
                # 縮小係数を解析的に当てはめるための最終層の和(計画3 改)
                spp=float(p @ p), spr=float(p @ r), srr=float(r @ r), n=int(p.size))


# 計画0 で固定した評価分割(mini の MLP 番号)。最終確認用は当てはめに使わない。
SPLITS = {
    "tune": list(range(0, 12)),
    "valid": list(range(12, 40)),
    "test": list(range(40, 100)),
}


class Evaluator:
    def __init__(self, idxs, workers: int = 4, threads: int = 8):
        self.idxs = list(idxs)
        # fork だと親が先に numpy を読み込んだ場合に BLAS のスレッド数が子へ引き継がれ
        # 過剰購読になる(実測で 9 倍遅くなった)。spawn で子ごとに環境変数を反映する。
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_init, initargs=(threads,))

    def score(self, path: str, **kwargs):
        res = list(self.pool.map(_run, [(path, kwargs, i) for i in self.idxs]))
        B = 2**41
        adj = [r["mse"] * max(0.1, r["flops"] / B) for r in res]
        return dict(
            adj=sum(adj) / len(adj),
            mse=sum(r["mse"] for r in res) / len(res),
            util=sum(r["flops"] for r in res) / len(res) / B,
            resid=max(r["resid"] for r in res),
            sec=max(r["sec"] for r in res),
            per=res,
        )

    def score_fit_calib(self, path: str, **kwargs):
        """縮小係数 calib を、この MLP 群で調整後スコアが最小になる値に解析的に決めて評価する。

        予測を (1-ε) 倍すると最終層の残差は r + ε p になるので、MLP m の MSE は
            MSE_m(ε) = (Σr² + 2ε Σpr + ε² Σp²) / n
        調整後スコアは Σ_m g_m MSE_m(ε)(g_m = max(0.1, C_m/B))で ε の 2 次式なので、
            ε* = -Σ_m g_m Σpr_m / Σ_m g_m Σp²_m
        が最小値を与える。探索の次元を 1 つ減らし、その分の雑音もなくす。
        """
        kwargs = dict(kwargs, calib=0.0)
        res = list(self.pool.map(_run, [(path, kwargs, i) for i in self.idxs]))
        B = 2**41
        g = [max(0.1, r["flops"] / B) / r["n"] for r in res]
        eps = -sum(gi * r["spr"] for gi, r in zip(g, res)) / sum(gi * r["spp"] for gi, r in zip(g, res))
        per_mse = [(r["srr"] + 2 * eps * r["spr"] + eps * eps * r["spp"]) / r["n"] for r in res]
        adj = [m * max(0.1, r["flops"] / B) for m, r in zip(per_mse, res)]
        return dict(
            adj=sum(adj) / len(adj), mse=sum(per_mse) / len(res), calib=eps,
            util=sum(r["flops"] for r in res) / len(res) / B,
            resid=max(r["resid"] for r in res), sec=max(r["sec"] for r in res),
            per=res, per_adj=adj,
        )


if __name__ == "__main__":
    ev = Evaluator(range(4))
    t = time.time()
    r = ev.score(str(ROOT / "estimator.py"))
    print({k: v for k, v in r.items() if k != "per"}, f"{time.time()-t:.0f}s")
