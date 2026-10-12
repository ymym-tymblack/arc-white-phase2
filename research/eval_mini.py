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
    return dict(idx=idx, mse=mse, all_mse=all_mse, flops=used, sec=dt,
                resid=float(ctx.residual_wall_time_s),
                layers=((pred - truth) ** 2).mean(1).tolist())


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


if __name__ == "__main__":
    ev = Evaluator(range(4))
    t = time.time()
    r = ev.score(str(ROOT / "candidates/k3_factored.py"))
    print({k: v for k, v in r.items() if k != "per"}, f"{time.time()-t:.0f}s")
