"""段階5: K=3 推定器の内部量と残差(正解 - 予測)を mini 100 MLP 全部で集める。"""
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "research")
from eval_mini import ROOT, _init, _load_mlp  # noqa: E402


def _collect(idx):
    import importlib.util
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    mlp, truth = _load_mlp(idx)
    spec = importlib.util.spec_from_file_location("cand_k3", str(ROOT / "estimator.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator(collect=True)
    budget = 2**41
    est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=budget,
                           api_version="1.0", submission_dir=str(ROOT), seed=0))
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
        pred = est.predict(mlp, budget)
    pred = np.asarray(pred, dtype=np.float64)
    # 全層で定義される配列だけを保存し、辞書のcomputeや途中層だけのκ4診断を混ぜない。
    keys = ("m", "sigma", "alpha", "tdiag", "mu", "mu_gauss", "var_post", "D3", "D21")
    feats = {k: np.stack([np.asarray(f[k], dtype=np.float64) for f in est.features]) for k in keys}
    w = [np.asarray(x, dtype=np.float64) for x in mlp.weights]
    wnorm2 = np.stack([(x**2).sum(0) for x in w])  # (depth, n)
    wsum = np.stack([x.sum(0) for x in w])
    return dict(idx=idx, pred=pred, truth=truth, flops=int(ctx.flops_used),
                wnorm2=wnorm2, wsum=wsum, **{f"f_{k}": v for k, v in feats.items()})


if __name__ == "__main__":
    import numpy as np
    pool = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                               initializer=_init, initargs=(8,))
    t = time.time()
    res = list(pool.map(_collect, range(100)))
    keys = [k for k in res[0] if k != "idx"]
    np.savez_compressed("/tmp/stage5.npz", **{k: np.stack([r[k] for r in res]) for k in keys})
    print("collected", len(res), f"{time.time()-t:.0f}s", keys)
