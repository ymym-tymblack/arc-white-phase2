"""flopscope の推定器で開発集合の全層予測を保存する(PyTorch 移植の照合用。repo 環境で実行)。

例: uv run python research/torch_port/ref_dump.py research/frozen_phase2/combined.py \
        --indices 0-3 --workers 4 --threads 2 --out /kaggle/ext/torch_port_ref/combined
    --kwargs '{"strassen_levels": 0}' で Estimator の属性を上書きできる。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_mini  # noqa: E402

DEV = "/kaggle/ext/full_dev64"


def _job(args):
    path, kwargs, idx, out = args
    import importlib.util
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    mlp, truth = eval_mini._load_mlp(idx)
    spec = importlib.util.spec_from_file_location("cand_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator(**kwargs)
    budget = 2**41
    est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=budget,
                           api_version="1.0", submission_dir=str(eval_mini.ROOT), seed=0))
    t = time.time()
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=600.0, quiet=True) as ctx:
        pred = est.predict(mlp, budget)
    pred = np.asarray(pred, dtype=np.float64)
    np.save(Path(out) / f"dev_{idx:02d}.npy", pred)
    mse = float(((pred[-1] - truth[-1]) ** 2).mean())
    return dict(idx=idx, mse=mse, flops=int(ctx.flops_used), sec=time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--indices", default="0-3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.indices.split("-"))
    ident = eval_mini.dataset_identity(DEV, "full")
    pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                               initializer=eval_mini._init, initargs=(a.threads, ident))
    kwargs = json.loads(a.kwargs)
    jobs = [(str(a.candidate.resolve()), kwargs, i, str(a.out)) for i in range(lo, hi + 1)]
    res = list(pool.map(_job, jobs))
    (a.out / f"summary_{lo}_{hi}.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
