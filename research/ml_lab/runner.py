"""研究コピー(oracle_estimator.py)を MLP ごとに異なる差し込み設定で並列評価する。

eval_mini の _init/_load_mlp を流用し、真値は fetch_truth.py が保存した
/kaggle/ext/ml_lab_truth/row_XXXXX.npz(公式行番号)から読む。
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import eval_mini  # noqa: E402

TRUTH = Path("/kaggle/ext/ml_lab_truth")
ORACLE_PATH = str(ROOT / "research" / "ml_lab" / "oracle_estimator.py")
DEV = "/kaggle/ext/full_dev64"
DEV_OFFSET = 64  # full_dev64 の局所番号 0 = 公式行 64


def load_truth(row: int):
    import numpy as np
    z = np.load(TRUTH / f"row_{row:05d}.npz")
    var = z["pre_var"]
    return dict(
        m=z["pre_mean"], var=var,
        tdiag=z["pre_k3"] / var**1.5, omega=z["pre_k4"] / var**2,
    )


def build_oracle(spec: dict, truth: dict):
    """spec: {"omega": bool, "tdiag": bool, "mv_from": int|None, "k_from": int, "k_to": int}。"""
    if not spec:
        return None
    o = {"k_from": spec.get("k_from", 1)}
    if "k_to" in spec:
        o["k_to"] = spec["k_to"]
    if spec.get("omega"):
        o["omega"] = truth["omega"]
    if spec.get("omega_shape"):
        o["omega_shape"] = truth["omega"]
    if spec.get("omega_mean"):
        o["omega_mean"] = truth["omega"].mean(axis=1)
    if spec.get("tdiag"):
        o["tdiag"] = truth["tdiag"]
    if spec.get("mv_from") is not None:
        o["m"], o["var"], o["mv_from"] = truth["m"], truth["var"], spec["mv_from"]
    return o


def _module(path: str):
    key = "ml_lab_" + Path(path).stem
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(args):
    """(path, idx, row, spec, kwargs, trace) → 結果 dict。trace=True なら鎖の周辺量も返す。"""
    path, idx, row, spec, kwargs, trace = args
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    import flopscope as flops
    from whestbench import SetupContext

    mlp, truth_mean = eval_mini._load_mlp(idx)
    kw = dict(kwargs)
    if spec:
        kw["oracle"] = build_oracle(spec, load_truth(row))
    if trace:
        kw["trace"] = True
    est = _module(path).Estimator(**kw)
    budget = 2**41
    est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=budget,
                           api_version="1.0", submission_dir=str(ROOT), seed=0))
    t = time.time()
    try:
        with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=600.0, quiet=True) as ctx:
            pred = est.predict(mlp, budget)
    except Exception as exc:  # 差し込みで正値密度の分散が負になる等
        return dict(idx=idx, row=row, error=repr(exc))
    dt = time.time() - t
    pred = np.asarray(pred, dtype=np.float64)
    out = dict(
        idx=idx, row=row, mse=float(((pred[-1] - truth_mean[-1]) ** 2).mean()),
        layers=((pred - truth_mean) ** 2).mean(1).tolist(), flops=int(ctx.flops_used),
        sec=dt, resid=float(ctx.residual_wall_time_s),
    )
    if trace:
        rows = []
        for r in est.trace_rows:
            rows.append({k: (None if v is None else np.asarray(v, dtype=np.float64))
                         for k, v in r.items()})
        out["trace"] = rows
        out["pred"] = pred
    return out


class Runner:
    def __init__(self, workers=2, threads=4, dataset=DEV):
        ident = eval_mini.dataset_identity(dataset, "full")
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=eval_mini._init, initargs=(threads, ident))

    def map(self, jobs):
        return list(self.pool.map(run_one, jobs))
