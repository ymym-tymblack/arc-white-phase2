"""研究コピーの補正器を DEV で対比較する(MLP ごとの最終層 MSE 比、改善数、調整後スコア)。

基準(補正なし)は results/ml_lab_dev_base_<lo>_<hi>.json にキャッシュする。
例: python research/ml_lab/eval_dev.py --idxs 0-63 --name omega --kwargs '{"omega_corrector": "research/ml_lab/omega_corrector.json"}'
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, os.environ.get("ML_LAB_THREADS", "4"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from eval_mini import Evaluator  # noqa: E402

ORACLE_PATH = str(ROOT / "research" / "ml_lab" / "oracle_estimator.py")
DEV = "/kaggle/ext/full_dev64"
B = 2**41


def summary(res):
    adj = [r["mse"] * max(0.1, r["flops"] / B) for r in res]
    return dict(adj=float(np.mean(adj)), mse=float(np.mean([r["mse"] for r in res])),
                util=float(np.mean([r["flops"] for r in res]) / B),
                resid=float(max(r["resid"] for r in res)), sec=float(max(r["sec"] for r in res)))


def boot_ci(x, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="0-63")
    ap.add_argument("--name", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--path", default=ORACLE_PATH, help="研究コピーのパス(基準のキャッシュもこれで分ける)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.idxs.split("-"))
    idxs = list(range(lo, hi + 1))
    ev = Evaluator(idxs, workers=a.workers, threads=a.threads, dataset_source=DEV, dataset_split="full")
    stem = Path(a.path).stem
    suffix = "" if stem == "oracle_estimator" else "_" + stem
    base_path = ROOT / "results" / f"ml_lab_dev_base{suffix}_{lo}_{hi}.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        base = ev.score(a.path)
        base_path.write_text(json.dumps(base))
    kwargs = json.loads(a.kwargs)
    cand = ev.score(a.path, **kwargs)
    b = np.array([r["mse"] for r in base["per"]])
    c = np.array([r["mse"] for r in cand["per"]])
    ratio = c / b
    sb, sc = summary(base["per"]), summary(cand["per"])
    out = dict(name=a.name, kwargs=kwargs, idxs=idxs, base=sb, cand=sc,
               mean_ratio=float(ratio.mean()), ratio_ci95=boot_ci(ratio),
               improved=int((ratio < 1).sum()), mse_reduction=float(1 - c.mean() / b.mean()),
               adj_ratio=sc["adj"] / sb["adj"],
               flops_overhead=float(np.mean([r["flops"] for r in cand["per"]])
                                    - np.mean([r["flops"] for r in base["per"]])),
               per=[dict(idx=i, base=float(x), cand=float(y), ratio=float(y / x),
                         base_layers=rb["layers"], cand_layers=rc["layers"])
                    for i, x, y, rb, rc in zip(idxs, b, c, base["per"], cand["per"])])
    (ROOT / "results" / f"ml_lab_dev_{a.name}_{lo}_{hi}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
