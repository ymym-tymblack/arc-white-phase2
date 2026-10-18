"""手順3: 補正器の学習データ。補正なしの鎖を trace=True で走らせ、層ごとの鎖の周辺量と
特徴を真値(N=1e9)と一緒に /kaggle/ext/ml_lab_feats/row_XXXXX.npz に保存する。

例: python research/ml_lab/collect_features.py --dataset /kaggle/ext/compute_full64 --offset 0 --idxs 0-63
    python research/ml_lab/collect_features.py --dataset /kaggle/ext/full_train --rows-json /kaggle/ext/full_train/ml_lab_rows.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ORACLE_PATH, Runner, load_truth, run_one  # noqa: E402

FEATS = Path("/kaggle/ext/ml_lab_feats")
KEYS = ("m", "var", "tdiag", "omega", "metric", "dG", "tg", "tv",
        "star", "two", "lin", "k31", "diag4", "part", "tri", "diag3", "q22", "q21")

def save(res, path_tag=""):
    row = res["row"]
    truth = load_truth(row)
    depth = len(res["trace"])
    n = res["trace"][0]["m"].shape[0]
    out = {}
    for k in KEYS:
        arr = np.zeros((depth, n))
        for layer, r in enumerate(res["trace"]):
            v = r.get(k)
            if v is not None:
                arr[layer] = np.broadcast_to(v, (n,))
        out["chain_" + k] = arr
    out["lam"] = np.array([float(r["lam"]) for r in res["trace"]])
    for k, v in truth.items():
        out["true_" + k] = v
    out["pred"] = res["pred"]
    np.savez(FEATS / f"row_{row:05d}{path_tag}.npz", **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--idxs", default=None)
    ap.add_argument("--rows-json", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--path", default=ORACLE_PATH)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--every", type=int, default=1, help="対象の k 個おきだけを集める")
    a = ap.parse_args()
    FEATS.mkdir(parents=True, exist_ok=True)
    if a.rows_json:
        rows = json.loads(Path(a.rows_json).read_text())["rows"]
        pairs = list(enumerate(rows))
    else:
        lo, hi = (int(x) for x in a.idxs.split("-"))
        pairs = [(i, i + a.offset) for i in range(lo, hi + 1)]
    pairs = [(i, r) for i, r in pairs[::a.every]
             if not (FEATS / f"row_{r:05d}{a.tag}.npz").exists()]
    assert all(not (128 <= r < 256) for _, r in pairs), "FINAL rows must never be used"
    runner = Runner(a.workers, a.threads, dataset=a.dataset)
    kwargs = json.loads(a.kwargs)
    jobs = [(a.path, i, r, {}, kwargs, True) for i, r in pairs]
    for res in runner.pool.map(run_one, jobs):
        if "error" in res:
            print("error", res["row"], res["error"], flush=True)
            continue
        save(res, a.tag)
        print("saved", res["row"], f"mse={res['mse']:.3e}", f"sec={res['sec']:.1f}", flush=True)


if __name__ == "__main__":
    main()
