"""Keenan Pepper の N=1e9 MC モーメント(HF)から周辺量だけを範囲読みして保存する。

対の 4 次配列(337 MB)は読まない。npz は非圧縮 zip なので、HfFileSystem の
シーク可能なハンドルに np.load すると、参照したメンバーだけを取得する。

保存先: /kaggle/ext/ml_lab_truth/row_{global_index:05d}.npz
  pre_mean, pre_var, pre_k3, pre_k4: (16, n) 事前活性化 z_l の平均・分散・κ3・κ4
  post_mean, gt_mean: (16, n)

使い方: python research/ml_lab/fetch_truth.py START COUNT [--check DATASET_DIR LOCAL_OFFSET]
  --check を付けると、ローカルの whestbench データセットの all_layer_means と gt_mean を照合する
  (DEV は full_dev64 の局所番号 = 公式行 - 64)。
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem

REPO = "datasets/keenanpepper/arc-whestbench-p2-full1000-N1e9"
OUT = Path("/kaggle/ext/ml_lab_truth")


def cumulants(m1, m2, m3, m4):
    """raw モーメント E[z^p] → 平均・分散・κ3・κ4。"""
    mu = m1
    c2 = m2 - mu * mu
    c3 = m3 - 3 * mu * m2 + 2 * mu**3
    c4 = m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 3 * mu**4
    k4 = c4 - 3 * c2 * c2
    return mu, c2, c3, k4


def fetch(row: int) -> Path:
    path = OUT / f"row_{row:05d}.npz"
    if path.exists():
        return path
    fs = HfFileSystem()
    with fs.open(f"{REPO}/moments/mlp_{row:05d}.npz", "rb", block_size=1 << 20) as f:
        z = np.load(f)
        assert int(z["global_index"]) == row, (row, int(z["global_index"]))
        raw = [np.asarray(z[k], dtype=np.float64) for k in ("pre_mean", "pre_m2", "pre_m3", "pre_m4")]
        post_mean = np.asarray(z["mean"], dtype=np.float64)
        gt = np.asarray(z["gt_mean"], dtype=np.float64)
    mu, var, k3, k4 = cumulants(*raw)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, pre_mean=mu, pre_var=var, pre_k3=k3, pre_k4=k4, post_mean=post_mean, gt_mean=gt)
    tmp.rename(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int)
    ap.add_argument("count", type=int)
    ap.add_argument("--check", default=None)
    ap.add_argument("--offset", type=int, default=0, help="ローカル番号 = 公式行 - offset")
    ap.add_argument("--report", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(range(a.start, a.start + a.count))
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(fetch, rows))
    report = {"rows": rows}
    if a.check:
        from whestbench import load_dataset
        ds = load_dataset(a.check, split="full")
        checks = []
        for row in rows:
            z = np.load(OUT / f"row_{row:05d}.npz")
            local = np.asarray(ds[row - a.offset]["all_layer_means"], dtype=np.float64)
            checks.append(dict(
                row=row, local=row - a.offset,
                gt_vs_local_maxabs=float(np.abs(z["gt_mean"] - local).max()),
                mc_vs_local_rms=float(np.sqrt(((z["post_mean"] - local) ** 2).mean())),
            ))
            print(checks[-1])
        report["checks"] = checks
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
