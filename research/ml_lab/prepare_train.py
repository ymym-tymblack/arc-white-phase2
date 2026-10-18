"""学習用に公式 full の行 ≥256 の MLP を取得する(入力サンプリングなし、予測評価なし)。

prepare_compute_eval.py は先頭からの行しか選べないので、シャードを選んで取得する版。
シャードの行数はファイルサイズから決まる(8 行 ≈ 538 MB、7 行 ≈ 471 MB、先頭から 8 行のシャードが並ぶ)。
最終評価用の行 128〜255 を決して含まないよう、開始行 ≥ --min-row のシャードだけを取り、
保存後に all_layer_means を Keenan の N=1e9 データの gt_mean と完全一致で照合して行番号を確定する
(照合に使う gt_mean は fetch_truth.py が保存した /kaggle/ext/ml_lab_truth のもの)。

例: python research/ml_lab/prepare_train.py --output /kaggle/ext/full_train --shards 33-52
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = "aicrowd/arc-whestbench-public-2026"
REVISION = "aa99830fdc09fad15407b10e8e3459d3e18bba0a"
ROW_BYTES = 67_247_540  # 1 MLP あたりの arrow バイト数(およそ)


def shard_table():
    from huggingface_hub import HfApi
    files = sorted(
        (f for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION,
                                           path_in_repo="prepared/full")
         if f.path.endswith(".arrow")),
        key=lambda f: f.path,
    )
    table, start = [], 0
    for f in files:
        rows = round(f.size / ROW_BYTES)
        table.append(dict(path=f.path, start=start, rows=rows))
        start += rows
    if start != 1000:
        raise ValueError(f"row count from shard sizes = {start}, expected 1000")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shards", required=True, help="例 33-52(両端含む)")
    ap.add_argument("--min-row", type=int, default=256)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.shards.split("-"))
    table = shard_table()
    chosen = table[lo:hi + 1]
    if any(s["start"] < a.min_row for s in chosen):
        raise ValueError("chosen shards include rows below --min-row")
    rows = [s["start"] + k for s in chosen for k in range(s["rows"])]
    print("rows", rows[0], "..", rows[-1], len(rows), flush=True)

    import numpy as np
    from datasets import Dataset, concatenate_datasets
    from huggingface_hub import snapshot_download
    import whestbench

    root = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                  allow_patterns=[s["path"] for s in chosen] + ["metadata.json"],
                                  max_workers=8))
    pieces = [Dataset.from_file(str(root / s["path"])) for s in chosen]
    subset = concatenate_datasets(pieces)
    assert len(subset) == len(rows)
    a.output.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(a.output / "prepared/full"), max_shard_size="500MB", num_proc=4)
    source_metadata = json.loads((root / "metadata.json").read_text())
    metadata = copy.deepcopy(source_metadata)
    metadata["splits"] = {"full": {**metadata["splits"]["full"], "n_mlps": len(rows)}}
    metadata["default_split"] = "full"
    metadata["prepared_splits"] = {"full": {"path": "prepared/full", "format": "save_to_disk"}}
    metadata["compute_subset"] = dict(repo=REPO, revision=REVISION, split="full", rows=len(rows),
                                      source_indices=rows,
                                      source_arrow_files=[s["path"] for s in chosen],
                                      sampling_performed=False)
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # 行番号の照合(真値は先に fetch_truth.py で取得しておく)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_truth import OUT, fetch
    ds = whestbench.load_dataset(str(a.output), split="full")
    bad = []
    for local, row in enumerate(rows):
        fetch(row)
        gt = np.load(OUT / f"row_{row:05d}.npz")["gt_mean"]
        here = np.asarray(ds[local]["all_layer_means"], dtype=np.float64)
        if not np.array_equal(gt, here):
            bad.append((local, row, float(np.abs(gt - here).max())))
    print("mismatch", bad, flush=True)
    (a.output / "ml_lab_rows.json").write_text(json.dumps(dict(rows=rows, mismatch=bad)))
    if bad:
        raise SystemExit("row mapping mismatch")


if __name__ == "__main__":
    main()
