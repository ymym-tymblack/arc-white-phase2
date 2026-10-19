"""PyTorch 移植の学習・検証用に、MLP の重みと正解の全層平均を numpy 形式へ書き出す(repo 環境で実行)。

各集合 NAME について /kaggle/ext/torch_data/NAME/ に
  w_XXXX.npy   (depth, n, n) float32。mlp.weights[l] そのもの(向きは (入力, 出力)、z = Wᵀ x)
  truth.npy    (行数, depth, n) float64。ds[i]["all_layer_means"]
  ids.json     公式 full の行番号(mlp_id)の並び
を保存する。重みは学習中に mmap で読む。バイアスは whestbench の MLP に無い。
最終確認用の行(full_final128・full_holdout126)は読まない。

例: uv run python research/torch_port/prep_data.py dev64 train64 train160
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path("/kaggle/ext/torch_data")
SETS = {
    "train64": "/kaggle/ext/compute_full64",  # 公式 full 0〜63
    "dev64": "/kaggle/ext/full_dev64",  # 公式 full 64〜127(開発・選択用)
    "train160": "/kaggle/ext/full_train",  # 公式 full 264〜423
    "train198": "/kaggle/ext/full_train2",  # 公式 full 424〜621
}
FORBIDDEN = ("full_final128", "full_holdout126")


def main():
    import numpy as np
    import whestbench

    for name in sys.argv[1:]:
        source = SETS[name]
        if any(tag in source for tag in FORBIDDEN):
            raise SystemExit(f"forbidden dataset {source}")
        ds = whestbench.load_dataset(source, split="full")
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        ids, truth = [], []
        for i in range(len(ds)):
            row = ds[i]
            gid = int(row["mlp_id"])
            path = out / f"w_{gid:04d}.npy"
            if not path.exists():
                mlp = whestbench.mlp_at(ds, i)
                weights = np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights])
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
                np.save(tmp, weights)
                tmp.rename(path)
            ids.append(gid)
            truth.append(np.asarray(row["all_layer_means"], dtype=np.float64))
            print(name, i, gid, flush=True)
        np.save(out / "truth.npy", np.stack(truth))
        (out / "ids.json").write_text(json.dumps(ids))
        print(name, "done", len(ids), ids[0], ids[-1], flush=True)


if __name__ == "__main__":
    main()
