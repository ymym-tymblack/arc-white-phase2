"""凍結前の候補を開発集合(公式 full64〜127)で評価し、ネットワークごとの結果を保存する。

提出物ではない。最終確認用の集合(full128〜255)には触れない。
  uv run python research/eval_dev_candidate.py estimator.py --out results/dev64_candidate.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import Evaluator  # noqa: E402

DEV = "/kaggle/ext/full_dev64"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", default="0-63")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--kwargs", default="{}", help="Estimator のクラス属性の上書き(JSON)")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.indices.split("-"))
    started = time.time()
    ev = Evaluator(range(lo, hi + 1), workers=args.workers, threads=args.threads,
                   dataset_source=DEV, dataset_split="full")
    result = ev.score(str(args.candidate.resolve()), **json.loads(args.kwargs))
    record = dict(
        candidate=str(args.candidate), sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        kwargs=json.loads(args.kwargs), dataset=DEV, indices=[lo, hi],
        adj=result["adj"], mse=result["mse"], util=result["util"],
        max_residual_parallel=result["resid"], max_wall_parallel=result["sec"],
        per=[{k: p[k] for k in ("idx", "mse", "flops", "sec", "resid")} for p in result["per"]],
        wall_seconds=time.time() - started,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("sha256", "adj", "mse", "util")}), flush=True)


if __name__ == "__main__":
    main()
