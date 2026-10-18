"""凍結した候補を最終確認用の集合(公式 full128〜255)で一度だけ評価し、事前の採否条件を判定する。

提出物ではない。research/protocol_phase2_final.json の条件と、凍結記録
research/frozen_phase2_final.json(候補のファイル・sha256・クラス属性の上書き)に従う。
結果ファイルが既にあれば実行しない(最終確認用の集合は一度しか使わない)。
  uv run python research/eval_final128.py --out results/final128_phase2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import Evaluator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = 2**41


def boot_ci(x, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=ROOT / "research/frozen_phase2_final.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists: the final set is evaluated only once")
    protocol = json.loads((ROOT / "research/protocol_phase2_final.json").read_text())
    frozen = json.loads(args.frozen.read_text())
    final = protocol["datasets"]["final"]
    meta = Path(final["path"]) / "metadata.json"
    if hashlib.sha256(meta.read_bytes()).hexdigest() != final["metadata_sha256"]:
        raise SystemExit("final dataset metadata differs from the protocol")
    for spec in frozen["candidates"].values():
        digest = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise SystemExit(f"frozen source changed: {spec['path']}")
    ev = Evaluator(range(128), workers=args.workers, threads=args.threads,
                   dataset_source=final["path"], dataset_split="full")
    started = time.time()
    runs = {}
    for name, spec in frozen["candidates"].items():
        t = time.time()
        result = ev.score(spec["path"], **spec.get("kwargs", {}))
        per = result["per"]
        runs[name] = dict(
            mse=[p["mse"] for p in per], flops=[p["flops"] for p in per],
            adj=[p["mse"] * max(0.1, p["flops"] / B) for p in per],
            failures=int(sum(1 for p in per if not np.isfinite(p["mse"]) or p["flops"] > B)),
            mean_adj=float(np.mean([p["mse"] * max(0.1, p["flops"] / B) for p in per])),
            mean_mse=float(np.mean([p["mse"] for p in per])),
            mean_util=float(np.mean([p["flops"] for p in per]) / B),
            max_parallel_residual=float(max(p["resid"] for p in per)),
            seconds=time.time() - t,
        )
        print(name, json.dumps({k: runs[name][k] for k in ("mean_adj", "mean_mse", "mean_util",
                                                            "failures")}), flush=True)
    comparisons = {}
    for name, (cand, ref) in frozen["comparisons"].items():
        mse_ratio = np.asarray(runs[cand]["mse"]) / np.asarray(runs[ref]["mse"])
        adj_ratio = np.asarray(runs[cand]["adj"]) / np.asarray(runs[ref]["adj"])
        comparisons[name] = dict(
            candidate=cand, reference=ref,
            mean_mse_ratio=float(mse_ratio.mean()), mse_ratio_ci95=boot_ci(mse_ratio),
            mean_adj_ratio=float(adj_ratio.mean()), adj_ratio_ci95=boot_ci(adj_ratio),
            ratio_of_mean_adj=runs[cand]["mean_adj"] / runs[ref]["mean_adj"],
            improved_adj=int((adj_ratio < 1).sum()), n=int(len(adj_ratio)),
        )
        print(name, json.dumps(comparisons[name]), flush=True)
    record = dict(protocol=protocol, frozen=frozen, runs=runs, comparisons=comparisons,
                  wall_seconds=time.time() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
