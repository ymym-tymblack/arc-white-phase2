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
import sys
import time
import traceback
from concurrent.futures import as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import Evaluator, _run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = 2**41


def boot_ci(x, reps=20000, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _run_final(args):
    """flopscope例外をpickleせず、失敗も親へ返して保存する。予測の制限は変更しない。"""
    try:
        return _run(args)
    except Exception as error:
        return dict(idx=args[2], error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc())


def main():
    # spawnされた子はモジュール先頭を再実行する。NumPyは_initのスレッド設定後に読む。
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=ROOT / "research/frozen_phase2_final.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
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
    checkpoint = args.out.with_suffix(".jsonl")
    identity = dict(frozen=frozen, protocol=protocol,
                    execution=dict(workers=args.workers, threads=args.threads))
    saved = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        if not args.resume:
            raise SystemExit(
                f"{checkpoint} exists: use --resume, never repeat completed predictions")
        with checkpoint.open() as stream:
            if json.loads(next(stream)) != identity:
                raise SystemExit("checkpoint source/protocol/execution differs")
            for line in stream:
                entry = json.loads(line)
                key = (entry["candidate"], entry["result"]["idx"])
                if key in saved:
                    raise SystemExit(f"duplicate checkpoint row: {key}")
                saved[key] = entry["result"]
    else:
        with checkpoint.open("x") as stream:
            stream.write(json.dumps(identity) + "\n")
    ev = Evaluator(range(128), workers=args.workers, threads=args.threads,
                   dataset_source=final["path"], dataset_split="full")
    started = time.time()
    runs = {}
    for name, spec in frozen["candidates"].items():
        t = time.time()
        pending = [i for i in range(128) if (name, i) not in saved]
        futures = {
            ev.pool.submit(_run_final, (spec["path"], spec.get("kwargs", {}), i)): i
            for i in pending
        }
        for future in as_completed(futures):
            result = future.result()
            with checkpoint.open("a") as stream:
                entry = dict(candidate=name, result=result)
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            saved[(name, result["idx"])] = result
            print(f"{name}: {sum((name, i) in saved for i in range(128))}/128"
                  f" idx={result['idx']} {'FAILED' if 'error' in result else 'OK'}", flush=True)
        per = [saved[(name, i)] for i in range(128)]
        errors = [p for p in per if "error" in p]
        if errors:
            runs[name] = dict(failures=len(errors), errors=errors,
                              mean_adj=None, mean_mse=None, mean_util=None)
            print(name, json.dumps(runs[name]), flush=True)
            continue
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
        if runs[cand]["failures"] or runs[ref]["failures"]:
            comparisons[name] = dict(candidate=cand, reference=ref,
                                     accepted=False, reason="prediction failures")
            continue
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
    ev.pool.shutdown()
    record = dict(protocol=protocol, frozen=frozen, runs=runs, comparisons=comparisons,
                  execution=identity["execution"], wall_seconds=time.time() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
