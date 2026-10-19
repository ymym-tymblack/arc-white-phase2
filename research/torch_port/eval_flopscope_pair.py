"""固定した基準・Adam候補を同じ開発64網でflopscope評価する。holdoutは読まない。

120秒/FLOP予算の条件を変えず、網ごとの成功・失敗をJSONLへ保存する。
--seed-candidate-refは同一候補で直前に作ったref_dumpの4網を再利用する。
refディレクトリのprovenance.jsonにcandidate_sha256が一致することを要求する。
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_mini import Evaluator, _run, dataset_identity  # noqa: E402

DATASET = "/kaggle/ext/full_dev64"
BUDGET = 2**41


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_record(job):
    try:
        return _run(job)
    except Exception as error:
        return dict(idx=job[2], error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc())


def aggregate(path, sha, per):
    errors = [row for row in per if "error" in row]
    if errors:
        return dict(candidate=str(path), sha256=sha, failures=len(errors), errors=errors, per=per)
    n = len(per)
    return dict(
        candidate=str(path), sha256=sha, kwargs={}, dataset=DATASET, indices=[0, 63],
        failures=0, mse=sum(row["mse"] for row in per) / n,
        adj=sum(row["mse"] * max(0.1, row["flops"] / BUDGET) for row in per) / n,
        util=sum(row["flops"] for row in per) / n / BUDGET,
        max_wall_parallel=max(row["sec"] for row in per),
        max_residual_parallel=max((row["resid"] for row in per
                                   if row.get("resid") is not None), default=None), per=per,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed-candidate-ref", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Completed result exists: {args.out}")
    sources = {name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in
               (("reference", args.reference), ("candidate", args.candidate))}
    identity = dict(sources=sources, dataset=dataset_identity(DATASET, "full"),
                    execution=dict(workers=args.workers, threads=args.threads),
                    wall_limit=120.0, flop_budget=BUDGET)
    checkpoint = args.out.with_suffix(".jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    if checkpoint.exists():
        if not args.resume:
            raise SystemExit("Checkpoint exists; use --resume")
        with checkpoint.open() as stream:
            if json.loads(next(stream)) != identity:
                raise SystemExit("Checkpoint identity differs")
            for line in stream:
                entry = json.loads(line)
                key = (entry["name"], entry["result"]["idx"])
                if key in saved:
                    raise SystemExit(f"Duplicate checkpoint row: {key}")
                saved[key] = entry["result"]
    else:
        checkpoint.write_text(json.dumps(identity) + "\n")

    def save(name, result):
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(dict(name=name, result=result), allow_nan=False) + "\n")
        saved[(name, result["idx"])] = result

    if args.seed_candidate_ref:
        ref = args.seed_candidate_ref
        provenance = json.loads((ref / "provenance.json").read_text())
        if provenance["candidate_sha256"] != sources["candidate"]["sha256"]:
            raise SystemExit("Reference dump is from a different candidate")
        for row in json.loads((ref / "summary_0_3.json").read_text()):
            if row["idx"] not in range(4) or row["sec"] > 120.0 or row["flops"] > BUDGET:
                raise SystemExit("Seed reference violates index/wall/FLOP conditions")
            if ("candidate", row["idx"]) not in saved:
                save("candidate", dict(row, resid=None, reused_reference_dump=str(ref)))
    started = time.time()
    ev = Evaluator(range(64), workers=args.workers, threads=args.threads,
                   dataset_source=DATASET, dataset_split="full")
    runs = {}
    try:
        for name, spec in sources.items():
            pending = [i for i in range(64) if (name, i) not in saved]
            futures = [ev.pool.submit(run_record, (spec["path"], {}, i)) for i in pending]
            for future in as_completed(futures):
                result = future.result()
                save(name, result)
                count = sum((name, i) in saved for i in range(64))
                print(f"{name} {count}/64 idx={result['idx']} "
                      f"{'FAILED' if 'error' in result else 'OK'}", flush=True)
            per = [saved[(name, i)] for i in range(64)]
            runs[name] = aggregate(spec["path"], spec["sha256"], per)
            print(name, json.dumps({k: v for k, v in runs[name].items()
                                    if k not in ("per", "errors")}), flush=True)
    finally:
        ev.pool.shutdown()
    record = dict(identity=identity, runs=runs, seconds=time.time() - started,
                  passed=all(run["failures"] == 0 for run in runs.values()),
                  timing_note="Parallel residual is diagnostic; quiet safety check is separate.")
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
