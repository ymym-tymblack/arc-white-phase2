"""探索で固定した候補を未使用MLPで一度だけ比較する(入力サンプリングなし)。

GPU参照は事前にgpu_joint4.pyで作る。ここでは保存済みの平均を読むだけ。
  uv run python research/eval_directional_final.py --frozen results/directional_frozen.json --out results/directional_final.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent


def run_network(task):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from closure_wide import candidate

    seed, kwargs, baseline_kwargs, threads = task
    network_path = Path(f"/kaggle/ext/closure/network_n1024_s{seed}.npz")
    weights = np.load(network_path)["weights"]
    records = [np.load(f"/kaggle/ext/closure/final_n1024_s{seed}_rep{i}.npz") for i in (0, 1)]
    with threadpool_limits(limits=threads):
        before = candidate(weights, seed, records, kwargs=baseline_kwargs)
        after = candidate(weights, seed, records, kwargs=kwargs)
    prediction = np.asarray(after.pop("prediction"))
    p0, p1 = np.asarray(before.pop("prediction"))[-1], prediction[-1]
    changes = []
    for record in records:
        truth = record["hmean_all_16"]
        noise = float(np.mean(record["hvar_all_16"]) / int(record["N"]))
        f0, f1 = max(.1, before["C_over_B"]), max(.1, after["C_over_B"])
        s0 = (float(np.mean((p0-truth)**2)) - noise) * f0
        s1 = (float(np.mean((p1-truth)**2)) - noise) * f1
        changes.append(s0-s1)
    feasible = bool(prediction.shape == weights.shape[:2] and np.isfinite(prediction).all()
                    and after["flops"] <= after["budget"] and after["wall_seconds"] <= 120.0)
    return dict(seed=seed, baseline=before, candidate=after, feasible=feasible,
                paired_reference_replicate_improvements=changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text())
    protocol = json.loads((ROOT / "results/directional_protocol.json").read_text())
    source = ROOT / "candidates/k3_factored.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != frozen["source_sha256"]:
        raise ValueError("Frozen candidate source hash differs from current estimator")
    if frozen["final_seeds"] != protocol["final_seeds"]:
        raise ValueError("Final seeds differ from the predeclared protocol")
    if args.workers * args.threads != 32:
        raise ValueError("Evaluation must allocate exactly32 CPU threads in total")
    tasks = [(s, frozen["kwargs"], protocol["baseline_kwargs"], args.threads)
             for s in protocol["final_seeds"]]
    started = time.perf_counter()
    result = dict(frozen=str(args.frozen), source_sha256=digest,
                  kwargs=frozen["kwargs"], protocol="results/directional_protocol.json",
                  networks=[], accepted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(run_network, tasks):
            result["networks"].append(row)
            args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"seed {row['seed']}: baseline={row['baseline']['debiased_adjusted_score']:.4e} candidate={row['candidate']['debiased_adjusted_score']:.4e}", flush=True)
    baseline = sum(x["baseline"]["debiased_adjusted_score"] for x in result["networks"]) / len(tasks)
    current = sum(x["candidate"]["debiased_adjusted_score"] for x in result["networks"]) / len(tasks)
    raw_baseline = sum(x["baseline"]["adjusted_score"] for x in result["networks"]) / len(tasks)
    raw_current = sum(x["candidate"]["adjusted_score"] for x in result["networks"]) / len(tasks)
    reference_changes = [sum(x["paired_reference_replicate_improvements"][i] for x in result["networks"]) / len(tasks) for i in (0, 1)]
    improved = sum(x["candidate"]["debiased_adjusted_score"] < x["baseline"]["debiased_adjusted_score"] for x in result["networks"])
    relative_gain = 1.0 - current / baseline
    gate = protocol["final_acceptance"]
    accepted = (relative_gain >= gate["minimum_relative_gain"]
                and improved >= gate["minimum_improved_networks"]
                and all(x > 0 for x in reference_changes)
                and all(x["feasible"] for x in result["networks"])
                and frozen["single_thread_preflight_passed"])
    result.update(accepted=accepted, debiased_baseline=baseline, debiased_candidate=current,
                  raw_baseline=raw_baseline, raw_candidate=raw_current,
                  relative_gain=relative_gain, improved_networks=improved,
                  independent_reference_mean_improvements=reference_changes,
                  reference_uncertainty="Two independent reference series, not a confidence interval; no final-data tuning.",
                  wall_seconds=time.perf_counter()-started)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"final: baseline={baseline:.4e} candidate={current:.4e} gain={relative_gain:.2%} improved={improved}/{len(tasks)} accepted={accepted}", flush=True)


if __name__ == "__main__":
    main()
