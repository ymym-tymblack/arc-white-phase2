"""Evaluate frozen compute candidates once on an independent official subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

from eval_mini import ROOT, _init, _load_mlp, dataset_identity

BUDGET = 2**41


def score_model(model, mlp, truth):
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    ctx = None
    started = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location("frozen_" + model["name"], model["path"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        estimator = module.Estimator(**model.get("kwargs", {}))
        estimator.setup(
            SetupContext(
                width=mlp.width,
                depth=mlp.depth,
                flop_budget=BUDGET,
                api_version="2.0",
                submission_dir=str(ROOT),
                seed=0,
            )
        )
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120.0, quiet=True) as ctx:
            prediction = estimator.predict(mlp, BUDGET)
        elapsed = time.monotonic() - started
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != truth.shape or prediction.shape != (mlp.depth, mlp.width):
            raise ValueError(f"Wrong prediction shape {prediction.shape}; expected {truth.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        used = int(ctx.flops_used)
        if used > BUDGET or elapsed >= 120.0:
            raise ValueError("Compute or wall limit exceeded")
        mse = float(np.mean((prediction[-1] - truth[-1]) ** 2))
        return dict(
            feasible=True,
            mse=mse,
            adjusted_score=mse * max(0.1, used / BUDGET),
            flops=used,
            C_over_B=used / BUDGET,
            wall_seconds=elapsed,
            residual_seconds=float(ctx.residual_wall_time_s),
            layers=np.mean((prediction - truth) ** 2, axis=1).tolist(),
        )
    except Exception as error:
        # Match the official penalty explicitly in the evaluator, never hide a
        # failure or switch models inside the submitted Estimator.predict().
        zero_mse = float(np.mean(truth[-1] ** 2))
        return dict(
            feasible=False,
            mse=zero_mse,
            adjusted_score=zero_mse,
            flops=int(ctx.flops_used) if ctx is not None else None,
            wall_seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
            failure_scoring="zero prediction, multiplier1; failure retained",
        )


def run_network(task):
    import numpy as np

    index, models = task
    mlp, truth = _load_mlp(index)
    if truth.shape != (mlp.depth, mlp.width) or not np.isfinite(truth).all():
        raise ValueError("Official reference has invalid shape or values")
    return dict(
        index=index,
        seed=str(mlp.seed),
        name=mlp.name,
        models={model["name"]: score_model(model, mlp, truth) for model in models},
    )


def verify_models(models):
    for model in models:
        actual = hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest()
        if actual != model["source_sha256"]:
            raise ValueError(f"Frozen source differs for {model['name']}")


def summarize(networks, name):
    rows = [network["models"][name] for network in networks]
    successful = [row for row in rows if row["feasible"]]
    return dict(
        adjusted_score=sum(row["adjusted_score"] for row in rows) / len(rows),
        mse=sum(row["mse"] for row in rows) / len(rows),
        failures=len(rows) - len(successful),
        mean_flops=sum(row["flops"] for row in successful) / len(successful)
        if successful
        else None,
        mean_C_over_B=sum(row["C_over_B"] for row in successful) / len(successful)
        if successful
        else None,
        max_wall_seconds=max(row["wall_seconds"] for row in rows),
        parallel_residual_note="Diagnostic only; single-thread preflight is the official residual gate.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.workers * args.threads != 32:
        parser.error("Offline evaluation must allocate exactly32 native threads")
    frozen = json.loads(args.frozen.read_text())
    models = frozen["models"]
    labels = [model["name"] for model in models]
    required = {"positive_before", "legacy_before", "exact_only", "tuned"}
    if not required.issubset(labels) or len(set(labels)) != len(labels):
        raise ValueError(
            "Frozen comparison must include unique baselines, exact-only and tuned roles"
        )
    verify_models(models)
    data = dataset_identity(frozen["dataset"]["source"], frozen["dataset"]["split"])
    if data != frozen["dataset"]:
        raise ValueError("Frozen dataset identity differs")
    if not frozen["single_thread_preflight_passed"]:
        raise ValueError("Official safety must pass before inspecting final predictions")
    indices = frozen["indices"]
    if indices != list(range(64)):
        raise ValueError("Final indices differ from the predeclared64-record protocol")
    started = time.monotonic()
    document = dict(
        frozen=str(args.frozen),
        models=models,
        dataset=data,
        networks=[],
        summaries={},
        adopted=None,
        final_tuning=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init,
        initargs=(args.threads, data),
    ) as pool:
        for row in pool.map(run_network, [(index, models) for index in indices]):
            document["networks"].append(row)
            args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
            print(
                f"full{row['index']}: "
                + " ".join(
                    f"{name}={row['models'][name]['adjusted_score']:.4e}" for name in labels
                ),
                flush=True,
            )
    verify_models(models)
    summaries = {name: summarize(document["networks"], name) for name in labels}
    positive, legacy = summaries["positive_before"], summaries["legacy_before"]
    exact, tuned = summaries["exact_only"], summaries["tuned"]
    gain_positive = 1 - tuned["adjusted_score"] / positive["adjusted_score"]
    gain_legacy = 1 - tuned["adjusted_score"] / legacy["adjusted_score"]
    improved = sum(
        row["models"]["tuned"]["adjusted_score"]
        < row["models"]["positive_before"]["adjusted_score"]
        for row in document["networks"]
    )
    gate = frozen["acceptance"]
    tuned_ok = (
        tuned["failures"] == 0
        and positive["failures"] == 0
        and legacy["failures"] == 0
        and gain_positive >= gate["minimum_gain"]
        and gain_legacy >= gate["minimum_gain"]
        and improved >= gate["minimum_improved_networks"]
    )
    exact_drift = exact["mse"] / positive["mse"] - 1
    exact_ok = (
        exact["failures"] == 0
        and positive["failures"] == 0
        and abs(exact_drift) <= gate["maximum_exact_mse_drift"]
        and exact["mean_flops"] < positive["mean_flops"]
    )
    adopted = "tuned" if tuned_ok else "exact_only" if exact_ok else None
    if adopted is None and "exact_legacy" in summaries:
        safe = summaries["exact_legacy"]
        safe_drift = safe["mse"] / legacy["mse"] - 1
        if (
            safe["failures"] == 0
            and legacy["failures"] == 0
            and abs(safe_drift) <= gate["maximum_exact_mse_drift"]
            and safe["mean_flops"] < legacy["mean_flops"]
        ):
            adopted = "exact_legacy"
    document.update(
        summaries=summaries,
        tuned_accepted=tuned_ok,
        exact_accepted=exact_ok,
        tuned_gain_vs_positive=gain_positive,
        tuned_gain_vs_legacy=gain_legacy,
        tuned_improved_networks=improved,
        exact_mse_drift=exact_drift,
        adopted=adopted,
        wall_seconds=time.monotonic() - started,
        uncertainty="Independent official public full64, not a submitted/private leaderboard result.",
    )
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "adopted": adopted,
                "tuned_gain_vs_positive": gain_positive,
                "tuned_gain_vs_legacy": gain_legacy,
                "improved_networks": improved,
                "exact_mse_drift": exact_drift,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
