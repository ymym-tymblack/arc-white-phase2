"""Exercise thin family factors, pruning, rank allocation and measured FLOPs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 2**41


def initialize():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(ROOT))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(path, kwargs, mlp, name):
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    module = load_module(path, name)
    estimator = module.Estimator(collect=True, **kwargs)
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
        prediction = np.asarray(estimator.predict(mlp, BUDGET), dtype=np.float64)
    if prediction.shape != (mlp.depth, mlp.width) or not np.isfinite(prediction).all():
        raise AssertionError("Full prediction must have the expected shape and finite values")
    moments = [
        {key: np.asarray(row[key], dtype=np.float64) for key in ("D3", "D21")}
        for row in estimator.features
    ]
    stats = [row.get("compute", {}) for row in estimator.features]
    physical = []
    for row in estimator.features:
        if "k4_raw_variance" in row:
            physical.append(
                dict(
                    post=float(np.min(np.asarray(row["k4_raw_variance"]))),
                    pre=float(np.min(np.asarray(row["k4_raw_pre_variance"]))),
                )
            )
    return (
        prediction,
        moments,
        dict(
            flops=int(ctx.flops_used),
            wall_seconds=time.monotonic() - started,
            residual_seconds=float(ctx.residual_wall_time_s),
            compute=stats,
            raw_variance_minima=physical,
        ),
    )


def run_case(task):
    import numpy as np
    from local_engine import build_mlp

    name, width, depth, kwargs, before_path, after_path, approximate = task
    mlp = build_mlp(width, depth, seed=1)
    after, moments, metrics = evaluate(after_path, kwargs, mlp, "compute_audit_after")
    result = dict(name=name, width=width, depth=depth, kwargs=kwargs, after=metrics, ok=True)
    if not approximate:
        before, old_moments, old_metrics = evaluate(
            before_path, kwargs, mlp, "compute_audit_before"
        )
        np.testing.assert_allclose(after, before, rtol=5e-5, atol=2e-6)
        differences = []
        for old, new in zip(old_moments, moments):
            differences.append({key: float(np.max(np.abs(old[key] - new[key]))) for key in old})
            for key in old:
                np.testing.assert_allclose(new[key], old[key], rtol=3e-4, atol=2e-5)
        result.update(
            before=old_metrics,
            max_prediction_difference=float(np.max(np.abs(after - before))),
            moment_differences=differences,
            flop_reduction=1 - metrics["flops"] / old_metrics["flops"],
        )
    else:
        # The same rank as width keeps the entire basis, even under an SVD rotation.
        if name == "full_rank_energy_rotation":
            reference = {**kwargs, "rank_old_energy": 1.0}
        elif name == "prune_all_old_full_rank":
            reference = {**kwargs, "age_old": None}
        else:
            reference = {**kwargs, "family_prune_rel": 0.0, "rank_old_energy": 1.0}
        base, _, base_metrics = evaluate(after_path, reference, mlp, "compute_audit_reference")
        result.update(
            reference=base_metrics,
            max_prediction_change=float(np.max(np.abs(after - base))),
            flop_change=metrics["flops"] / base_metrics["flops"] - 1,
        )
        if name in ("full_rank_energy_rotation", "prune_all_old_full_rank"):
            np.testing.assert_allclose(after, base, rtol=5e-5, atol=2e-6)
        if name == "prune_all_old_full_rank" and not any(
            row.get("old_families_removed", 0) for row in metrics["compute"]
        ):
            raise AssertionError("The empty shared-tier transition was not exercised")
    if kwargs.get("directional_k4", True):
        first = np.asarray(mlp.weights[0], dtype=np.float64)
        analytic = np.sqrt(np.sum(first * first, axis=0) / (2 * np.pi))
        np.testing.assert_allclose(after[0], analytic, rtol=3e-6, atol=2e-6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("/tmp/compute_before.py"))
    parser.add_argument("--candidate", type=Path, default=ROOT / "estimator.py")
    parser.add_argument("--output", type=Path, default=ROOT / "results/compute_audit.json")
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    base = dict(age_old=2, rank_old=16, iik_rank=8, k3leg_rank=8)
    cases = [
        ("directional_shared", 64, 10, {**base, "directional_k4": True}, False),
        ("legacy_shared", 64, 10, {**base, "directional_k4": False}, False),
        (
            "legacy_fourth_birth",
            64,
            10,
            {**base, "directional_k4": False, "beta_k4": 0.1, "k4_birth_rad": True},
            False,
        ),
        ("no_residual_or_legs", 64, 10, {**base, "iik_rank": None, "k3leg": False}, False),
        ("unshared", 32, 6, {**base, "age_old": None}, False),
        ("rebuild_shared", 64, 10, {**base, "rebuild_old": True}, False),
        ("one_layer", 16, 1, {**base}, False),
        (
            "width_four",
            4,
            2,
            {**base, "age_old": None, "rank_old": 4, "iik_rank": 4, "k3leg_rank": 4},
            False,
        ),
        (
            "full_rank_energy_rotation",
            32,
            8,
            {**base, "rank_old": 32, "rank_old_min": 32, "rank_old_energy": 0.97},
            True,
        ),
        (
            "prune_all_old_full_rank",
            32,
            8,
            {**base, "age_old": 0, "rank_old": 32, "family_prune_rel": 1.0},
            True,
        ),
        (
            "adaptive_rank",
            128,
            12,
            {**base, "rank_old": 64, "rank_old_min": 16, "rank_old_energy": 0.97},
            True,
        ),
        ("prune_contributions", 128, 12, {**base, "family_prune_rel": 0.03}, True),
    ]
    if args.wide:
        cases.extend(
            [
                ("wide_default", 1024, 16, {}, False),
                ("wide_legacy", 1024, 16, dict(directional_k4=False, age_old=6), False),
                ("wide_no_residual", 1024, 16, dict(iik_rank=None), False),
                ("wide_no_legs", 1024, 16, dict(k3leg=False), False),
                ("wide_rebuild", 1024, 16, dict(age_old=3, rank_old=128, rebuild_old=True), False),
                ("wide_prune", 1024, 16, dict(family_prune_rel=0.03), True),
                ("wide_energy", 1024, 16, dict(rank_old_energy=0.995), True),
                ("deep_absent_third", 256, 32, {}, False),
            ]
        )
    # New compute knobs do not exist on the immutable model; they are used only
    # in approximate cases or fixed at defaults on both imported implementations.
    tasks = [
        (
            name,
            width,
            depth,
            kwargs,
            str(args.before.resolve()),
            str(args.candidate.resolve()),
            approximate,
        )
        for name, width, depth, kwargs, approximate in cases
    ]
    document = dict(
        ok=False,
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        before_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
        workers=8,
        threads_per_worker=4,
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=8, mp_context=mp.get_context("spawn"), initializer=initialize
    ) as pool:
        for row in pool.map(run_case, tasks):
            document["cases"].append(row)
            args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
            print(
                f"{row['name']}: OK flops={row['after']['flops']} max-diff={row.get('max_prediction_difference', row.get('max_prediction_change')):.3g}",
                flush=True,
            )
    document.update(ok=True, wall_seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(f"All {len(cases)} compute audit scenarios passed", flush=True)


if __name__ == "__main__":
    main()
