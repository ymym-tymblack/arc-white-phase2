"""Full-width network / prefix-moment cross-fit diagnostics (never a reduced network).

Usage: python research/closure_wide.py NETWORK MC_STEM --out RESULT.json --nodes 5
No sampling or calibration is performed. All tensor MSEs average over ordered entries.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "32"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import k3_dense as k3
from closure_experiment import GROUPS, elliptical_fourth, joint_step, radial_fourth, standard_fourth
from harmonic4 import core_slices, project_trace
from scale_mixture import state_to_h as mixture_step

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODES = ("radial", "radial_scalar", "true_pair", "true_joint", "elliptical_scalar", "projected_trace", "gamma_scalar")


def mse(a, b):
    return float(np.mean(np.square(a - b)))


def relative(a, b):
    norm = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / norm) if norm else None


def metrics(prediction, target, noise):
    row = {}
    for name, pred, truth in zip(("mean", "covariance", "third"), prediction, target):
        raw = mse(pred, truth)
        row[name + "_mse"] = raw
        row[name + "_debiased_mse"] = raw - noise[name]
        if name != "mean":
            row[name + "_relative_norm"] = relative(pred, truth)
    return row


def candidate(weights, seed, records, path=None, kwargs=None):
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import MLP, SetupContext

    path = ROOT / "candidates/k3_factored.py" if path is None else Path(path)
    spec = importlib.util.spec_from_file_location("closure_wide_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    depth, width, _ = weights.shape
    mlp = MLP(width=width, depth=depth, weights=[fnp.array(w) for w in weights], seed=seed)
    budget = 2**41
    estimator = module.Estimator(**(kwargs or {}))
    estimator.setup(SetupContext(width=width, depth=depth, flop_budget=budget,
                                api_version="1.0", submission_dir=str(ROOT), seed=0))
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
        prediction = estimator.predict(mlp, budget)
    wall = time.perf_counter() - started
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (depth, width) or not np.isfinite(prediction).all():
        raise ValueError("Candidate returned invalid full-network prediction")
    counts = [int(r["N"]) for r in records]
    means = [np.stack([r[f"hmean_all_{l}"] for l in range(1, depth + 1)]) for r in records]
    variances = [np.stack([r[f"hvar_all_{l}"] for l in range(1, depth + 1)]) for r in records]
    truth = (means[0] + means[1]) * .5
    # For equal N this is mean(hvar_all)/(N_A+N_B), as in the measurement protocol.
    noise_entries = .25 * (variances[0] / counts[0] + variances[1] / counts[1])
    noise = noise_entries.mean(axis=1)
    raw = np.square(prediction - truth).mean(axis=1)
    util = float(ctx.flops_used / budget)
    factor = max(.1, util)
    return dict(path=str(path), kwargs=kwargs or {}, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prediction=prediction.tolist(), raw_layer_mse=raw.tolist(),
                debiased_layer_mse=(raw - noise).tolist(), reference_noise=noise.tolist(),
                raw_final_mse=float(raw[-1]), debiased_final_mse=float(raw[-1] - noise[-1]),
                flops=int(ctx.flops_used), budget=budget, C_over_B=util,
                adjusted_score=float(raw[-1] * factor),
                debiased_adjusted_score=float((raw[-1] - noise[-1]) * factor),
                residual_seconds=float(ctx.residual_wall_time_s), wall_seconds=wall,
                reference="Arithmetic average of independent full-width hmean_all replicates",
                uncertainty_note="Plug-in target-mean variance only; not a confidence interval. Equal N gives mean(hvar_all)/(sum N). Debiasing does not remove source-moment noise or finite-sample cumulant bias.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path)
    parser.add_argument("mc_stem", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nodes", default=5, type=int)
    parser.add_argument("--estimator", type=Path)
    parser.add_argument("--kwargs", type=json.loads, default={})
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    with np.load(args.network) as network:
        weights = network["weights"].copy()
        seed = int(network["seed"]) if "seed" in network else 0
    records = []
    for i in (0, 1):
        with np.load(f"{args.mc_stem}_rep{i}.npz") as record:
            records.append({key: record[key] for key in record.files})
    depth, width, input_width = weights.shape
    if width != input_width:
        raise ValueError("Expected square network weights")
    indices = np.asarray(records[0]["observed_indices"], dtype=int)
    n = len(indices)
    if n < 3 or len(set(indices.tolist())) != n or np.any(indices < 0) or np.any(indices >= width):
        raise ValueError("At least three unique valid observed coordinates are required")
    for record in records:
        if not np.array_equal(record["observed_indices"], indices):
            raise ValueError("Replicates must observe the same coordinates")
        if int(record["N"]) <= 0:
            raise ValueError("Replicate N must be positive")
        for layer in range(1, depth + 1):
            for key in ("m", "S", "T", "K4", "K22", "K31", "K211", "hmean", "hcov", "hthird", "hmean_all", "hvar_all"):
                if f"{key}_{layer}" not in record:
                    raise ValueError(f"Missing {key}_{layer}; all layers and full-width means/variances are required")
            if record[f"hmean_all_{layer}"].shape != (width,) or record[f"hvar_all_{layer}"].shape != (width,):
                raise ValueError("Full-width reference has incorrect shape")
    k3.GROUPS = set(GROUPS)
    distinct = np.ones((n, n, n), dtype=bool)
    ix = np.arange(n)
    distinct[ix, ix, :] = False
    distinct[ix, :, ix] = False
    distinct[:, ix, ix] = False
    result = dict(network=str(args.network), mc_stem=str(args.mc_stem), width=width,
                  observed_indices=indices.tolist(), depth=depth, nodes=args.nodes,
                  samples=[int(r["N"]) for r in records], layers={},
                  protocol="Layer-local crossfit A→B and B→A. Prefix coordinates are observations of the full network; no prefix propagation. Layer 1 uses exact Gaussian preactivation. Negative debiased errors are retained.",
                  metric_definition="MSE averages squared error over every ordered tensor entry (including repeated indices); relative_norm is Frobenius error / target Frobenius norm. Noise is half squared independent-replicate difference per entry, estimating an individual target's noise under equal replicate sizes. Unequal N yields the average target-noise floor.")
    with threadpool_limits(limits=32):
        for layer in (() if args.score_only else range(1, depth + 1)):
            targets = [tuple(r[f"{key}_{layer}"] for key in ("hmean", "hcov", "hthird")) for r in records]
            noise = {name: .5 * mse(a, b) for name, a, b in zip(("mean", "covariance", "third"), *targets)}
            cross = {mode: [] for mode in MODES}
            standardized = [standard_fourth(r, layer) for r in records]
            empirical = [x[1][distinct] for x in standardized]
            structure_noise = .5 * mse(*empirical)
            structure = dict(individual_target_noise=structure_noise,
                             replication_cross_moment=float(np.mean(empirical[0] * empirical[1])),
                             replicate_second_moments=[float(np.mean(x*x)) for x in empirical],
                             ordered_distinct_entries=int(distinct.sum()), projections={})
            for source in (0, 1):
                rec = records[source]
                k4, k211 = standardized[source]
                # Structure uses measured values even at layer 1 to expose MC null noise.
                G = project_trace(k211)
                scalar_G = np.eye(n) * np.trace(G) / n
                sigma = np.sqrt(np.maximum(np.diag(rec[f"S_{layer}"]), 1e-20))
                rho_emp = rec[f"S_{layer}"] / np.outer(sigma, sigma)
                projections = {"scalar_trace": core_slices(scalar_G, np.eye(n))[1],
                               "matrix_trace": core_slices(G, np.eye(n))[1],
                               "elliptical_scalar": elliptical_fourth(k4[0], rho_emp)[1]}
                for name, projection in projections.items():
                    value = projection[distinct]
                    raw = mse(value, empirical[1-source])
                    structure["projections"].setdefault(name, []).append(dict(
                        crossfit_mse=raw, debiased_mse=raw-structure_noise,
                        relative_norm=relative(value, empirical[1-source]),
                        projection_second_moment=float(np.mean(value*value)),
                        cross_moment=float(np.mean(value*empirical[1-source]))))
                m, S, T = (rec[f"{key}_{layer}"] for key in ("m", "S", "T"))
                if layer == 1:
                    W = weights[0][:, indices].astype(np.float64)
                    m, S, T = np.zeros(n), W.T @ W, np.zeros_like(T)
                    k4 = tuple(np.zeros_like(x) for x in k4)
                    k211 = np.zeros_like(k211)
                sigma = np.sqrt(np.maximum(np.diag(S), 1e-20))
                rho = S / np.outer(sigma, sigma)
                omega = float(np.mean(k4[0]))
                choices = {"radial": (radial_fourth(k4[0]), None),
                           "radial_scalar": (radial_fourth(np.full(n, omega)), None),
                           "true_pair": (k4, None), "true_joint": (k4, k211),
                           "elliptical_scalar": elliptical_fourth(k4[0], rho),
                           "projected_trace": core_slices(project_trace(k211), np.eye(n))}
                for mode in MODES:
                    if mode == "gamma_scalar":
                        mixed = mixture_step(m, S, T, omega, nodes=args.nodes, kmax=3, emax=2)
                        pred = tuple(mixed[key] for key in ("mu", "C", "U"))
                    else:
                        pred = joint_step(m, S, T, *choices[mode])
                    cross[mode].append(metrics(pred, targets[1-source], noise))
            summary = {}
            for mode, rows in cross.items():
                summary[mode] = {key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None for key in rows[0]}
            result["layers"][str(layer)] = dict(target_noise=noise, crossfit=summary,
                                                directions=cross, structure=structure)
            print(f"layer {layer}/{depth} mean target noise={noise['mean']:.3e}", flush=True)
            for mode in MODES:
                row = summary[mode]
                print(f"  {mode:20s} mean={row['mean_mse']:.4e} debiased={row['mean_debiased_mse']:.4e} cov={row['covariance_mse']:.3e} third={row['third_mse']:.3e} third_rel={row['third_relative_norm']}", flush=True)
        result["candidate"] = candidate(weights, seed, records, args.estimator, args.kwargs)
    c = result["candidate"]
    print(f"candidate final raw={c['raw_final_mse']:.4e} debiased={c['debiased_final_mse']:.4e} C/B={c['C_over_B']:.4f} adjusted={c['adjusted_score']:.4e} residual={c['residual_seconds']:.3f}s wall={c['wall_seconds']:.2f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
