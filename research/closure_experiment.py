"""高次閉包の研究: 未使用乱数 MLP の固定、κ4 介入、独立 MC 標本での比較。

提出物ではない。MC は gpu_joint4.py に分離し、ここでは統計量だけを使う。
  research/gpu_env.sh research/closure_experiment.py prepare
  research/gpu_env.sh research/closure_experiment.py assess NETWORK MC_STEM --out RESULT.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import k3_dense as k3

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/kaggle/ext/closure")
PROTOCOL = ROOT / "results/closure_protocol.json"
GROUPS = {"rho", "tau1", "tau1e", "txxy", "txxx", "pairtau", "meantau", "iik", "iii"}


def prepare():
    """探索用と最終確認用の seed を結果を見る前に固定する。"""
    records = []
    for width, depth, seeds, role in (
        (64, 8, (26051501, 26051502), "diagnostic"),
        (128, 8, (26051503,), "validation"),
        (1024, 16, (26051511, 26051512), "diagnostic"),
        (1024, 16, (26051591, 26051592, 26051593, 26051594), "final_untouched"),
    ):
        for seed in seeds:
            path = DATA / f"network_n{width}_s{seed}.npz"
            if not path.exists():
                rng = np.random.default_rng(seed)
                weights = np.asarray([
                    (rng.standard_normal((width, width)) * math.sqrt(2.0 / width)).astype(np.float32)
                    for _ in range(depth)
                ])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, weights=weights, seed=seed)
            records.append(dict(width=width, depth=depth, seed=seed, role=role, path=str(path)))
    protocol = dict(
        version=1, networks=records, baseline="candidates/k3_factored.py at 4d56125",
        inference_sampling=False, fitting_final_seeds=False,
        small_width_samples_per_replicate=2_000_000, independent_replicates=2,
        precision="float32 GEMM, float64 accumulation, TF32 disabled",
        decision="Reject if cross-fitted mean error does not improve beyond independent-replicate noise; measure FLOPs before adoption.",
        note="Previous mini indices 0..99 are development data, not an untouched test set.",
    )
    PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)


def standard_fourth(record, layer):
    S = record[f"S_{layer}"]
    var = np.maximum(np.diag(S), 1e-20)
    sigma = np.sqrt(var)
    k4 = (record[f"K4_{layer}"] / var**2,
          record[f"K22_{layer}"] / np.outer(var, var),
          record[f"K31_{layer}"] / (var[:, None] * np.outer(sigma, sigma)))
    key = f"K211_{layer}"
    k211 = record[key] / (sigma[:, None, None] * sigma[None, :, None] * var[None, None, :]) if key in record else None
    return k4, k211


def radial_fourth(omega):
    n = omega.size
    # Existing SIMPLE approximation: only (4) and the leading (2,2) trace.
    return omega, np.full((n, n), float(np.mean(omega)) / 3.0), np.zeros((n, n))


def elliptical_fourth(omega, rho):
    value = float(np.mean(omega))
    k22 = (value / 3.0) * (1.0 + 2.0 * rho**2)
    k31 = value * rho
    k211 = (value / 3.0) * (rho[:, :, None] + 2.0 * rho[:, None, :] * rho[None, :, :])
    return (np.full(rho.shape[0], value), k22, k31), k211


def fourth_birth(coeffs, rho, k4, k211, edge_order=1):
    """4 本ブロック1個と共分散辺から、相異なる3頂点の connected diagram を作る。

    分母は各頂点のブロック脚数 a! と各辺の重複度 e!。同一添字は pair_cov の
    power-cumulant 法で別計算済みなので、この関数からの寄与はゼロにする。
    """
    n = rho.shape[0]
    omega, k22, k31 = k4
    out = np.zeros((n, n, n))
    if not np.any(omega) and not np.any(k22) and not np.any(k31) and (k211 is None or not np.any(k211)):
        return out
    powers = k3.rho_pows(rho, edge_order)
    for a in itertools.product(range(5), repeat=3):
        if sum(a) != 4:
            continue
        support = [i for i in range(3) if a[i]]
        if max(a) == 4:
            block = k3._place1(omega, a.index(4), n)
        elif max(a) == 3:
            block = k3._place2(k31, a.index(3), a.index(1), n)
        elif sorted(a) == [0, 2, 2]:
            block = k3._place2(k22, support[0], support[1], n)
        else:
            if k211 is None:
                continue
            repeated = a.index(2)
            singles = [i for i in range(3) if a[i] == 1]
            block = k211.transpose(tuple(np.argsort((*singles, repeated))))
        for e01, e02, e12 in itertools.product(range(edge_order + 1), repeat=3):
            if e01 + e02 + e12 > edge_order:
                continue
            adjacency = [set() for _ in range(3)]
            for i, j in itertools.combinations(support, 2):
                adjacency[i].add(j)
                adjacency[j].add(i)
            for (i, j), count in zip(((0, 1), (0, 2), (1, 2)), (e01, e02, e12)):
                if count:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
            reached, pending = set(), [0]
            while pending:
                i = pending.pop()
                if i not in reached:
                    reached.add(i)
                    pending.extend(adjacency[i] - reached)
            if len(reached) != 3:
                continue
            degree = (a[0] + e01 + e02, a[1] + e01 + e12, a[2] + e02 + e12)
            coefficient = (k3._place1(coeffs[degree[0]], 0, n)
                           * k3._place1(coeffs[degree[1]], 1, n)
                           * k3._place1(coeffs[degree[2]], 2, n))
            edge = (k3._place2(powers[e01], 0, 1, n)
                    * k3._place2(powers[e02], 0, 2, n)
                    * k3._place2(powers[e12], 1, 2, n))
            denominator = math.prod(math.factorial(x) for x in a)
            out += coefficient * edge * block / denominator
    idx = np.arange(n)
    out[idx, idx, :] = 0.0
    out[idx, :, idx] = 0.0
    out[:, idx, idx] = 0.0
    return out


def joint_step(m, S, T, k4, k211=None, edge_order=1):
    mu, C, U = k3.state_to_h(m, S, T, kmax=3, emax=2, use_tau=True, k4=k4)
    if k211 is not None:
        sigma = np.sqrt(np.maximum(np.diag(S), 1e-20))
        rho = S / np.outer(sigma, sigma)
        coeffs = k3.relu_coeffs(m, sigma, 8)[1]
        U += fourth_birth(coeffs, rho, k4, k211, edge_order=edge_order)
    return mu, C, U


def intervention(weights, record, mode, edge_order=1):
    n = weights.shape[1]
    mu, C, U = np.zeros(n), np.eye(n), np.zeros((n, n, n))
    rows = []
    for index, W in enumerate(weights):
        layer = index + 1
        m, S, T = k3.linear_step(W.astype(float), mu, C, U)
        true_k4, true_k211 = standard_fourth(record, layer)
        if layer == 1:
            true_k4 = tuple(np.zeros_like(x) for x in true_k4)
            true_k211 = np.zeros((n, n, n)) if true_k211 is not None else None
        if mode == "radial":
            k4, k211 = radial_fourth(true_k4[0]), None
        elif mode == "true_pair":
            k4, k211 = true_k4, None
        elif mode == "true_joint":
            k4, k211 = true_k4, true_k211
        elif mode == "elliptical":
            sigma = np.sqrt(np.maximum(np.diag(S), 1e-20))
            k4, k211 = elliptical_fourth(true_k4[0], S / np.outer(sigma, sigma))
        else:
            raise ValueError(mode)
        mu, C, U = joint_step(m, S, T, k4, k211, edge_order=edge_order)
        rows.append(mu)
    return np.stack(rows)


def relative(a, b):
    denominator = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denominator) if denominator else None


def assess(args):
    weights = np.load(args.network)["weights"]
    records = [np.load(Path(f"{args.mc_stem}_rep{i}.npz")) for i in range(2)]
    depth = weights.shape[0]
    truths = [np.stack([r[f"hmean_{layer}"] for layer in range(1, depth + 1)]) for r in records]
    # Equal independent replicate sizes: this estimates noise of their average and each individual target.
    average_noise = np.mean((truths[0] - truths[1])**2, axis=1) / 4.0
    target_noise = 2.0 * average_noise
    k3.GROUPS = set(GROUPS)
    result = dict(network=str(args.network), mc_stem=str(args.mc_stem),
                  average_reference_noise=average_noise.tolist(), per_replicate_reference_noise=target_noise.tolist(),
                  modes={}, one_step={})
    for mode in ("radial", "true_pair", "true_joint", "elliptical"):
        cross_losses = []
        for source in range(2):
            pred = intervention(weights, records[source], mode, edge_order=args.edge_order)
            cross_losses.append(np.mean((pred - truths[1 - source])**2, axis=1))
        raw = np.mean(cross_losses, axis=0)
        result["modes"][mode] = dict(crossfit_mse=raw.tolist(), debiased_mse=(raw - target_noise).tolist(),
                                      replicate_mse=[x.tolist() for x in cross_losses])
        print(f"{mode}: final crossfit={raw[-1]:.4e} noise={target_noise[-1]:.3e} debiased={raw[-1]-target_noise[-1]:.4e}", flush=True)
    for layer in range(1, depth + 1):
        local = []
        for source in range(2):
            rec, target = records[source], records[1 - source]
            m, S, T = rec[f"m_{layer}"], rec[f"S_{layer}"], rec[f"T_{layer}"]
            true_k4, true_k211 = standard_fourth(rec, layer)
            if layer == 1:
                m, S = np.zeros(weights.shape[1]), weights[0].astype(float).T @ weights[0]
                T = np.zeros_like(T)
                true_k4 = tuple(np.zeros_like(x) for x in true_k4)
                true_k211 = None
            sigma = np.sqrt(np.maximum(np.diag(S), 1e-20))
            choices = {"radial": (radial_fourth(true_k4[0]), None),
                       "true_pair": (true_k4, None), "true_joint": (true_k4, true_k211),
                       "elliptical": elliptical_fourth(true_k4[0], S / np.outer(sigma, sigma))}
            row = {}
            for mode, (k4, k211) in choices.items():
                mu, C, U = joint_step(m, S, T, k4, k211, edge_order=args.edge_order)
                row[mode] = dict(mean_mse=float(np.mean((mu - target[f"hmean_{layer}"])**2)),
                                 covariance_relative=relative(C, target[f"hcov_{layer}"]),
                                 third_relative=relative(U, target[f"hthird_{layer}"]))
            local.append(row)
        result["one_step"][str(layer)] = local
    from scale_mixture import propagate
    for mode in ("analytic", "oracle"):
        values = []
        traces = []
        for source in range(2):
            omega = "analytic" if mode == "analytic" else [
                0.0 if layer == 1 else float(np.mean(standard_fourth(records[source], layer)[0][0]))
                for layer in range(1, depth + 1)]
            output = propagate(weights, omega=omega, nodes=args.nodes, kmax=3, emax=2)
            pred = output["prediction"]
            values.append(np.mean((pred - truths[1 - source])**2, axis=1))
            traces.append([
                dict(omega=row["omega"], omega_next_raw=row["omega_next_raw"])
                for row in output["trace"]
            ])
        raw = np.mean(values, axis=0)
        result["modes"][f"mixture_{mode}"] = dict(crossfit_mse=raw.tolist(), debiased_mse=(raw-target_noise).tolist(),
                                                  replicate_mse=[x.tolist() for x in values], trace=traces)
        print(f"mixture_{mode}: final crossfit={raw[-1]:.4e} noise={target_noise[-1]:.3e} debiased={raw[-1]-target_noise[-1]:.4e}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")
    print(f"saved {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    p = commands.add_parser("assess")
    p.add_argument("network", type=Path)
    p.add_argument("mc_stem", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--edge-order", type=int, default=1)
    args = parser.parse_args()
    prepare() if args.command == "prepare" else assess(args)


if __name__ == "__main__":
    main()
