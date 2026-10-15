"""κ4 の行列コアへの直交射影を測定値で検証する(研究用、提出には含めない)。

標準化座標で K_abcd ≈ Sym6(G_ab δ_cd) と置く。H_ab=Σ_c K_abcc なら
H=(n+4)G+tr(G)I、従って G=(H-tr(H)I/(2(n+2)))/(n+4)。
これは4次テンソルの2重トレースを保つ射影であり、一般のκ4を厳密には表さない。
  uv run python research/harmonic4.py NETWORK MC_STEM --out RESULT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import k3_dense as k3
from closure_experiment import GROUPS, joint_step, radial_fourth, standard_fourth


def project_trace(k211):
    n = k211.shape[0]
    H = np.sum(k211, axis=2)
    H = (H + H.T) * .5
    return (H - np.eye(n) * np.trace(H) / (2.0 * (n + 2))) / (n + 4)


def core_slices(G, D):
    """Sym6(G⊗D) の (4),(2,2),(3,1),(2,1,1) 切り口。G,D は対称行列。"""
    gd, dd = np.diag(G), np.diag(D)
    k4 = 6.0 * gd * dd
    k22 = np.outer(gd, dd) + np.outer(dd, gd) + 4.0 * G * D
    k31 = 3.0 * (gd[:, None] * D + dd[:, None] * G)
    k211 = (G[:, :, None] * dd[None, None, :]
            + 2.0 * G[:, None, :] * D[None, :, :]
            + 2.0 * G[None, :, :] * D[:, None, :]
            + D[:, :, None] * gd[None, None, :])
    return (k4, k22, k31), k211


def oracle_propagate(weights, record, mode):
    n = weights.shape[1]
    mu, C, U = np.zeros(n), np.eye(n), np.zeros((n, n, n))
    rows = []
    for index, W in enumerate(weights):
        layer = index + 1
        m, S, T = k3.linear_step(W.astype(float), mu, C, U)
        true_k4, true_k211 = standard_fourth(record, layer)
        if layer == 1:
            true_k4 = tuple(np.zeros_like(x) for x in true_k4)
            true_k211 = np.zeros_like(true_k211)
        if mode == "radial_scalar":
            omega = np.full(n, np.mean(true_k4[0]))
            k4, k211 = radial_fourth(omega), None
        else:
            G = project_trace(true_k211)
            projected, k211 = core_slices(G, np.eye(n))
            k4 = projected if mode == "projected" else true_k4
        mu, C, U = joint_step(m, S, T, k4, k211)
        rows.append(mu)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path)
    parser.add_argument("mc_stem", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--analytic", action="store_true", help="run the no-MC matrix-core recurrence instead of oracle projections")
    args = parser.parse_args()
    weights = np.load(args.network)["weights"]
    records = [np.load(f"{args.mc_stem}_rep{i}.npz") for i in (0, 1)]
    truths = [np.stack([r[f"hmean_{l}"] for l in range(1, len(weights) + 1)]) for r in records]
    noise = np.mean((truths[0] - truths[1])**2, axis=1) / 2.0
    k3.GROUPS = set(GROUPS)
    result = dict(network=str(args.network), reference_noise=noise.tolist(), modes={})
    if args.analytic:
        from harmonic_closure import propagate
        output = propagate(weights)
        average_truth = (truths[0] + truths[1]) * .5
        raw = np.mean((output["prediction"] - average_truth)**2, axis=1)
        average_noise = noise * .5
        result["modes"]["analytic"] = dict(
            mse=raw.tolist(), debiased_mse=(raw-average_noise).tolist(),
            reference_noise=average_noise.tolist(),
            trace=[dict(
                omega=float(np.mean(row["pre"]["k4"][0])),
                fourth_trace_residual=row["fourth_next"]["relative_residual"],
                minimum_variance=float(np.min(np.diag(row["C"]))),
            ) for row in output["trace"]],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"analytic matrix core: final MSE={raw[-1]:.4e} debiased={raw[-1]-average_noise[-1]:.4e}", flush=True)
        return
    for mode in ("radial_scalar", "projected", "birth_trace"):
        losses = []
        for source in (0, 1):
            prediction = oracle_propagate(weights, records[source], mode)
            losses.append(np.mean((prediction - truths[1 - source])**2, axis=1))
        raw = np.mean(losses, axis=0)
        result["modes"][mode] = dict(crossfit_mse=raw.tolist(), debiased_mse=(raw-noise).tolist(),
                                      replicate_mse=[x.tolist() for x in losses])
        print(f"{mode}: final crossfit={raw[-1]:.4e} debiased={raw[-1]-noise[-1]:.4e}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
