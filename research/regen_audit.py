"""Deterministic tiny-tensor audit of directional fourth-cumulant regeneration.

Run ``python research/regen_audit.py`` for algebra alone.  Optionally add
``--network weights.npz --kwargs '{"regen_p0":0.8}'`` to inspect the actual
candidate's collected recurrence.  This program never fetches data or samples
an input distribution; finite weighted fixtures are exact discrete moments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class Audit:
    def __init__(self):
        self.records = []

    def compare(self, name, actual, expected, *, atol=2e-11, rtol=2e-11):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise ValueError(f"{name}: shapes {actual.shape} != {expected.shape}")
        error = np.abs(actual - expected)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        maximum = float(np.max(error)) if error.size else 0.0
        ok = bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                  and np.all(error <= atol + rtol * np.abs(expected)))
        self.records.append(dict(name=name, max_abs=maximum,
                                 relative_to_max=maximum / max(scale, 1e-30), ok=ok))

    def sensitive(self, name, wrong, expected):
        gap = float(np.max(np.abs(np.asarray(wrong) - expected)))
        self.records.append(dict(name=name, max_abs=gap, ok=bool(gap > 1e-8),
                                 interpretation="wrong normalization must disagree"))


def sym6(core, metric):
    """Explicit normalized tensor, reserved for tiny fixtures only."""
    n = len(metric)
    tensor = np.empty((n, n, n, n), dtype=np.float64)
    for a, b, c, d in np.ndindex(tensor.shape):
        tensor[a, b, c, d] = (
            core[a, b] * metric[c] * (c == d)
            + core[a, c] * metric[b] * (b == d)
            + core[a, d] * metric[b] * (b == c)
            + core[b, c] * metric[a] * (a == d)
            + core[b, d] * metric[a] * (a == c)
            + core[c, d] * metric[a] * (a == b)
        ) / 6.0
    return tensor


def audit_slices(audit, rng):
    n = 5
    w = rng.normal(size=(n, n)) * np.linspace(0.25, 0.7, n)
    q = rng.normal(size=(n, n))
    covariance = q @ q.T + np.diag(np.linspace(0.4, 1.2, n))
    metric = (w * w).sum(axis=0)
    g = np.linspace(-0.8, 1.3, n)
    lam = -0.37
    core = lam * covariance
    np.fill_diagonal(core, g)
    tensor = sym6(core, metric)
    diagonal = np.array([tensor[i, i, i, i] for i in range(n)])
    k22 = np.array([[tensor[i, i, j, j] for j in range(n)] for i in range(n)])
    k31 = np.array([[tensor[i, i, i, j] for j in range(n)] for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    audit.compare("Sym6 diagonal", diagonal, metric * g)
    formula22 = (np.outer(metric, g) + np.outer(g, metric)) / 6.0
    formula31 = metric[:, None] * lam * covariance / 2.0
    audit.compare("Sym6 K22 offdiagonal", k22[off], formula22[off])
    audit.compare("Sym6 K31 offdiagonal", k31[off], formula31[off])
    actual211, expected211 = [], []
    for i, j, k in np.ndindex((n, n, n)):
        if len({i, j, k}) == 3:
            actual211.append(tensor[i, j, k, k])
            expected211.append(metric[k] * lam * covariance[i, j] / 6.0)
    audit.compare("Sym6 K211 distinct", actual211, expected211)
    var = covariance.diagonal()
    sigma = np.sqrt(var)
    rho = covariance / np.outer(sigma, sigma)
    audit.compare("K31 standardization", (k31 / np.outer(sigma**3, sigma))[off],
                  (lam * (metric / var)[:, None] * rho / 2.0)[off])
    audit.sensitive("detect missing Sym6 /6", 6.0 * diagonal, diagonal)
    audit.sensitive("detect K31 missing /2", 2.0 * k31[off], k31[off])
    return covariance, metric, lam, tensor


def audit_projection(audit, rng):
    n = 5
    # Unequal positive masses and offsets make raw/standardized projections distinct.
    points = np.maximum(rng.normal(size=(23, n)) * np.linspace(0.5, 2.3, n)
                        + np.linspace(-0.6, 1.1, n), 0.0)
    mass = np.arange(1.0, 24.0)
    mass /= mass.sum()
    mu = mass @ points
    centered = points - mu
    covariance = np.einsum("s,si,sj->ij", mass, centered, centered)
    fourth = np.einsum("s,si,sj,sk,sl->ijkl", mass, centered, centered, centered, centered)
    fourth -= (np.einsum("ij,kl->ijkl", covariance, covariance)
               + np.einsum("ik,jl->ijkl", covariance, covariance)
               + np.einsum("il,jk->ijkl", covariance, covariance))
    powers = [mass @ points**p for p in range(1, 5)]
    cov21 = np.einsum("s,si,sj->ij", mass, points**2, points) - np.outer(powers[1], mu)
    cov22 = np.einsum("s,si,sj->ij", mass, points**2, points**2) - np.outer(powers[1], powers[1])
    k22 = (cov22 - 2.0 * mu[:, None] * cov21.T - 2.0 * mu[None, :] * cov21
           + 4.0 * np.outer(mu, mu) * covariance - 2.0 * covariance**2)
    var = covariance.diagonal()
    k4 = powers[3] - 4 * mu * powers[2] + 6 * mu**2 * powers[1] - 3 * mu**4 - 3 * var**2
    exact22 = np.array([[fourth[i, i, j, j] for j in range(n)] for i in range(n)])
    audit.compare("raw postactivation K22", k22, exact22)
    audit.compare("raw postactivation K4", k4, exact22.diagonal())
    np.fill_diagonal(k22, 0.0)
    trace = k4 + k22.sum(axis=1)
    projected = 6.0 * trace / (n + 4) - 3.0 * trace.sum() / ((n + 2) * (n + 4))
    # Independent full Frobenius least-squares projection over every symmetric
    # core entry, not a second evaluation of the trace formula.
    basis = []
    positions = []
    for i in range(n):
        for j in range(i, n):
            core = np.zeros((n, n))
            core[i, j] = core[j, i] = 1.0
            basis.append(sym6(core, np.ones(n)).ravel())
            positions.append((i, j))
    coefficients = np.linalg.lstsq(np.stack(basis, axis=1), fourth.ravel(), rcond=None)[0]
    expected = np.array([coefficients[positions.index((i, i))] for i in range(n)])
    audit.compare("raw core diagonal vs dense Frobenius projection", projected, expected)
    audit.sensitive("detect forbidden postvariance division", projected / var, expected)


def audit_transport(audit, rng):
    n = 5
    w = rng.normal(size=(n, n)) / np.sqrt(n)
    q = rng.normal(size=(n, n))
    previous_covariance = q @ q.T + np.eye(n)
    previous_variance = previous_covariance.diagonal()
    pre_covariance = w.T @ previous_covariance @ w
    g_previous = np.linspace(-0.4, 1.6, n)
    ww = w * w
    tg = ww.T @ g_previous
    tv = pre_covariance.diagonal() - ww.T @ previous_variance
    strength = 0.43
    denominator = pre_covariance.diagonal().mean() - strength * tv.mean()
    if denominator <= 0:
        raise ValueError("tiny deterministic transport fixture has unsafe denominator")
    lam = strength * tg.mean() / denominator
    previous_core = lam * previous_covariance
    np.fill_diagonal(previous_core, g_previous)
    dense = w.T @ previous_core @ w
    dg = tg + lam * tv
    audit.compare("exact diagonal core transport", dg, dense.diagonal())
    audit.compare("regenerated lambda fixed point", np.asarray(lam),
                  np.asarray(strength * dense.diagonal().mean() / pre_covariance.diagonal().mean()))


def slices3(tensor):
    n = len(tensor)
    return (np.array([tensor[i, i, i] for i in range(n)]),
            np.array([[tensor[i, i, j] for j in range(n)] for i in range(n)]))


def audit_feed(audit, rng, covariance, metric, lam, fourth):
    n = len(metric)
    var = covariance.diagonal()
    sigma = np.sqrt(var)
    rho = covariance / np.outer(sigma, sigma)
    feed = 1.4
    # Hermite coefficients deliberately unequal and not standardized derivatives.
    c1 = np.linspace(0.17, 1.31, n)
    c2 = np.array([0.31, -0.27, 0.81, 0.43, -0.19])
    g = (lam * feed / 24.0) * c1
    u_birth = metric * c2 / var
    a_birth = c1[:, None] * rho
    x_birth = (a_birth - np.diag(c1)) * g
    kernel = x_birth + x_birth.T
    np.fill_diagonal(kernel, 0.0)
    # Build distinct birth entries directly from the dense fourth tensor and
    # derivatives, including the 1/2 from the doubled Hermite index.
    birth = np.zeros((n, n, n))
    for i, j, k in np.ndindex(birth.shape):
        if len({i, j, k}) == 3:
            birth[i, j, k] = feed * 0.5 * (
                fourth[i, i, j, k] * c2[i] / var[i] * c1[j] / sigma[j] * c1[k] / sigma[k]
                + fourth[j, j, i, k] * c2[j] / var[j] * c1[i] / sigma[i] * c1[k] / sigma[k]
                + fourth[k, k, i, j] * c2[k] / var[k] * c1[i] / sigma[i] * c1[j] / sigma[j])
        elif i == j == k:
            birth[i, j, k] = 0.0
        else:
            repeated = i if i == j or i == k else j
            other = k if i == j else (j if i == k else i)
            birth[i, j, k] = 2.0 * kernel[repeated, other] * u_birth[repeated]
    represented = (kernel[:, :, None] * u_birth[None, None, :]
                   + kernel[:, None, :] * u_birth[None, :, None]
                   + kernel[None, :, :] * u_birth[:, None, None])
    audit.compare("birth distinct plus repeated correction", represented, birth)
    b3, b21 = slices3(birth)
    audit.compare("birth Uiii zero", b3, np.zeros(n))
    audit.compare("birth Uiik counted once", b21, 2.0 * kernel * u_birth[:, None])
    # Two independent families catch errors hidden by single-family broadcasting.
    transforms = [rng.normal(size=(n, n)) / np.sqrt(n), rng.normal(size=(n, n)) / np.sqrt(n)]
    for mode in ("young", "shared"):
        if mode == "shared":
            shared, _ = np.linalg.qr(rng.normal(size=(n, 3)))
            factors = [shared.T @ m for m in transforms]
            matrices = [shared @ f for f in factors]
        else:
            matrices = transforms
        expected_tensor = sum(np.einsum("ai,bj,ck,ijk->abc", m, m, m, birth)
                              for m in matrices)
        expected3, expected21 = slices3(expected_tensor)
        d3 = np.zeros(n)
        d21 = np.zeros((n, n))
        wrong21 = np.zeros((n, n))
        for family, m in enumerate(matrices):
            a = m @ a_birth
            x = (a - m * c1) * g
            u = m @ u_birth
            kd = 2.0 * (x * m).sum(axis=1)
            d3 += 3.0 * kd * u
            xm = 2.0 * u[:, None] * (a * g - 2.0 * m * c1 * g)
            xa = 2.0 * u[:, None] * (m * g)
            if mode == "young":
                readout = xm @ m.T + xa @ a.T
            else:
                fm = factors[family]
                fa = fm @ a_birth
                readout = (xm @ fm.T + xa @ fa.T) @ shared.T
            d21 += readout + np.outer(kd, u)
            bad_u = m @ c2
            transported_kernel = x @ m.T + m @ x.T
            wrong21 += np.outer(kd, bad_u) + 2.0 * bad_u[:, None] * transported_kernel
        audit.compare(f"{mode} folded D3 vs transported dense tensor", d3, expected3)
        audit.compare(f"{mode} folded D21 vs transported dense tensor", d21, expected21)
        audit.compare(f"{mode} D21 diagonal equals D3", d21.diagonal(), d3)
        audit.sensitive(f"{mode} detect missing metric/var in u", wrong21, expected21)
        audit.sensitive(f"{mode} detect feed factor2", 2.0 * d21, expected21)


def audit_network(audit, args):
    # Imports and instrumented execution are needed only for this optional path.
    sys.path.insert(0, str(ROOT))
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import MLP, SetupContext

    with np.load(args.network, allow_pickle=False) as network:
        weights = np.asarray(network["weights"])
    if weights.ndim != 3 or weights.shape[1] != weights.shape[2]:
        raise ValueError("network NPZ must contain square weights[depth,input,output]")
    depth, n, _ = weights.shape
    spec = importlib.util.spec_from_file_location("directional_audit_candidate", args.candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--kwargs must be a JSON object")
    kwargs.setdefault("directional_k4", True)
    kwargs["collect"] = True
    if not kwargs["directional_k4"] or not kwargs.get("use_k4", True):
        raise ValueError("network recurrence audit requires directional_k4 and use_k4")
    est = module.Estimator(**kwargs)
    mlp = MLP(weights=[fnp.array(w, dtype=fnp.float32) for w in weights],
              width=n, depth=depth, seed=args.seed)
    est.setup(SetupContext(width=n, depth=depth, flop_budget=args.budget, api_version="1.0",
                           submission_dir=str(ROOT), seed=args.seed))
    try:
        with flops.BudgetContext(flop_budget=args.budget, wall_time_limit_s=args.wall, quiet=True) as context:
            prediction = est.predict(mlp, args.budget)
        prediction = np.asarray(prediction)
        features = [{key: np.asarray(value, dtype=np.float64) for key, value in item.items()
                     if value is not None} for item in est.features]
    finally:
        est.teardown()
    if len(features) != depth:
        raise ValueError(f"collect returned {len(features)} layers for depth={depth}")
    if not np.isfinite(prediction).all():
        raise ValueError("candidate prediction is nonfinite")
    g_prev, var_prev = np.zeros(n), np.ones(n)
    recurrence_layers, projection_layers = [], []
    for layer, (w, feature) in enumerate(zip(weights, features)):
        required = ("sigma", "k4_lambda", "k4_gdiag", "k4_metric", "k4_omega")
        missing = [name for name in required if name not in feature]
        if missing:
            raise ValueError(f"layer {layer}: collect lacks {missing}")
        ww = np.asarray(w, dtype=np.float64)**2
        var = feature["sigma"]**2
        tg = ww.T @ g_prev
        tv = var - ww.T @ var_prev
        strength = float(np.interp(layer, np.linspace(0, max(depth - 1, 1), 4),
                                  [getattr(est, f"regen_p{i}") for i in range(4)]))
        if not est.regen_start <= layer <= est.regen_end:
            strength = 0.0
        denominator = float(var.mean() - strength * tv.mean())
        if denominator <= 0:
            raise ValueError(f"layer {layer}: unsafe regeneration denominator {denominator}")
        lam = strength * tg.mean() / denominator
        dg = tg + lam * tv
        tolerance = dict(atol=args.atol, rtol=args.rtol)
        audit.compare(f"network layer{layer} lambda", feature["k4_lambda"], np.asarray(lam), **tolerance)
        audit.compare(f"network layer{layer} transported diagonal", feature["k4_gdiag"], dg, **tolerance)
        audit.compare(f"network layer{layer} metric", feature["k4_metric"], ww.sum(axis=0), **tolerance)
        audit.compare(f"network layer{layer} standardized omega", feature["k4_omega"],
                      ww.sum(axis=0) * dg / var**2, **tolerance)
        recurrence_layers.append(layer)
        if "k4_post_core_diag" in feature and "k4_post_K22" in feature:
            k22 = feature["k4_post_K22"]
            # Full K22 retains its true fourth diagonal for independent projection.
            h = k22.sum(axis=1)
            if np.all(k22.diagonal() == 0):
                if "k4_post_K4" not in feature:
                    raise ValueError(f"layer {layer}: zero-diagonal post_K22 requires k4_post_K4 for independent projection")
                h = h + feature["k4_post_K4"]
            projected = 6.0 * h / (n + 4) - 3.0 * h.sum() / ((n + 2) * (n + 4))
            audit.compare(f"network layer{layer} raw core projection", feature["k4_post_core_diag"], projected,
                          **tolerance)
            g_prev = feature["k4_post_core_diag"]
            var_prev = feature["var_post"]
            projection_layers.append(layer)
        elif layer != depth - 1:
            raise ValueError(f"layer {layer}: missing postactivation core/K22 needed for next recurrence")
    return dict(network=str(args.network), candidate=str(args.candidate), kwargs=kwargs,
                flops=int(context.flops_used), residual_wall_time_s=float(context.residual_wall_time_s),
                recurrence_layers=recurrence_layers, projection_layers=projection_layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=504211)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--candidate", type=Path, default=ROOT / "candidates/k3_factored.py")
    parser.add_argument("--kwargs", "--candidate-kwargs", default="{}", help="candidate constructor JSON")
    parser.add_argument("--budget", type=int, default=2**41)
    parser.add_argument("--wall", type=float, default=120.0)
    parser.add_argument("--atol", type=float, default=3e-5, help="actual-network absolute tolerance")
    parser.add_argument("--rtol", type=float, default=3e-4, help="actual-network relative tolerance")
    parser.add_argument("--output", type=Path, help="save the complete audit report")
    args = parser.parse_args()
    audit = Audit()
    rng = np.random.default_rng(args.seed)
    covariance, metric, lam, fourth = audit_slices(audit, rng)
    audit_projection(audit, rng)
    audit_feed(audit, rng, covariance, metric, lam, fourth)
    audit_transport(audit, rng)
    runtime = audit_network(audit, args) if args.network else None
    ok = all(record["ok"] for record in audit.records)
    report = dict(ok=ok, seed=args.seed, checks=audit.records, runtime=runtime)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(ok=ok, checks=len(audit.records), runtime=runtime,
                              output=str(args.output)), allow_nan=False))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
