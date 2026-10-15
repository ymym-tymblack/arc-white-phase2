"""Deterministic, small-width matrix-core fourth-cumulant closure.

The carried fourth state is two symmetric n-by-n matrices, with
    K_abcd = Sym6(G_ab D_cd).
Starting from standard Gaussian input uses G=0, D=I. A linear layer x @ W
carries this *represented tensor* exactly by G <- W.T @ G @ W and likewise
D. ``harmonic4.core_slices`` supplies its (4), (2,2), (3,1), and (2,1,1)
slices without ever forming an n**4 tensor.

The ReLU step retains the same diagrams as closure_experiment.joint_step:
Gaussian pair diagrams through kmax covariance edges, connected Gaussian
third diagrams through emax total edges, one incoming third-cumulant block,
and one incoming fourth block. Pair fourth-block terms use kmax edges;
three-vertex fourth-block terms use edge_order total edges (default one).
Marginal raw means additionally retain the existing tau_diag**2 correction.
There are no nonlocal two-third-block diagrams, two-fourth-block diagrams,
third/fourth mixed diagrams, or higher incoming cumulants. The heterogeneous
three-vertex calculation for (h_i,h_j,h_k**2) uses exactly these same retained
blocks. Coincident indices instead use raw marginal powers and two-vertex
power covariances, not a distinct-index formula evaluated on a diagonal.

After activation we compute K211_ijk = cum(h_i,h_j,h_k,h_k), including all
coincidences, and standardize by s_i*s_j*var_k. Its trace H_ij=sum_k K211_ijk
is projected with harmonic4.project_trace to Gamma. The next carried state
is G=diag(s) @ Gamma @ diag(s), D=diag(var). This preserves the standardized
fourth trace and discards the trace-free fourth harmonic. Consequently this
is NOT an exact fixed-W fourth-cumulant closure after the nonlinear
projection. No Monte Carlo samples, fitted coefficients, eigenvalue repair,
cumulant clipping, or variance floors are used. Nonpositive marginal
variances raise because this standardized projection is undefined there;
indefinite covariance/core matrices with positive diagonals are not fixed.
The k3_dense.GROUPS research ablations are respected.

Import with the research directory on sys.path, as for harmonic4 and
closure_experiment. ``propagate(weights,kmax=3,emax=2,edge_order=1)`` returns
prediction (depth,n) and a per-layer trace. ``state_to_h`` exposes one full
nonlinear/projection step. Extra fourth-state storage is O(n**2); temporary
slices, third tensors, and retained trace entries are O(n**3) per layer.
The hot spots are the connected-diagram loops over n**3 arrays and the
existing dense third-tensor linear contraction (O(n**4) at square width).
Pair-power work is O(kmax*n**2); the two core congruences are O(n**3).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

import k3_dense as k3
from harmonic4 import core_slices, project_trace
from scale_mixture import relu_power_coeffs


_PAIRS = ((0, 1), (0, 2), (1, 2))


def _edge_counts(order):
    for counts in itertools.product(range(order + 1), repeat=3):
        if sum(counts) <= order:
            yield counts


def _connected(support, counts):
    """Connectivity after treating the cumulant block as a hyperedge."""
    adjacency = [set() for _ in range(3)]
    for i, j in itertools.combinations(support, 2):
        adjacency[i].add(j)
        adjacency[j].add(i)
    for (i, j), count in zip(_PAIRS, counts):
        if count:
            adjacency[i].add(j)
            adjacency[j].add(i)
    reached, pending = set(), [0]
    while pending:
        vertex = pending.pop()
        if vertex not in reached:
            reached.add(vertex)
            pending.extend(adjacency[vertex] - reached)
    return len(reached) == 3


def _diagram(coefficients, powers, allocation, counts, n):
    """Heterogeneous vertex coefficients times factorial-weighted edges."""
    e01, e02, e12 = counts
    degrees = (allocation[0] + e01 + e02,
               allocation[1] + e01 + e12,
               allocation[2] + e02 + e12)
    term = (k3._place1(coefficients[0][degrees[0]], 0, n)
            * k3._place1(coefficients[1][degrees[1]], 1, n)
            * k3._place1(coefficients[2][degrees[2]], 2, n))
    for (i, j), count in zip(_PAIRS, counts):
        if count:
            term *= k3._place2(powers[count], i, j, n)
    return term


def _clear_repeats(tensor):
    idx = np.arange(tensor.shape[0])
    tensor[idx, idx, :] = 0.0
    tensor[idx, :, idx] = 0.0
    tensor[:, idx, idx] = 0.0


def third_distinct(coefficients, rho, tau3, emax=2):
    """Connected third cumulants for three possibly different functions.

    ``coefficients[v][degree, neuron]`` supplies the Hermite sequence at
    vertex v. This is the heterogeneous extension of k3.third_distinct:
    covariance edges have total degree <= emax and at most one tau block
    is present. Repeated-coordinate entries are zero, to be replaced by
    two-vertex power cumulants by the caller.
    """
    n = rho.shape[0]
    powers = k3.rho_pows(rho, emax)
    out = np.zeros((n, n, n))
    counts_list = tuple(_edge_counts(emax))
    if "rho" in k3.GROUPS:
        for counts in counts_list:
            if _connected((), counts):
                out += _diagram(coefficients, powers, (0, 0, 0), counts, n)

    t_iij = np.einsum("iij->ij", tau3)
    tdiag = np.einsum("iii->i", tau3)
    for allocation in itertools.product(range(4), repeat=3):
        if sum(allocation) != 3:
            continue
        support = tuple(i for i, degree in enumerate(allocation) if degree)
        if max(allocation) == 3:
            if "txxx" not in k3.GROUPS:
                continue
            block = k3._place1(tdiag, allocation.index(3), n)
        elif max(allocation) == 2:
            if "txxy" not in k3.GROUPS:
                continue
            block = k3._place2(t_iij, allocation.index(2), allocation.index(1), n)
        else:
            block = tau3
        denominator = math.prod(math.factorial(degree) for degree in allocation)
        for counts in counts_list:
            if max(allocation) == 1:
                group = "tau1" if sum(counts) == 0 else "tau1e"
                if group not in k3.GROUPS:
                    continue
            if _connected(support, counts):
                out += _diagram(coefficients, powers, allocation, counts, n) * block / denominator
    _clear_repeats(out)
    return out


def fourth_birth(coefficients, rho, k4, k211, edge_order=1):
    """Local heterogeneous counterpart of closure_experiment.fourth_birth.

    Include every connected diagram with one fourth block and at most
    edge_order covariance edges. The coefficients may differ at all three
    vertices; in particular (c^(1),c^(1),c^(2)) gives the fourth contribution
    to cum(h_i,h_j,h_k**2). Coincident coordinates are zero here.
    """
    n = rho.shape[0]
    out = np.zeros((n, n, n))
    omega, k22, k31 = k4
    if not any(np.any(block) for block in (*k4, k211)):
        return out
    powers = k3.rho_pows(rho, edge_order)
    counts_list = tuple(_edge_counts(edge_order))
    for allocation in itertools.product(range(5), repeat=3):
        if sum(allocation) != 4:
            continue
        support = tuple(i for i, degree in enumerate(allocation) if degree)
        if max(allocation) == 4:
            block = k3._place1(omega, allocation.index(4), n)
        elif max(allocation) == 3:
            block = k3._place2(k31, allocation.index(3), allocation.index(1), n)
        elif sorted(allocation) == [0, 2, 2]:
            block = k3._place2(k22, support[0], support[1], n)
        else:
            repeated = allocation.index(2)
            singles = tuple(i for i in range(3) if allocation[i] == 1)
            block = k211.transpose(tuple(np.argsort((*singles, repeated))))
        denominator = math.prod(math.factorial(degree) for degree in allocation)
        for counts in counts_list:
            if _connected(support, counts):
                out += _diagram(coefficients, powers, allocation, counts, n) * block / denominator
    _clear_repeats(out)
    return out


def _sigma(variance, stage):
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError(
            f"{stage}: standardized harmonic closure requires finite positive "
            f"marginal variances; minimum={np.min(variance)!r}; no clipping applied"
        )
    return np.sqrt(variance)


def state_to_h(m, S, T, G, D, *, kmax=3, emax=2, edge_order=1):
    """Apply ReLU, compute fourth slices, and project the next matrix core.

    G,D represent the incoming *unstandardized* fourth tensor. Returned
    G,D are the projected postactivation core, not its preactivation copy.
    K4,K22,K31,K211 are unprojected, unstandardized postactivation slices.
    ``fourth_next`` contains Gamma and both the measured and projected
    standardized fourth traces, including the projection residual norms.
    ``k4_pre`` and ``k211_pre`` are the standardized incoming slices.
    """
    for name, order in (("kmax", kmax), ("emax", emax), ("edge_order", edge_order)):
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    m, S, T, G, D = (np.asarray(x, dtype=np.float64) for x in (m, S, T, G, D))
    n = m.size
    variance_pre = np.diag(S)
    sigma = _sigma(variance_pre, "preactivation")
    rho = S / np.outer(sigma, sigma)
    scale3 = sigma[:, None, None] * sigma[None, :, None] * sigma[None, None, :]
    tau3 = T / scale3
    tdiag = np.einsum("iii->i", tau3)
    t_iij, t_ijj = np.einsum("iij->ij", tau3), np.einsum("ijj->ij", tau3)
    raw_k4, raw_k211 = core_slices(G, D)
    k4 = (raw_k4[0] / variance_pre**2,
          raw_k4[1] / np.outer(variance_pre, variance_pre),
          raw_k4[2] / (variance_pre[:, None] * np.outer(sigma, sigma)))
    k211 = raw_k211 / (sigma[:, None, None] * sigma[None, :, None] * variance_pre[None, None, :])
    # Pair fourth blocks need degree kmax+4; marginal tau**2 needs degree 6.
    degree = max(kmax + 4, emax + 3, edge_order + 4, 6)
    coefficients = relu_power_coeffs(m, sigma, degree)
    raw = {p: k3.corrected_raw_mean(coefficients[p], tdiag, True, k4[0])
           for p in (1, 2, 3, 4)}
    mu = raw[1]
    variance = raw[2] - mu**2
    sigma_next = _sigma(variance, "postactivation")

    def pair(p, q):
        return k3.pair_cov(coefficients[p], coefficients[q], rho,
                           t_iij, t_ijj, tdiag, tdiag, kmax, True, k4)

    C = pair(1, 1)
    np.fill_diagonal(C, variance)
    cov21 = pair(2, 1)
    np.fill_diagonal(cov21, raw[3] - raw[2] * mu)
    repeated_third = cov21 - 2.0 * mu[:, None] * C
    U = third_distinct((coefficients[1],) * 3, rho, tau3, emax)
    U += fourth_birth((coefficients[1],) * 3, rho, k4, k211, edge_order)
    marginal_third = raw[3] - 3.0 * mu * raw[2] + 2.0 * mu**3
    for i in range(n):
        if "iik" in k3.GROUPS:
            U[i, i, :] = repeated_third[i]
            U[i, :, i] = repeated_third[i]
            U[:, i, i] = repeated_third[i]
        if "iii" in k3.GROUPS:
            U[i, i, i] = marginal_third[i]

    cov22 = pair(2, 2)
    np.fill_diagonal(cov22, raw[4] - raw[2]**2)
    K22 = (cov22 - 2.0 * mu[:, None] * cov21.T - 2.0 * mu[None, :] * cov21
           + 4.0 * np.outer(mu, mu) * C - 2.0 * C**2)
    cov31 = pair(3, 1)
    np.fill_diagonal(cov31, raw[4] - raw[3] * mu)
    K31 = cov31 - 3.0 * mu[:, None] * cov21 + 3.0 * (mu**2 - variance)[:, None] * C
    K4 = (raw[4] - 4.0 * mu * raw[3] + 6.0 * mu**2 * raw[2]
          - 3.0 * mu**4 - 3.0 * variance**2)
    np.fill_diagonal(K22, K4)
    np.fill_diagonal(K31, K4)

    heterogeneous = (coefficients[1], coefficients[1], coefficients[2])
    K211 = third_distinct(heterogeneous, rho, tau3, emax)
    K211 += fourth_birth(heterogeneous, rho, k4, k211, edge_order)
    K211 -= 2.0 * mu[None, None, :] * U
    K211 -= 2.0 * C[:, None, :] * C[None, :, :]
    # Each coincidence is a genuine fourth slice, not a three-vertex limit.
    for i in range(n):
        K211[i, i, :] = K22[i]
        K211[i, :, i] = K31[i]
        K211[:, i, i] = K31[i]
        K211[i, i, i] = K4[i]

    standardized = K211 / (sigma_next[:, None, None] * sigma_next[None, :, None]
                           * variance[None, None, :])
    H = np.sum(standardized, axis=2)
    Gamma = project_trace(standardized)
    H_projected = (n + 4) * Gamma + np.eye(n) * np.trace(Gamma)
    residual_norm = float(np.linalg.norm(H_projected - H))
    trace_norm = float(np.linalg.norm(H))
    fourth_next = {
        "Gamma": Gamma, "H": H, "H_projected": H_projected,
        "trace": float(np.trace(H)),
        "projected_trace": float(np.trace(H_projected)),
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / trace_norm if trace_norm else (0.0 if residual_norm == 0.0 else float("nan")),
    }
    return {
        "mu": mu, "C": C, "U": U, "raw": raw,
        "K4": K4, "K22": K22, "K31": K31, "K211": K211,
        "G": np.outer(sigma_next, sigma_next) * Gamma, "D": np.diag(variance),
        "k4_pre": k4, "k211_pre": k211, "fourth_next": fourth_next,
    }


def propagate(weights, kmax=3, emax=2, edge_order=1):
    """Propagate standard Gaussian input through x @ W and ReLU per layer.

    ``prediction`` is (depth,n). Each trace row contains flat m,S,T,G,D
    preactivation state and mu,C,U postactivation state, the unprojected
    postactivation fourth slices, G_next,D_next after projection, and
    fourth_next standardized trace diagnostics. Explicit pre/post mappings
    are also supplied; their arrays alias the flat entries, not copies.
    All trace entries remain available even though only G,D,mu,C,U are
    carried forward. No sampling or empirical moment data are used.
    """
    weights = tuple(weights)
    if not weights:
        raise ValueError("weights must contain at least one layer")
    first = np.asarray(weights[0])
    if first.ndim != 2:
        raise ValueError("each weight must be a two-dimensional (in,out) matrix")
    n = first.shape[0]
    mu, C, U = np.zeros(n), np.eye(n), np.zeros((n, n, n))
    G, D = np.zeros((n, n)), np.eye(n)
    prediction, trace = [], []
    for layer, weight in enumerate(weights, 1):
        W = np.asarray(weight, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != mu.size:
            raise ValueError(f"layer {layer}: weights must have shape ({mu.size},out)")
        m, S, T = k3.linear_step(W, mu, C, U)
        G_pre, D_pre = W.T @ G @ W, W.T @ D @ W
        post = state_to_h(m, S, T, G_pre, D_pre,
                          kmax=kmax, emax=emax, edge_order=edge_order)
        mu, C, U, G, D = (post[key] for key in ("mu", "C", "U", "G", "D"))
        pre = {"m": m, "S": S, "T": T, "G": G_pre, "D": D_pre,
               "k4": post["k4_pre"], "k211": post["k211_pre"]}
        trace.append({
            "layer": layer, "m": m, "S": S, "T": T, "G": G_pre, "D": D_pre,
            "mu": mu, "C": C, "U": U, "G_next": G, "D_next": D,
            "K4": post["K4"], "K22": post["K22"], "K31": post["K31"],
            "K211": post["K211"], "fourth_next": post["fourth_next"],
            "pre": pre, "post": post,
        })
        prediction.append(mu)
    return {"prediction": np.stack(prediction), "trace": trace}
