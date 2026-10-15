"""Deterministic small-width Gamma scale-mixture K3 closure (NumPy only).

For preactivation state (m, S, T), use z = m + sqrt(V) Y, independently,
V ~ Gamma(a, rate=a), a=3/omega, E[Y]=0, Cov(Y)=S, and
cum3(Y)=T/E[V**1.5]. Conditional on a quadrature node v, the existing
``k3_dense.state_to_h`` receives (m, v*S, v**1.5*T/E[V**1.5]), with no
explicit fourth-cumulant correction. Thus the *specified input* covariance
and third cumulant are S and T after integrating V. This is a cumulant
closure, not a construction of a positive distribution for every S,T:
the conditional K3 diagram truncations and marginal tau/tau**2 corrections
are inherited unchanged, including k3_dense.GROUPS and variance clipping.

The Gamma contribution to cum4(z) is
    omega/3 * (S_ij*S_kl + S_ik*S_jl + S_il*S_jk).
For the next layer we replace the full fourth tensor by its rotationally
averaged radial part. Write Q=sum_ij cum4(h_i,h_i,h_j,h_j); then
    omega_next_raw = 3*Q / ((trace(C))**2 + 2*sum(C*C)).
Indeed contraction of the radial tensor gives exactly that denominator
multiplied by omega/3. Q and the denominator are orthogonal invariants.
This projection is NOT exact through a fixed weight matrix: it discards
orientation-dependent fourth cumulants before the next linear step. No
Monte Carlo samples, empirical layer tables, or fitted constants are used.

``propagate`` returns a dict containing ``prediction`` (depth,n), the full
per-layer ``trace``, used preactivation ``omega``, and ``omega_next_raw``.
The input is standard Gaussian, matching k3_dense.propagate. Oracle omega
values apply to preactivation layers, in order; a scalar applies to all
layers. Analytic propagation starts at initial_omega=0. A Gamma variance
cannot be negative: by default a negative analytic next value raises when
used at the next layer. Set nonnegative_omega=True explicitly to project
used values onto [0,infinity); raw estimates remain available unchanged.
A completely deterministic zero-fourth-cumulant state is assigned omega=0;
a zero denominator with nonzero Q yields NaN rather than a fabricated fit.

Examples (run from repository root)::

    from research import k3_dense, scale_mixture
    baseline = k3_dense.propagate(weights, emax=2)
    zero = scale_mixture.propagate(weights, omega=0.0, emax=2)
    oracle = scale_mixture.propagate(weights, omega=per_layer_omega,
                                     nodes=12, emax=2)
    analytic = scale_mixture.propagate(weights, omega="analytic", nodes=8,
                                       nonnegative_omega=True, emax=2)
    prediction = analytic["prediction"]
    first_K22 = analytic["trace"][0]["K22"]

omega=0 bypasses quadrature and calls the existing K3 step exactly. Nodes
are generalized Gauss-Laguerre nodes for the normalized Gamma measure,
computed via a scaled Jacobi matrix so tiny omega and large shape do not
require enormous Gamma functions. Noninteger moments are quadrature
approximations; the conditional T normalization uses the analytic moment,
not a quadrature-dependent replacement. Increase nodes to assess this
error. Main costs are ``nodes`` dense K3 steps (n**3 tensors and the
existing Hermite/diagram loops), then dense linear third-tensor propagation
(O(n**4) at square width). Extra fourth-moment work is O(nodes*kmax*n**2),
not n**4; quadrature keeps only O(nodes*n**2 + n**3) state storage.
"""

from __future__ import annotations

import math

import numpy as np

if __package__:
    from . import k3_dense as k3
else:
    import k3_dense as k3


def gamma_quadrature(omega: float, nodes: int = 8):
    """Return v, weights, E[V**1.5] for Gamma(3/omega, rate=3/omega).

    The Jacobi recurrence is Laguerre(alpha=a-1), divided by a before
    diagonalization. Eigenvector weights integrate the normalized measure,
    avoiding gamma(a) overflow. omega=0 is the exact point mass at one.
    """
    omega = float(omega)
    if not np.isfinite(omega) or omega < 0.0:
        raise ValueError(f"Gamma scale mixture requires finite omega >= 0, got {omega!r}")
    if not isinstance(nodes, (int, np.integer)) or nodes < 1:
        raise ValueError("nodes must be a positive integer")
    if omega == 0.0:
        return np.ones(1), np.ones(1), 1.0
    inv_a = omega / 3.0
    k = np.arange(nodes, dtype=np.float64)
    diagonal = 1.0 + 2.0 * k * inv_a
    r = np.arange(1, nodes, dtype=np.float64)
    off_diagonal = np.sqrt(r * inv_a * (1.0 + (r - 1.0) * inv_a))
    jacobi = np.diag(diagonal)
    jacobi += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    v, vectors = np.linalg.eigh(jacobi)
    weights = vectors[0] ** 2
    weights /= weights.sum()
    if np.any(v <= 0.0):
        raise FloatingPointError("Gamma quadrature lost positive nodes; reduce omega or node count")
    if inv_a < 1e-4:
        # log Gamma(a+3/2)-log Gamma(a)-(3/2)log(a), asymptotic
        # through a**-3; avoids subtracting nearly equal lgamma values.
        log_moment = inv_a * (3.0 / 8.0 + inv_a * (-1.0 / 8.0 + inv_a * 3.0 / 64.0))
    else:
        a = 1.0 / inv_a
        log_moment = math.lgamma(a + 1.5) - math.lgamma(a) - 1.5 * math.log(a)
    return v, weights, math.exp(log_moment)


def relu_power_coeffs(m, s, kmax: int, powers=(1, 2, 3, 4)):
    """Hermite coefficients of ReLU(z)**p, including p=4.

    I_p=E[ReLU(z)**p], I_0=Phi(m/s), and
    I_p=m*I_(p-1)+(p-1)*s**2*I_(p-2). For k<=p, c_k is
    p!/(p-k)! * s**k * I_(p-k); for k>p the distributional derivative
    gives p! * s**p * phi(m/s) * He_(k-p-1)(-m/s).
    """
    m, s = np.asarray(m, dtype=np.float64), np.asarray(s, dtype=np.float64)
    a = m / s
    ph = k3.npdf(a)
    moments = [k3.ncdf(a)]
    max_power = max(powers)
    if max_power >= 1:
        moments.append(m * moments[0] + s * ph)
    for p in range(2, max_power + 1):
        moments.append(m * moments[-1] + (p - 1) * s**2 * moments[-2])
    result = {}
    for p in powers:
        coefficients = np.empty((kmax + 1, m.size))
        factorial = math.factorial(p)
        for k in range(kmax + 1):
            if k <= p:
                coefficients[k] = (factorial / math.factorial(p - k)) * s**k * moments[p - k]
            else:
                coefficients[k] = factorial * s**p * ph * k3.he(k - p - 1, -a)
        result[p] = coefficients
    return result


def _conditional_power_moments(m, S, T, mu, kmax):
    """Raw powers 2..4 and E[h_i**2*h_j], E[h_i**2*h_j**2]."""
    s = np.sqrt(np.maximum(np.diag(S), 1e-12))
    rho = S / np.outer(s, s)
    tau = T / (s[:, None, None] * s[None, :, None] * s[None, None, :])
    tdiag = np.einsum("iii->i", tau)
    t_iij, t_ijj = np.einsum("iij->ij", tau), np.einsum("ijj->ij", tau)
    # Reuse exactly the existing coefficients for powers 1..3, extending
    # only the fourth power by the general derivative/moment recurrence.
    kc = max(kmax + 3, 6)
    cs = k3.relu_coeffs(m, s, kc)
    cs.update(relu_power_coeffs(m, s, kc, powers=(4,)))
    raw = {p: k3.corrected_raw_mean(cs[p], tdiag, True) for p in (2, 3, 4)}
    cov21 = k3.pair_cov(cs[2], cs[1], rho, t_iij, t_ijj, tdiag, tdiag, kmax, True, k4=None)
    cov22 = k3.pair_cov(cs[2], cs[2], rho, t_iij, t_ijj, tdiag, tdiag, kmax, True, k4=None)
    # Pair expansions are for distinct coordinates. Coincident indices
    # must instead use exact one-coordinate raw powers at this closure.
    np.fill_diagonal(cov21, raw[3] - raw[2] * mu)
    np.fill_diagonal(cov22, raw[4] - raw[2] ** 2)
    return raw, cov21 + np.outer(raw[2], mu), cov22 + np.outer(raw[2], raw[2])


def fourth_trace_omega(C, K22):
    """Unconstrained rotational scalar fit; no fixed-W exactness is implied."""
    numerator = 3.0 * float(np.sum(K22))
    denominator = float(np.trace(C) ** 2 + 2.0 * np.sum(C * C))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("nan")
    return numerator / denominator


def state_to_h(m, S, T, omega: float, *, nodes: int = 8, kmax: int = 6, emax: int = 3):
    """Return mu,C,U,K22 and raw next omega for one deterministic mixture.

    Total covariance is E[C_v]+E[d_v*d_v.T]. Total third cumulant is
    E[U_v]+E[d_i*C_v[j,k]+d_j*C_v[i,k]+d_k*C_v[i,j]+d_i*d_j*d_k],
    where d_v=mu_v-E[mu_v]. Merely averaging centered tensors is wrong.
    K22 is assembled from *unconditional* raw power moments, so covariance
    of conditional means of the powers is included automatically.
    """
    m, S, T = (np.asarray(x, dtype=np.float64) for x in (m, S, T))
    v, weights, vthreehalf = gamma_quadrature(omega, nodes)
    n = m.size
    means, covariances = [], []
    mean = np.zeros(n)
    covariance = np.zeros((n, n))
    third = np.zeros((n, n, n))
    raw = {p: np.zeros(n) for p in (2, 3, 4)}
    raw21, raw22 = np.zeros((n, n)), np.zeros((n, n))
    for scale, weight in zip(v, weights):
        if omega == 0.0:
            Sv, Tv = S, T
        else:
            Sv, Tv = scale * S, (scale**1.5 / vthreehalf) * T
        mu_v, C_v, U_v = k3.state_to_h(m, Sv, Tv, kmax, emax, True, k4=None)
        powers, pair21, pair22 = _conditional_power_moments(m, Sv, Tv, mu_v, kmax)
        means.append(mu_v)
        covariances.append(C_v)
        mean += weight * mu_v
        covariance += weight * C_v
        third += weight * U_v
        for p in raw:
            raw[p] += weight * powers[p]
        raw21 += weight * pair21
        raw22 += weight * pair22
    if omega != 0.0:
        for weight, mu_v, C_v in zip(weights, means, covariances):
            delta = mu_v - mean
            covariance += weight * np.outer(delta, delta)
            third += weight * (
                delta[:, None, None] * C_v[None, :, :]
                + delta[None, :, None] * C_v[:, None, :]
                + delta[None, None, :] * C_v[:, :, None]
                + delta[:, None, None] * delta[None, :, None] * delta[None, None, :]
            )
    cov21 = raw21 - np.outer(raw[2], mean)
    cov22 = raw22 - np.outer(raw[2], raw[2])
    K22 = (
        cov22 - 2.0 * mean[:, None] * cov21.T - 2.0 * mean[None, :] * cov21
        + 4.0 * np.outer(mean, mean) * covariance - 2.0 * covariance**2
    )
    marginal_k4 = (
        raw[4] - 4.0 * mean * raw[3] + 6.0 * mean**2 * raw[2]
        - 3.0 * mean**4 - 3.0 * np.diag(covariance) ** 2
    )
    np.fill_diagonal(K22, marginal_k4)
    return {
        "mu": mean, "C": covariance, "U": third, "K22": K22,
        "omega_next_raw": fourth_trace_omega(covariance, K22),
    }


def propagate(weights, omega="analytic", *, nodes: int = 8, kmax: int = 6,
              emax: int = 3, initial_omega: float = 0.0,
              nonnegative_omega: bool = False):
    """Propagate Gaussian input through x @ W then ReLU at every layer.

    omega is 'analytic', a scalar, or a length-depth sequence. Trace entries
    contain m,S,T,mu,C,U,K22,omega,omega_requested,omega_next_raw. With an
    explicit nonnegative_omega constraint only the used omega is projected;
    requested oracle values and every raw analytic estimate remain visible.
    """
    weights = tuple(weights)
    if not weights:
        raise ValueError("weights must contain at least one layer")
    analytic = isinstance(omega, str) and omega == "analytic"
    if isinstance(omega, str) and not analytic:
        raise ValueError("omega must be 'analytic', a scalar, or one value per layer")
    oracle = None
    if not analytic:
        supplied = np.asarray(omega, dtype=np.float64)
        if supplied.ndim == 0:
            oracle = np.full(len(weights), supplied.item())
        elif supplied.shape == (len(weights),):
            oracle = supplied
        else:
            raise ValueError("oracle omega must be a scalar or a length-depth sequence")
    n = weights[0].shape[0]
    mu, C, U = np.zeros(n), np.eye(n), np.zeros((n, n, n))
    prediction, trace, used_omega, raw_next = [], [], [], []
    next_omega = float(initial_omega)
    for layer, W in enumerate(weights):
        requested = next_omega if analytic else float(oracle[layer])
        applied = max(0.0, requested) if nonnegative_omega and np.isfinite(requested) else requested
        if not np.isfinite(applied) or applied < 0.0:
            raise ValueError(
                f"layer {layer + 1}: requested omega={requested!r} cannot define a Gamma mixture; "
                "negative estimates require an explicit nonnegative_omega=True constraint"
            )
        m, S, T = k3.linear_step(np.asarray(W, dtype=np.float64), mu, C, U)
        result = state_to_h(m, S, T, applied, nodes=nodes, kmax=kmax, emax=emax)
        mu, C, U = result["mu"], result["C"], result["U"]
        next_omega = result["omega_next_raw"]
        trace.append(dict(result, m=m, S=S, T=T, omega=applied, omega_requested=requested))
        prediction.append(mu)
        used_omega.append(applied)
        raw_next.append(next_omega)
    return {
        "prediction": np.stack(prediction), "trace": trace,
        "omega": np.asarray(used_omega), "omega_next_raw": np.asarray(raw_next),
    }
