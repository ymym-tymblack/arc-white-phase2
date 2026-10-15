"""3次のトレースベクトルと4次の放射スカラーだけを運ぶ非MC推定器。

研究用。κ3[h_i,h_j,h_k]≈v_i D_jk+v_j D_ik+v_k D_ij。
ReLU後の標準化した3次トレースを射影すると
v_i=Σ_j κ3[h_j,h_j,h_i]/Var(h_j)/(n+2)、D=diag(Var(h))。
線形層は v←W.T v, D←W.T D W で、この表現内では厳密。
4次はK22の総和から放射スカラーを更新する。個々のMLPの向きに依存する
トレースなし成分は保持しない。経験的な尖度表・当てはめ・入力サンプリングなし。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator

from candidates.k3_factored import _hermite_neg_alpha, _pair_terms


class Estimator(BaseEstimator):
    def __init__(self, metric="elliptical", collect=False, precision="float64"):
        if metric not in ("elliptical", "radial"):
            raise ValueError(metric)
        if precision not in ("float32", "float64"):
            raise ValueError(precision)
        self.dtype = fnp.float64 if precision == "float64" else fnp.float32
        self.metric = metric
        self.collect = collect
        self.features = []

    def predict(self, mlp, budget):
        n, kmax = mlp.width, 3
        dtype = self.dtype
        mu = fnp.zeros(n, dtype=dtype)
        C = flops.as_symmetric(fnp.eye(n, dtype=dtype), symmetry=(0, 1))
        v = fnp.zeros(n, dtype=dtype)
        diagonal_metric = fnp.ones(n, dtype=dtype)
        omega = 0.0
        rows = []
        for layer, w in enumerate(mlp.weights):
            m = w.T @ mu
            S = fnp.einsum("ij,ia,jb->ab", C, w, w)
            D = fnp.matmul(w.T, diagonal_metric[:, None] * w)
            vp = w.T @ v
            variance = fnp.maximum(fnp.diag(S), 1e-12)
            sigma = fnp.sqrt(variance)
            rho = S / fnp.outer(sigma, sigma)
            d = fnp.diag(D)
            D3 = 3.0 * vp * d
            D21 = fnp.outer(d, vp) + 2.0 * vp[:, None] * D
            tau_diag = D3 / (sigma * variance)
            tau21 = D21 / fnp.outer(variance, sigma)
            alpha = m / sigma
            phi = flops.stats.norm.pdf(alpha).astype(dtype)
            Phi = flops.stats.norm.cdf(alpha).astype(dtype)
            he = _hermite_neg_alpha(alpha, 7)
            I = [Phi, m * Phi + sigma * phi]
            for p in range(2, 5):
                I.append(m * I[-1] + float(p - 1) * variance * I[-2])
            coefficients = {}
            for p in range(1, 5):
                cp = []
                for k in range(8):
                    if k <= p:
                        value = (float(math.factorial(p) / math.factorial(p-k))
                                 * sigma**k * I[p-k])
                    else:
                        value = float(math.factorial(p)) * sigma**p * phi * he[k-p-1]
                    cp.append(value)
                coefficients[p] = cp
            raw = {}
            for p, cp in coefficients.items():
                raw[p] = cp[0] + cp[3] * tau_diag / 6.0 + cp[6] * tau_diag**2 / 72.0 + cp[4] * omega / 24.0
            mu = raw[1]
            rows.append(mu)
            if layer == mlp.depth - 1:
                break
            postvar = fnp.maximum(raw[2] - mu*mu, 0.0)
            R = [fnp.ones_like(rho), rho]
            rp = rho
            for k in range(2, kmax+1):
                rp = rp * rho
                R.append(rp / float(math.factorial(k)))
            om = omega * fnp.ones_like(sigma)
            if self.metric == "elliptical":
                k22 = (omega / 3.0) * (1.0 + 2.0*rho*rho)
                k31 = omega * rho
            else:
                k22 = (omega / 3.0) * fnp.ones_like(rho)
                k31 = None

            def pair(p, q):
                a, b = coefficients[p], coefficients[q]
                value = _pair_terms(a, b, R, kmax, tau21, tau21.T, tau_diag, (om, k22))
                if k31 is not None:
                    for e in range(kmax+1):
                        value = value + (
                            fnp.outer(a[3+e], b[1+e]) * k31
                            + fnp.outer(a[1+e], b[3+e]) * k31.T
                        ) * (R[e] / 6.0)
                return value

            covariance = pair(1, 1)
            fnp.fill_diagonal(covariance, postvar)
            cov21 = pair(2, 1)
            fnp.fill_diagonal(cov21, raw[3] - raw[2] * mu)
            U21 = cov21 - 2.0 * mu[:, None] * covariance
            k3diag = raw[3] - 3.0*mu*raw[2] + 2.0*mu**3
            fnp.fill_diagonal(U21, k3diag)
            inverse = fnp.where(postvar > 1e-12, 1.0/fnp.maximum(postvar, 1e-12), 0.0)
            v = fnp.matmul(U21.T, inverse) / float(n+2)
            cov22 = pair(2, 2)
            fnp.fill_diagonal(cov22, raw[4] - raw[2]**2)
            K22 = (cov22 - 2.0*mu[:, None]*cov21.T - 2.0*mu[None, :]*cov21
                   + 4.0*fnp.outer(mu, mu)*covariance - 2.0*covariance*covariance)
            k4diag = raw[4] - 4.0*mu*raw[3] + 6.0*mu**2*raw[2] - 3.0*mu**4 - 3.0*postvar**2
            fnp.fill_diagonal(K22, k4diag)
            denominator = fnp.sum(postvar)**2 + 2.0*fnp.sum(covariance*covariance)
            omega = 3.0*fnp.sum(K22) / fnp.maximum(denominator, 1e-30)
            C = flops.as_symmetric(covariance, symmetry=(0, 1))
            diagonal_metric = postvar
            if self.collect:
                self.features.append(dict(omega=omega, v=v, variance=postvar))
        return fnp.stack(rows, axis=0)
