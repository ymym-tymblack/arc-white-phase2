"""共分散伝播(ReLU 後の共分散を Mehler 展開で厳密近似する版)。

examples/03 は ReLU 後の非対角共分散を gain[i]*gain[j]*cov_pre[i,j] で近似している。
これは下の展開の k=1 項だけを取った式に等しい。ここでは k=1..K 項まで足す。

  z = (z_i, z_j) が同時ガウス、周辺が N(mu_i, s_i^2)、相関が rho のとき

    Cov(ReLU(z_i), ReLU(z_j)) = sum_{k>=1} c_k(i) c_k(j) rho^k / k!

    c_1(i) = s_i * Phi(a_i)                       (a_i = mu_i / s_i)
    c_k(i) = s_i * He_{k-2}(-a_i) * phi(a_i)      (k >= 2)

  He_m は確率論者のエルミート多項式 (He_0=1, He_1=x, He_{m+1}=x He_m - m He_{m-1})。
  k=1 項が examples/03 の gain 近似、k>=2 項が追加分。

事前活性化が同時ガウスという仮定(各層のガウス近似)は examples/03 と同じ。
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e30


class Estimator(BaseEstimator):
    # 実測: K=1 は examples/03 と一致、K=2 で誤差 約7% 減、K>=3 は 3 桁で不変。
    n_terms = 2  # 展開の項数 K

    def __init__(self, n_terms: int | None = None) -> None:
        self._setup_rng = None
        if n_terms is not None:
            self.n_terms = n_terms

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)

            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # --- Mehler 展開で ReLU 後の共分散を作る ---
            rho = cov_pre / fnp.outer(sigma_pre, sigma_pre)
            x = -alpha
            he_prev = fnp.ones_like(x)  # He_0
            he_cur = x  # He_1
            rho_pow = rho  # rho^k
            # k = 1
            c = sigma_pre * Phi_alpha
            cov_new = fnp.multiply(fnp.outer(c, c), rho_pow)
            for k in range(2, self.n_terms + 1):
                m = k - 2
                if m == 0:
                    he = he_prev
                elif m == 1:
                    he = he_cur
                else:
                    he_next = x * he_cur - float(m - 1) * he_prev
                    he_prev, he_cur = he_cur, he_next
                    he = he_cur
                c = (sigma_pre * phi_alpha * he) / math.sqrt(math.factorial(k))
                rho_pow = fnp.multiply(rho_pow, rho)
                cov_new = cov_new + fnp.multiply(fnp.outer(c, c), rho_pow)
            cov = cov_new

            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)
