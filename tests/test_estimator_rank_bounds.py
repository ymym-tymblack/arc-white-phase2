"""Requested low ranks above width must not break the submission estimator."""

import numpy as np
import flopscope as flops
from whestbench import SetupContext

from estimator import Estimator
from local_engine import build_mlp


def test_requested_ranks_above_width_preserve_bounded_estimator():
    width, depth, budget = 4, 2, 2**41
    mlp = build_mlp(width, depth, seed=0)

    def predict(rank, old_rank):
        estimator = Estimator(
            directional_k4=True,
            age_old=None,
            iik_rank=rank,
            k3leg_rank=rank,
            rank_old=old_rank,
            family_prune_rel=0.0,
            rank_old_energy=1.0,
        )
        estimator.setup(
            SetupContext(
                width=width,
                depth=depth,
                flop_budget=budget,
                api_version="2.0",
                submission_dir=".",
                seed=0,
            )
        )
        with flops.BudgetContext(flop_budget=budget, quiet=True):
            return np.asarray(estimator.predict(mlp, budget), dtype=np.float64)

    oversized = predict(16, 256)
    bounded = predict(width, width)
    np.testing.assert_allclose(oversized, bounded, rtol=1e-6, atol=1e-7)
    weights = np.asarray(mlp.weights[0], dtype=np.float64)
    exact_first_mean = np.sqrt(np.sum(weights * weights, axis=0) / (2 * np.pi))
    np.testing.assert_allclose(oversized[0], exact_first_mean, rtol=3e-6, atol=2e-6)
