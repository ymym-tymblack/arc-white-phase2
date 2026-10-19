"""固定学習値・Strassen再構成・実MLPの予測とresidualを比較する研究用smoke。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    model = importlib.util.module_from_spec(spec)
    sys.modules[name] = model
    spec.loader.exec_module(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    from whestbench import SetupContext

    from local_engine import build_mlp
    from research.resource_search.search import source_audit

    audits = [source_audit(path) for path in (args.reference, args.candidate)]
    if audits[0]["fixed_parameter_sha256"] != audits[1]["fixed_parameter_sha256"]:
        raise ValueError("Learned parameters changed")
    old, new = load(args.reference, "residual_reference"), load(args.candidate, "residual_candidate")
    sw = new._Strassen()
    sw.min_flops = 0.0
    rng = np.random.default_rng(9)
    exact = []
    for m, k, p, levels in ((64, 64, 96, 3), (64, 128, 32, 2), (64, 48, 80, 2)):
        left = rng.integers(-3, 4, size=(m, k)).astype(np.float32)
        right_t = rng.integers(-3, 4, size=(p, k)).astype(np.float32)
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            output = fnp.full((m, p + 8), 123.0, dtype=fnp.float32)
            sw.matmul(fnp.asarray(left), fnp.asarray(right_t).T, levels, out=output[:, 3:3 + p])
        actual = np.asarray(output)
        expected = left.astype(np.int64) @ right_t.astype(np.int64).T
        if not np.array_equal(actual[:, 3:3 + p], expected):
            raise ValueError(f"Strassen result differs: {(m, k, p, levels)}")
        if not (np.all(actual[:, :3] == 123.0) and np.all(actual[:, 3 + p:] == 123.0)):
            raise ValueError("Strassen wrote outside the strided output")
        exact.append(dict(shape=[m, k, p], levels=levels, exact=True))
    budget = 2**41
    mlp = build_mlp(1024, 16, seed=1)
    predictions, runs = [], []
    for model in (old, new):
        estimator = model.Estimator()
        started = time.perf_counter()
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=budget,
                                      api_version="2.0", submission_dir=str(ROOT), seed=0))
        setup = time.perf_counter() - started
        with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120.0, quiet=True) as ctx:
            prediction = estimator.predict(mlp, budget)
        predictions.append(np.asarray(prediction))
        runs.append(dict(setup_seconds=setup, residual_seconds=ctx.residual_wall_time_s,
                         wall_seconds=ctx.wall_time_s, flops=ctx.flops_used,
                         ops=len(ctx._op_log)))
        print(json.dumps(runs[-1]), flush=True)
    difference = float(np.max(np.abs(predictions[0] - predictions[1])))
    report = dict(source_audits=audits, integer_cases=exact, runs=runs,
                  max_abs_prediction_difference=difference,
                  bitwise_equal=bool(np.array_equal(*predictions)),
                  passed=difference <= 1e-5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source_audits", "integer_cases")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
