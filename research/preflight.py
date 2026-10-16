"""Single-thread submission safety: shapes, finite values, FLOPs, time and RSS."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
from whestbench import SetupContext  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from local_engine import build_mlp  # noqa: E402

B = 2**41
CASES = ((1024, 16), (1024, 16), (1024, 24), (1024, 32), (1024, 48), (256, 8), (256, 32))


def run_case(model, width, depth, repeat):
    context = None
    started = time.monotonic()
    row = dict(width=width, depth=depth, repeat=repeat, passed=False)
    try:
        mlp = build_mlp(width, depth, seed=1)
        estimator = model.Estimator()
        estimator.setup(
            SetupContext(
                width=width,
                depth=depth,
                flop_budget=B,
                api_version="2.0",
                submission_dir=str(ROOT),
                seed=0,
            )
        )
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=B, wall_time_limit_s=120.0, quiet=True) as context:
            prediction = estimator.predict(mlp, B)
        elapsed = time.monotonic() - started
        prediction = np.asarray(prediction)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        checks = dict(
            shape=prediction.shape == (depth, width),
            finite=bool(np.isfinite(prediction).all()),
            flops=context.flops_used <= B,
            wall=elapsed < 120.0,
            residual=context.residual_wall_time_s < 0.4,
            memory=rss < 8.0,
        )
        row.update(
            shape=list(prediction.shape),
            finite=checks["finite"],
            flops=int(context.flops_used),
            C_over_B=context.flops_used / B,
            wall_seconds=elapsed,
            residual_seconds=float(context.residual_wall_time_s),
            max_rss_gib=rss,
            checks=checks,
            passed=all(checks.values()),
        )
    except Exception as error:
        row.update(
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.monotonic() - started,
            flops=int(context.flops_used) if context is not None else None,
            max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        )
    if "error" in row:
        print(f"{width}x{depth}: NG {row['error']}", flush=True)
    else:
        print(
            f"{width}x{depth}: shape={tuple(row['shape'])} finite={row['finite']} "
            f"C/B={row['C_over_B']:.3f} time={row['wall_seconds']:.1f}s "
            f"residual={row['residual_seconds']:.3f}s maxRSS={row['max_rss_gib']:.2f}GiB "
            f"{'OK' if row['passed'] else 'NG'}",
            flush=True,
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path, default=ROOT / "estimator.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    spec = importlib.util.spec_from_file_location("preflight_candidate", path)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    report = dict(
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        threads=1,
        limits=dict(flops=B, residual_seconds=0.4, wall_seconds=120.0, rss_gib=8.0),
        memory_note="RSS checked here; official subprocess separately enforces virtual memory.",
        cases=[],
        passed=False,
    )
    for repeat, (width, depth) in enumerate(CASES):
        report["cases"].append(run_case(model, width, depth, repeat))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["passed"] = all(row["passed"] for row in report["cases"])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("全項目 OK" if report["passed"] else "NG あり", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
