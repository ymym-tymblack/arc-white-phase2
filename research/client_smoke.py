"""公式同様のflopscope client/serverで開発先頭網を実行し、保存済み予測と照合する。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    from whestbench import SetupContext

    budget = 2**41
    with flops.BudgetContext(flop_budget=10**16, quiet=True):
        # prep_data.py が mlp_at と一致を確認した実データ。remote未実装の
        # SeedSequenceによるメタデータ生成は不要で、提出が読む３フィールドは同一。
        weights = np.load("/kaggle/ext/torch_data/dev64/w_0064.npy")
        mlp = SimpleNamespace(width=weights.shape[1], depth=weights.shape[0],
                              weights=[fnp.asarray(weight) for weight in weights])
        spec = importlib.util.spec_from_file_location("client_smoke_candidate", args.candidate)
        model = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = model
        spec.loader.exec_module(model)
        estimator = model.Estimator()
        started = time.perf_counter()
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=budget,
                                      api_version="2.0", submission_dir=str(ROOT), seed=0))
        setup = time.perf_counter() - started
    reference = np.load(args.reference)
    records = []
    for repeat in range(2):
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=budget, quiet=True) as ctx:
            prediction = estimator.predict(mlp, budget)
            array = np.asarray(prediction.tolist(), dtype=np.float64)
            del prediction
        difference = float(np.max(np.abs(array - reference)))
        record = dict(repeat=repeat, wall_seconds=time.perf_counter() - started,
                      residual_seconds=ctx.residual_wall_time_s, flops=ctx.flops_used,
                      max_abs_difference=difference, finite=bool(np.isfinite(array).all()))
        records.append(record)
        print(json.dumps(record), flush=True)
    passed = setup < 5 and all(record["finite"] and record["max_abs_difference"] <= 2e-5
                              and record["wall_seconds"] <= 120
                              and record["residual_seconds"] < 0.4 for record in records)
    report = dict(candidate=str(args.candidate),
                  source_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  reference=str(args.reference), setup_seconds=setup, records=records, passed=passed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("passed", passed, "setup", setup, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
