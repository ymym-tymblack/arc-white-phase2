"""計画4: 提出前の安全確認(形状・FLOP 上限・時間・メモリ)。

  uv run python research/preflight.py

確認すること:
  - 幅 1024 で深さ 16 / 24 / 32、幅 256 で深さ 8 / 32 の MLP で、予測の形が正しく有限で、
    FLOP が予算 2^41 以内か(超えるとその MLP の予測はゼロになる)。
  - 1 スレッドでの 1 MLP の実行時間(上限 120 s)と residual 時間(上限 0.4 s)。
  - プロセスの最大メモリ(上限 8 GB)。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import flopscope as flops  # noqa: E402
from whestbench import SetupContext  # noqa: E402

from local_engine import build_mlp  # noqa: E402

PATH = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "candidates/k3_factored.py")
B = 2**41


def main():
    spec = importlib.util.spec_from_file_location("cand", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand"] = mod
    spec.loader.exec_module(mod)
    ok = True
    for width, depth in ((1024, 16), (1024, 16), (1024, 24), (1024, 32), (1024, 48), (256, 8), (256, 32)):
        mlp = build_mlp(width, depth, seed=1)
        est = mod.Estimator()
        est.setup(SetupContext(width=width, depth=depth, flop_budget=B, api_version="1.0",
                               submission_dir=str(ROOT), seed=0))
        t = time.time()
        with flops.BudgetContext(flop_budget=B, wall_time_limit_s=120.0, quiet=True) as ctx:
            pred = est.predict(mlp, B)
        dt = time.time() - t
        pred = np.asarray(pred)
        good = (pred.shape == (depth, width) and np.isfinite(pred).all()
                and ctx.flops_used <= B and dt < 120 and ctx.residual_wall_time_s < 0.4)
        ok &= bool(good)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        print(f"{width}x{depth}: shape={pred.shape} finite={np.isfinite(pred).all()} "
              f"C/B={ctx.flops_used / B:.3f} time={dt:.1f}s residual={ctx.residual_wall_time_s:.3f}s "
              f"maxRSS={rss:.2f}GB {'OK' if good else 'NG'}", flush=True)
    print("全項目 OK" if ok else "NG あり")


if __name__ == "__main__":
    main()
