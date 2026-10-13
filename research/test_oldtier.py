"""古い家族の共有基底化(age_old, rank_old)の精度と FLOP を、基底化なしと比べる。

  uv run python research/test_oldtier.py <MLP番号> '<JSON: 設定の配列>'
"""
import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, "/kaggle/arc-white-phase2/scripts")
sys.path.insert(0, "/kaggle/arc-white-phase2")
from measure_layer_error import run_estimator  # noqa: E402
from whestbench import MLP  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", "/kaggle/arc-white-phase2/candidates/k3_factored.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["cand"] = mod
spec.loader.exec_module(mod)
i = int(sys.argv[1])
d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
w, t = d["weights"], d["all_layer_means"]
mlp = MLP(width=w.shape[1], depth=w.shape[0], weights=[x for x in w], seed=i)
base, _ = run_estimator(mod.Estimator(), mlp)
for kw in json.loads(sys.argv[2]):
    p, fl = run_estimator(mod.Estimator(**kw), mlp)
    e = ((p - t) ** 2).mean(1)
    print(f"MLP{i} {json.dumps(kw)}: 最終層 {e[-1]:.3e} C/B {fl / 2**41:.3f} 調整後 {e[-1] * max(0.1, fl / 2**41):.3e} "
          f"基底化なしとの差 rms {np.sqrt(((p[-1] - base[-1]) ** 2).mean()):.2e}", flush=True)
