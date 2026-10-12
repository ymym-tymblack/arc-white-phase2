"""計画1: iik 補正源を入れた因子化版が、密版(iik, iii あり)と一致するかを n=128 で確認する。"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "research")
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import k3_dense as k3  # noqa: E402
from ablate_k3 import mc_means  # noqa: E402
from local_engine import build_mlp  # noqa: E402
from measure_layer_error import run_estimator  # noqa: E402

spec = importlib.util.spec_from_file_location("k3f", "candidates/k3_factored.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["k3f"] = mod
spec.loader.exec_module(mod)

n, depth = int(sys.argv[1]), int(sys.argv[2])
plain = dict(window=depth, tau_scale=1.0, use_omega=False, calib=0.0)
for seed in (0, 1):
    mlp = build_mlp(n, depth, seed)
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    ref = mc_means([w.astype(np.float32) for w in W], 4_000_000)
    mse = lambda x: ((x - ref) ** 2).mean(1)  # noqa: E731
    k3.GROUPS = {"rho", "tau1", "pairtau", "meantau", "iik", "iii"}
    dense_iik = k3.propagate(W, kmax=3, emax=2)
    k3.GROUPS = {"rho", "tau1", "pairtau", "meantau"}
    dense_min = k3.propagate(W, kmax=3, emax=2)
    f0, fl0 = run_estimator(mod.Estimator(**plain, window_iik=0), mlp)
    f1, fl1 = run_estimator(mod.Estimator(**plain, window_iik=depth), mlp)
    print(f"seed {seed}: max|因子化(iikなし)-密(min)|={np.abs(f0-dense_min).max():.1e}  "
          f"max|因子化(iik)-密(iik)|={np.abs(f1-dense_iik).max():.1e}")
    for name, x in (("因子化 iikなし", f0), ("因子化 iik", f1), ("密 iik", dense_iik)):
        print(f"  {name:14s} 最終層 {mse(x)[-1]:.2e}  全層 {mse(x).mean():.2e}")
