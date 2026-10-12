"""因子化版が密版の最小構成と一致するか、n=128 で検証。さらに MC との誤差も出す。"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import k3_dense as k3
from measure_layer_error import run_estimator
from local_engine import build_mlp
from ablate_k3 import mc_means

spec = importlib.util.spec_from_file_location("k3f", "research/archive/k3_factored_v1.py")
mod = importlib.util.module_from_spec(spec); sys.modules["k3f"] = mod; spec.loader.exec_module(mod)

n, depth = int(sys.argv[1]), int(sys.argv[2])
for seed in (0, 1):
    mlp = build_mlp(n, depth, seed)
    W = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    ref = mc_means(W, 4_000_000)
    k3.GROUPS = {"rho", "tau1", "pairtau", "meantau"}
    dense = k3.propagate([w.astype(np.float64) for w in W], kmax=3, emax=2, use_tau=True)
    fac, fl = run_estimator(mod.Estimator(3), mlp)
    k3.GROUPS = {"rho","tau1","tau1e","txxy","txxx","pairtau","meantau","iik","iii"}
    full = k3.propagate([w.astype(np.float64) for w in W], kmax=6, emax=2, use_tau=True)
    k2 = k3.propagate([w.astype(np.float64) for w in W], kmax=6, use_tau=False)
    mse = lambda x: ((x - ref) ** 2).mean(1)
    print(f"seed {seed}: max|因子化-密(min)| = {np.abs(fac-dense).max():.2e}, flops={fl/2**41:.4%}B")
    print("  層別MSE 因子化 :", " ".join(f"{v:.1e}" for v in mse(fac)))
    print("  層別MSE 密(min):", " ".join(f"{v:.1e}" for v in mse(dense)))
    print("  層別MSE 密(full):", " ".join(f"{v:.1e}" for v in mse(full)))
    print("  層別MSE K2     :", " ".join(f"{v:.1e}" for v in mse(k2)))
