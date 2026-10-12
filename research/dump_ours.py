"""私の因子化版の層ごとの予測と内部量を mini の MLP について書き出す(参照実装・GPU MC との比較用)。

  uv run python research/dump_ours.py <設定名> '<JSON: Estimator の引数>' <MLP番号...>
出力: /kaggle/ext/ours/<設定名>_XXX.npz(pred と内部量 m, sigma, tdiag, mu, var_post)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
OUT = Path("/kaggle/ext/ours")

if __name__ == "__main__":
    from measure_layer_error import run_estimator
    from whestbench import MLP

    name, kw, idxs = sys.argv[1], json.loads(sys.argv[2]), [int(x) for x in sys.argv[3:]]
    spec = importlib.util.spec_from_file_location("cand", ROOT / "candidates/k3_factored.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand"] = mod
    spec.loader.exec_module(mod)
    OUT.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
        w = d["weights"]
        mlp = MLP(width=w.shape[1], depth=w.shape[0], weights=[x for x in w], seed=i)
        est = mod.Estimator(collect=True, **kw)
        pred, fl = run_estimator(est, mlp)
        feats = {k: np.stack([np.asarray(f[k], dtype=np.float64) for f in est.features])
                 for k in est.features[0]}
        np.savez(OUT / f"{name}_{i:03d}.npz", pred=pred, **feats)
        per = ((pred - d["all_layer_means"]) ** 2).mean(1)
        print(f"{name} MLP {i}: 最終層 {per[-1]:.3e} 全層 {per.mean():.3e} C/B {fl / 2**41:.3f}", flush=True)
