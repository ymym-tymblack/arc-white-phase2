"""参照実装の ω(ニューロンごとの超過尖度)を私の版に差し込み、深い層の誤差への効果を測る。

  uv run python research/oracle_refomega.py <MLP番号> '<JSON: Estimator の引数>'
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from measure_layer_error import run_estimator
    from whestbench import MLP

    i, kw = int(sys.argv[1]), json.loads(sys.argv[2])
    spec = importlib.util.spec_from_file_location("cand", ROOT / "research/archive/k3_factored_v1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand"] = mod
    spec.loader.exec_module(mod)
    d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
    ref = np.load(f"/kaggle/ext/ref/int_{i:03d}.npz")
    w = d["weights"]
    mlp = MLP(width=w.shape[1], depth=w.shape[0], weights=[x for x in w], seed=i)
    truth = d["all_layer_means"]
    orc_vec = {l - 1: {"omega": ref[f"K4_{l}"] / ref[f"var_{l}"] ** 2} for l in range(2, 17)}
    orc_mean = {l - 1: {"omega": np.full(w.shape[1], (ref[f"K4_{l}"] / ref[f"var_{l}"] ** 2).mean())}
                for l in range(2, 17)}
    for name, orc in (("表 G/n", None), ("参照の層平均 ω", orc_mean), ("参照のニューロン別 ω", orc_vec)):
        pred, _ = run_estimator(mod.Estimator(oracle=orc, **kw), mlp)
        print(f"{name:14s}", " ".join(f"{x:.1e}" for x in ((pred - truth) ** 2).mean(1)), flush=True)
