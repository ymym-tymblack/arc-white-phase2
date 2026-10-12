"""GPU MC の真値(D3, D21, ω)を私の推定器に層ごとに差し込み、平均の誤差がどこまで下がるかを測る。

  uv run python research/oracle_mc.py <MLP番号> '<JSON: Estimator の引数>'
差し込むのは MC がある層(2〜L_mc)だけ。誤差は mini の正解(N=1e9)との層ごとの MSE。
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
    spec = importlib.util.spec_from_file_location("cand", ROOT / "candidates/k3_factored.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand"] = mod
    spec.loader.exec_module(mod)
    d = np.load(f"/kaggle/ext/dumps/mlp_{i:03d}.npz")
    mc = np.load(f"/kaggle/ext/mc/mc_{i:03d}.npz")
    Lmc = max(int(k.split("_")[1]) for k in mc.files if k.startswith("m_"))
    w = d["weights"]
    mlp = MLP(width=w.shape[1], depth=w.shape[0], weights=[x for x in w], seed=i)
    truth = d["all_layer_means"]
    ref = np.load(f"/kaggle/ext/ref/kprop_k3_{i:03d}.npz")["pred"]

    def orc(keys):
        out = {}
        for l in range(2, Lmc + 1):
            e = {}
            if "D3" in keys:
                e["D3"] = mc[f"D3_{l}"]
            if "D21" in keys:
                e["D21"] = mc[f"D21_{l}"]
            if "omega" in keys:
                e["omega"] = mc[f"K4_{l}"] / mc[f"var_{l}"] ** 2
            out[l - 1] = e
        return out

    rows = {"参照実装 K3": ((ref - truth) ** 2).mean(1)}
    for name, keys in (("差し込みなし", ()), ("D3", ("D3",)), ("D21", ("D21",)),
                       ("D3+D21", ("D3", "D21")), ("ω", ("omega",)),
                       ("D3+D21+ω", ("D3", "D21", "omega"))):
        pred, _ = run_estimator(mod.Estimator(oracle=orc(keys) if keys else None, **kw), mlp)
        rows[name] = ((pred - truth) ** 2).mean(1)
    print(f"MLP {i} 設定 {kw}  層ごとの MSE(層1〜{Lmc + 2})")
    for name, v in rows.items():
        print(f"  {name:12s}", " ".join(f"{x:.1e}" for x in v[: Lmc + 2]))
