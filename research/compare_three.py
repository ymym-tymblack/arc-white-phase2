"""MC 真値に対する、私の版と参照実装の内部量の相対誤差を層ごとに並べる。

  uv run python research/compare_three.py <私の設定名> <MLP番号>
"""
import sys

import numpy as np

if __name__ == "__main__":
    name, i = sys.argv[1], int(sys.argv[2])
    ours = np.load(f"/kaggle/ext/ours/{name}_{i:03d}.npz")
    mc = np.load(f"/kaggle/ext/mc/mc_{i:03d}.npz")
    ref = np.load(f"/kaggle/ext/ref/int_{i:03d}.npz")
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    n = ours["m"].shape[1]
    off = ~np.eye(n, dtype=bool)
    print("相対誤差(MC 真値に対して)   私の版 / 参照実装")
    print(f"{'層':>2} | {'var':>17} | {'D3':>17} | {'D21 対角外':>17} | {'D21 対角(=D3)':>17} | {'ω 平均 真/私/参照':>22}")
    for l in range(2, 7):
        vt, d3t, d21t, k4t = mc[f"var_{l}"], mc[f"D3_{l}"], mc[f"D21_{l}"].astype(float), mc[f"K4_{l}"]
        vo, d3o, d21o = ours["sigma"][l - 1] ** 2, ours["D3"][l - 1], ours["D21"][l - 1]
        vr, d3r, d21r = ref[f"var_{l}"], ref[f"D3_{l}"], ref[f"D21_{l}"].astype(float)
        om_t = (k4t / vt**2).mean()
        om_r = (ref[f"K4_{l}"] / vr**2).mean() if f"K4_{l}" in ref.files else float("nan")
        print(f"{l:>2} | {rel(vo, vt):7.1e} / {rel(vr, vt):7.1e} | {rel(d3o, d3t):7.1e} / {rel(d3r, d3t):7.1e} | "
              f"{rel(d21o[off], d21t[off]):7.1e} / {rel(d21r[off], d21t[off]):7.1e} | "
              f"{rel(np.diag(d21o), d3t):7.1e} / {rel(np.diag(d21r), d3t):7.1e} | "
              f"{om_t:.4f} / {om_r:.4f}(参照)")
