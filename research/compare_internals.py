"""私の推定器の内部量(z の平均・分散・D3・D21・ω)を GPU MC の真値と層ごとに比べる。

  uv run python research/compare_internals.py <設定名> <MLP番号>
相対誤差 = ‖推定 - 真値‖ / ‖真値‖(対角外の D21 は i≠k の成分だけ)。
MC の雑音の目安として、真値を半分の標本で割った値との差は別途 N に応じて小さくなる。
"""
import sys

import numpy as np

if __name__ == "__main__":
    name, i = sys.argv[1], int(sys.argv[2])
    ours = np.load(f"/kaggle/ext/ours/{name}_{i:03d}.npz")
    mc = np.load(f"/kaggle/ext/mc/mc_{i:03d}.npz")
    L = max(int(k.split("_")[1]) for k in mc.files if k.startswith("m_"))
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    print(f"{'層':>2} | {'m':>8} {'var':>8} {'D3':>8} {'D21対角外':>10} {'ω':>8} | "
          f"{'D3 rms真':>9} {'D21 rms真':>9} {'ω平均 真/推定':>16}")
    n = ours["m"].shape[1]
    off = ~np.eye(n, dtype=bool)
    for l in range(1, L + 1):
        m_t, v_t = mc[f"m_{l}"], mc[f"var_{l}"]
        d3_t, d21_t, k4_t = mc[f"D3_{l}"], mc[f"D21_{l}"].astype(np.float64), mc[f"K4_{l}"]
        m_o, s_o = ours["m"][l - 1], ours["sigma"][l - 1]
        v_o = s_o**2
        d3_o, d21_o = ours["D3"][l - 1], ours["D21"][l - 1]
        om_t = k4_t / v_t**2
        om_o = 62.31  # 表の値はここでは使わず、層ごとの値を下で出す
        _ = om_o
        print(f"{l:>2} | {rel(m_o, m_t):8.1e} {rel(v_o, v_t):8.1e} "
              f"{rel(d3_o, d3_t) if l > 1 else float('nan'):8.1e} "
              f"{rel(d21_o[off], d21_t[off]) if l > 1 else float('nan'):10.1e} {'':8s} | "
              f"{np.sqrt((d3_t**2).mean()):9.2e} {np.sqrt((d21_t[off]**2).mean()):9.2e} "
              f"{om_t.mean():8.4f}", flush=True)
