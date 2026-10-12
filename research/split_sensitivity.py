"""評価分割の選び方で結論が変わるかを調べる: 2 つの設定を mini 100 MLP 全部で MLP ごとに評価し、
ランダムな分割を多数作って「最終確認用での改善率」がどれだけ揺れるかを見る。"""
import json
import sys

import numpy as np

sys.path.insert(0, "research")
from eval_mini import ROOT, Evaluator  # noqa: E402

OLD = dict(window=7, window_iik=0, tau_scale=1.14, omega_scale=1.05, calib=2.25e-4, kmax=3)
NEW = {}

if __name__ == "__main__":
    ev = Evaluator(range(100))
    path = str(ROOT / "research/archive/k3_factored_v1.py")
    per = {}
    for name, kw in (("old", OLD), ("new", NEW)):
        r = ev.score(path, **kw)
        per[name] = [x["mse"] * max(0.1, x["flops"] / 2**41) for x in r["per"]]
    (ROOT / "results/per_mlp_old_new.json").write_text(json.dumps(per))
    a, b = np.array(per["old"]), np.array(per["new"])
    print(f"MLP ごとの調整後スコア: 旧 平均 {a.mean():.3e}(変動係数 {a.std()/a.mean():.2f}) "
          f"新 平均 {b.mean():.3e}")
    ratio = b / a
    print(f"MLP ごとの 新/旧 の比: 平均 {ratio.mean():.3f}、新が良い MLP {np.mean(ratio < 1):.0%}、"
          f"最小 {ratio.min():.3f} 最大 {ratio.max():.3f}")
    rng = np.random.default_rng(0)
    for size in (12, 28, 60):
        r = [b[i].mean() / a[i].mean() for i in (rng.permutation(100)[:size] for _ in range(5000))]
        lo, hi = np.percentile(r, [2.5, 97.5])
        print(f"ランダムな {size:2d} 個で測った 新/旧: 中央値 {np.median(r):.3f}、95% 区間 [{lo:.3f}, {hi:.3f}]")
