"""設定の一覧を mini の分割上で評価する(縮小係数は設定ごとに解析的に最適化)。

  uv run python research/scan_fit.py <split> '<JSON: 設定の配列>' [推定器のパス]
"""
import json
import sys
import time

sys.path.insert(0, "research")
from eval_mini import ROOT, SPLITS, Evaluator  # noqa: E402

if __name__ == "__main__":
    split = sys.argv[1]
    idxs = SPLITS[split] if split in SPLITS else [int(x) for x in split.split(",")]
    configs = json.loads(sys.argv[2])
    path = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "estimator.py")
    ev = Evaluator(idxs, workers=8, threads=4)
    for kw in configs:
        t = time.time()
        r = ev.score_fit_calib(path, **kw)
        print(f"{json.dumps(kw, ensure_ascii=False)}: 調整後={r['adj']:.3e} 生MSE={r['mse']:.3e} "
              f"calib={r['calib']:.2e} C/B={r['util']:.3f} resid最大={r['resid']:.2f}s "
              f"({time.time()-t:.0f}s)", flush=True)
