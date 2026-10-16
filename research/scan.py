"""設定の一覧を mini の分割上で評価する汎用スクリプト。

使い方: uv run python research/scan.py <split> '<JSON: 設定の配列>'
例:     uv run python research/scan.py valid '[{"window_iik":0},{"window_iik":4}]'
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
    ev = Evaluator(idxs)
    for kw in configs:
        t = time.time()
        r = ev.score(path, **kw)
        print(f"{json.dumps(kw, ensure_ascii=False)}: 調整後={r['adj']:.3e} 生MSE={r['mse']:.3e} "
              f"C/B={r['util']:.3f} resid最大={r['resid']:.2f}s ({time.time()-t:.0f}s)", flush=True)
