"""計画3 改: 焼きなまし法(SA)→ ビームサーチで推定器の設定を探索する。

前回(tune_k3_v2.py)との違い:
  - 探索に使う MLP を 12 個 → 40 個(mini 0〜39 番)に増やし、目的関数の雑音を減らす。
  - 縮小係数 calib は探索せず、設定ごとに解析的に最適値を求める(Evaluator.score_fit_calib)。
  - 連続変数は格子に丸め、SA とビームサーチで評価結果のキャッシュを共有する。
最終確認用(40〜99 番)はここでは使わない。

  uv run python research/search_k3.py [SA の反復数] [ビーム幅]
"""
import json
import math
import random
import sys
import time

sys.path.insert(0, "research")
from eval_mini import ROOT, Evaluator  # noqa: E402

PATH = str(ROOT / "research/archive/k3_factored_v1.py")
OUT = ROOT / "results/search_k3.json"
SEARCH = list(range(0, 40))

# 変数ごとの候補値(格子)
GRID = {
    "window": [5, 6, 7, 8, 9, 10],
    "window_iik": [0, 1, 2],
    "kmax": [2, 3, 4],
    # 1 回目の探索で下限 0.94 の近くまで改善が続いたため、下限を 0.80 に広げた
    "tau_scale": [round(0.80 + 0.02 * i, 2) for i in range(21)],  # 0.80〜1.20
    "omega_scale": [round(0.5 + 0.1 * i, 1) for i in range(10)],  # 0.5〜1.4
}
START = dict(window=7, window_iik=0, kmax=2, tau_scale=1.02, omega_scale=0.9)
RESID_LIMIT = 0.38  # residual 時間の上限 0.4 s に余裕を残す(並列評価中の値)


class Objective:
    def __init__(self):
        self.ev = Evaluator(SEARCH, workers=8, threads=4)  # 実測で 4×8 より 1.4 倍速い
        self.cache: dict[str, dict] = {}

    def __call__(self, x: dict) -> float:
        key = json.dumps(x, sort_keys=True)
        if key not in self.cache:
            t = time.time()
            r = self.ev.score_fit_calib(PATH, **x)
            half = len(SEARCH) // 2
            self.cache[key] = dict(
                x=x, adj=r["adj"], mse=r["mse"], calib=r["calib"], util=r["util"],
                resid=r["resid"],
                adj_a=sum(r["per_adj"][:half]) / half,
                adj_b=sum(r["per_adj"][half:]) / (len(SEARCH) - half),
            )
            e = self.cache[key]
            print(f"  {x} → 調整後 {e['adj']:.4e} (前半 {e['adj_a']:.3e} 後半 {e['adj_b']:.3e}) "
                  f"calib {e['calib']:.2e} C/B {e['util']:.3f} resid {e['resid']:.2f}s "
                  f"[{time.time()-t:.0f}s, 評価 {len(self.cache)} 回目]", flush=True)
        e = self.cache[key]
        return e["adj"] if e["resid"] < RESID_LIMIT else 1.0


def neighbors(x: dict):
    """各変数を格子上で 1 つ隣に動かした設定をすべて返す。"""
    out = []
    for name, vals in GRID.items():
        i = vals.index(x[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                y = dict(x)
                y[name] = vals[j]
                out.append(y)
    return out


def anneal(f: Objective, x0: dict, steps: int, t0: float = 0.05, t1: float = 0.005, seed: int = 0):
    """焼きなまし法。比較は log(スコア) の差で行う(Δ=0.05 は約 5% の悪化)。

    悪化する移動も確率 exp(-Δ/T) で受け入れ、局所解から抜け出す。T は t0 → t1 に幾何的に下げる。
    """
    rnd = random.Random(seed)
    x, fx = x0, f(x0)
    best, fbest = x, fx
    for k in range(steps):
        T = t0 * (t1 / t0) ** (k / max(1, steps - 1))
        name = rnd.choice(list(GRID))
        vals = GRID[name]
        i = vals.index(x[name])
        j = min(len(vals) - 1, max(0, i + rnd.choice((-2, -1, 1, 2))))
        if j == i:
            continue
        y = dict(x)
        y[name] = vals[j]
        fy = f(y)
        d = math.log(fy) - math.log(fx)
        if d < 0 or rnd.random() < math.exp(-d / T):
            x, fx = y, fy
        if fx < fbest:
            best, fbest = x, fx
        print(f"SA {k+1}/{steps}: T={T:.4f} 現在 {fx:.4e} 最良 {fbest:.4e}", flush=True)
    return best, fbest


def beam(f: Objective, starts: list[dict], width: int, iters: int):
    """ビームサーチ。上位 width 個の設定を保持し、全員の近傍を評価して上位 width 個に入れ替える。

    貪欲法(1 個だけ保持)と違い、2 番手以下も残すので、一時的に悪い方向を経由する改善も拾える。
    """
    beam_ = sorted(starts, key=f)[:width]
    for it in range(iters):
        pool = {json.dumps(x, sort_keys=True): x for x in beam_}
        for x in beam_:
            for y in neighbors(x):
                pool[json.dumps(y, sort_keys=True)] = y
        new = sorted(pool.values(), key=f)[:width]
        print(f"ビーム {it+1}/{iters}: " + " | ".join(f"{f(x):.4e}" for x in new), flush=True)
        if new == beam_:
            break
        beam_ = new
    return beam_


def main():
    sa_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    resume = len(sys.argv) > 3 and sys.argv[3] == "resume"
    f = Objective()
    if resume:
        # 前回の評価結果を読み込み、SA を省いてビームサーチだけを続ける
        for e in json.loads(OUT.read_text())["log"]:
            f.cache[json.dumps(e["x"], sort_keys=True)] = e
        f(START)
        sa_best = min(f.cache.values(), key=lambda e: e["adj"])["x"]
        iters = 6
    else:
        print("== 焼きなまし法 ==", flush=True)
        sa_best, _ = anneal(f, START, sa_steps)
        iters = 3
    print("== ビームサーチ ==", flush=True)
    top = sorted(f.cache.values(), key=lambda e: e["adj"] if e["resid"] < RESID_LIMIT else 1.0)
    final = beam(f, [sa_best] + [e["x"] for e in top[:width]], width, iters=iters)
    choice = f.cache[json.dumps(final[0], sort_keys=True)]
    start = f.cache[json.dumps(START, sort_keys=True)]
    print(f"開始点: {start}", flush=True)
    print(f"採用  : {choice}", flush=True)
    OUT.write_text(json.dumps(dict(start=start, choice=choice, log=list(f.cache.values())),
                              indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
