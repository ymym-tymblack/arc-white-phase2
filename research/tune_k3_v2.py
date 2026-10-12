"""計画3: 源の窓(種類別の年齢ゲート)と倍率を、貪欲法 → TPE の順に探索する。

目的関数 = mean(final_layer_mse × max(0.1, C/B))(公式スコアと同じ)を探索用 MLP(0〜11)で評価。
上位候補を検証用(12〜39)で比べて 1 つ選ぶ。最終確認用(40〜99)はここでは使わない。

  uv run --with optuna python research/tune_k3_v2.py [TPE の試行数]
"""
import json
import sys
import time

sys.path.insert(0, "research")
from eval_mini import ROOT, SPLITS, Evaluator  # noqa: E402

PATH = str(ROOT / "research/archive/k3_factored_v1.py")
OUT = ROOT / "results/tune_k3_v2.json"

# 探索空間: (名前, 下限, 上限, 貪欲法の刻み, 整数か)
SPACE = [
    ("window", 4, 10, 1, True),
    ("window_iik", 0, 3, 1, True),
    ("tau_scale", 0.95, 1.20, 0.03, False),
    ("omega_scale", 0.6, 1.5, 0.15, False),
    ("calib", 1.0e-4, 3.5e-4, 0.4e-4, False),
    ("kmax", 2, 4, 1, True),
]
START = dict(window=7, window_iik=1, tau_scale=1.07, omega_scale=1.05, calib=2.25e-4, kmax=3)


def main():
    import optuna  # spawn した子が main を再 import したときに BLAS 設定を壊さないよう、ここで読む

    ev = Evaluator(SPLITS["tune"])
    cache: dict[str, float] = {}
    log = []

    def score(kw):
        key = json.dumps(kw, sort_keys=True)
        if key not in cache:
            t = time.time()
            r = ev.score(PATH, **kw)
            # residual 時間の上限 0.4 s に余裕を残す(並列評価中の値なので実運用より大きめ)
            cache[key] = r["adj"] if r["resid"] < 0.38 else 1.0
            log.append(dict(kw=kw, adj=r["adj"], mse=r["mse"], util=r["util"], resid=r["resid"]))
            print(f"  {kw} → 調整後 {r['adj']:.3e} 生MSE {r['mse']:.3e} C/B {r['util']:.3f} "
                  f"resid {r['resid']:.2f}s ({time.time()-t:.0f}s)", flush=True)
        return cache[key]

    # --- 貪欲法(座標降下): 1 変数ずつ ±1 刻みを試し、良くなれば採用 ---
    best = dict(START)
    best_v = score(best)
    for sweep in range(2):
        improved = False
        for name, lo, hi, step, is_int in SPACE:
            for d in (-1, 1):
                cand = dict(best)
                v = cand[name] + d * step
                if v < lo - 1e-12 or v > hi + 1e-12:
                    continue
                cand[name] = int(v) if is_int else round(v, 6)
                cv = score(cand)
                if cv < best_v:
                    best, best_v, improved = cand, cv, True
        print(f"貪欲法 sweep {sweep}: {best} {best_v:.3e}", flush=True)
        if not improved:
            break

    # --- TPE: 貪欲法の結果を初期点にして周辺を探索 ---
    def objective(trial):
        kw = {}
        for name, lo, hi, _step, is_int in SPACE:
            kw[name] = (trial.suggest_int(name, lo, hi) if is_int
                        else round(trial.suggest_float(name, lo, hi), 6))
        return score(kw)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
    study.enqueue_trial(best)
    study.optimize(objective, n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 30)

    # --- 上位 4 候補を検証用で比較 ---
    ranked = sorted({json.dumps(e["kw"], sort_keys=True): e for e in log}.values(),
                    key=lambda e: e["adj"] if e["resid"] < 0.38 else 1.0)[:4]
    ev2 = Evaluator(SPLITS["valid"])
    valid = []
    for e in [dict(kw=START)] + ranked:
        r = ev2.score(PATH, **e["kw"])
        valid.append(dict(kw=e["kw"], tune_adj=e.get("adj"), valid_adj=r["adj"],
                          valid_mse=r["mse"], util=r["util"], resid=r["resid"]))
        print(f"検証 {e['kw']}: 調整後 {r['adj']:.3e} 生MSE {r['mse']:.3e} "
              f"C/B {r['util']:.3f} resid {r['resid']:.2f}s", flush=True)
    choice = min(valid[1:], key=lambda v: v["valid_adj"] if v["resid"] < 0.38 else 1.0)
    print("採用:", choice, flush=True)
    OUT.write_text(json.dumps(dict(log=log, valid=valid, choice=choice), indent=1,
                              ensure_ascii=False))


if __name__ == "__main__":
    main()
