"""段階4: SA/ベイズ最適化(Optuna TPE)で推定器の調整項を mini 上で自動探索する。

目的関数 = mean(final_layer_mse × max(0.1, C/B))(公式スコアと同じ)。
探索変数: window(同時に保持する源の数), omega_scale(κ4 対角の倍率),
          tau_scale(3 次キュムラントの倍率), kmax(Mehler 次数)。
チューニング用 MLP と検証用 MLP を分ける(過学習の確認)。
"""
import json
import sys

sys.path.insert(0, "research")
from eval_mini import Evaluator, ROOT  # noqa: E402

PATH = str(ROOT / "candidates/k3_factored.py")
TUNE = list(range(0, 12))
VALID = list(range(12, 40))

if __name__ == "__main__":
    # optuna(→numpy)は spawn した子プロセスが main を再 import するときに読み込まれて
    # BLAS のスレッド設定を壊すため、main ガードの内側で import する。
    import optuna

    ev = Evaluator(TUNE)

    def objective(trial):
        kw = dict(
            window=trial.suggest_int("window", 6, 16),
            omega_scale=trial.suggest_float("omega_scale", 0.6, 1.6),
            tau_scale=trial.suggest_float("tau_scale", 0.8, 1.2),
            kmax=trial.suggest_int("kmax", 2, 4),
        )
        r = ev.score(PATH, **kw)
        if r["resid"] > 0.35:
            return 1.0
        print(f"trial {trial.number}: {kw} adj={r['adj']:.3e} mse={r['mse']:.3e} util={r['util']:.3f}", flush=True)
        return r["adj"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12))
    study.enqueue_trial(dict(window=16, omega_scale=1.0, tau_scale=1.0, kmax=3))
    study.optimize(objective, n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    best = study.best_params
    print("BEST", best, study.best_value)
    json.dump(dict(best=best, value=study.best_value), open(ROOT / "results/tune_k3.json", "w"), indent=1)

    ev2 = Evaluator(VALID)
    for name, kw in (("default", dict(window=16)), ("best", best)):
        r = ev2.score(PATH, **kw)
        print(f"検証({len(VALID)} MLP) {name}: adj={r['adj']:.3e} mse={r['mse']:.3e} util={r['util']:.3f} resid={r['resid']:.2f}", flush=True)
