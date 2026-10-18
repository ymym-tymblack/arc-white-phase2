"""手順2: 鎖の周辺量を真値に差し替えたときの最終層 MSE(DEV 0〜15、MLP ごとの比)。

変種: base / omega / tdiag / both / mv{L}(層 L 以降の m・var)/ 追加の上限確認。
結果: results/ml_lab_oracle.json(MLP ごとの MSE、比の平均・ブートストラップ 95% CI、
鎖の周辺量の層別誤差)。
"""
from __future__ import annotations

import os

# spawn の子もこのモジュールを読み込むので、numpy より先に BLAS のスレッド数を固定する。
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, os.environ.get("ML_LAB_THREADS", "4"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import DEV_OFFSET, ORACLE_PATH, ROOT, Runner, load_truth  # noqa: E402

VARIANTS = {
    "base": {},
    "omega": {"omega": True},
    "tdiag": {"tdiag": True},
    "both": {"omega": True, "tdiag": True},
    "mv8": {"mv_from": 8},
    "mv12": {"mv_from": 12},
    "mv15": {"mv_from": 15},
    "mv1": {"mv_from": 1},
    "mv8_both": {"mv_from": 8, "omega": True, "tdiag": True},
    "mv1_both": {"mv_from": 1, "omega": True, "tdiag": True},
    "omega_mean": {"omega_mean": True},
    "omega_k8": {"omega": True, "k_from": 8},
    "both_k15": {"omega": True, "tdiag": True, "k_from": 15},
    "both_k12": {"omega": True, "tdiag": True, "k_from": 12},
    "both_to14": {"omega": True, "tdiag": True, "k_to": 14},
    "omega_shape": {"omega_shape": True},
    "both_shape": {"omega_shape": True, "tdiag": True},
}


def boot_ci(x, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def marginal_errors(trace, truth):
    """層ごとの鎖の誤差(真値との差)。ω・τ は σ 規格化済みの量で比べる。"""
    out = []
    for layer, r in enumerate(trace):
        e = dict(layer=layer)
        tm, tv = truth["m"][layer], truth["var"][layer]
        ts = np.sqrt(tv)
        e["m_err_over_sigma_rms"] = float(np.sqrt((((r["m"] - tm) / ts) ** 2).mean()))
        e["var_relerr_rms"] = float(np.sqrt((((r["var"] - tv) / tv) ** 2).mean()))
        if layer >= 1:
            for key in ("tdiag", "omega"):
                if r[key] is None:
                    continue
                c, t = np.broadcast_to(r[key], tm.shape), truth[key][layer]
                e[f"{key}_true_rms"] = float(np.sqrt((t**2).mean()))
                e[f"{key}_err_rms"] = float(np.sqrt(((c - t) ** 2).mean()))
                e[f"{key}_corr"] = float(np.corrcoef(c, t)[0, 1])
                e[f"{key}_true_mean"] = float(t.mean())
                e[f"{key}_chain_mean"] = float(c.mean())
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="0-15")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "ml_lab_oracle.json"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.idxs.split("-"))
    idxs = list(range(lo, hi + 1))
    names = a.variants.split(",")
    runner = Runner(a.workers, a.threads)
    jobs = [(ORACLE_PATH, i, i + DEV_OFFSET, VARIANTS[v], {}, v == "base")
            for v in names for i in idxs]
    t0 = time.time()
    res = runner.map(jobs)
    by = {v: res[k * len(idxs):(k + 1) * len(idxs)] for k, v in enumerate(names)}
    report = dict(idxs=idxs, rows=[i + DEV_OFFSET for i in idxs], variants={}, wall=time.time() - t0)
    base = by["base"]
    report["marginal_errors"] = {
        r["row"]: marginal_errors(r["trace"], load_truth(r["row"])) for r in base
    }
    base_mse = np.array([r["mse"] for r in base])
    for v in names:
        rr = by[v]
        errs = [r for r in rr if "error" in r]
        ok = [k for k, r in enumerate(rr) if "error" not in r]
        mse = np.array([rr[k]["mse"] for k in ok])
        ratio = mse / base_mse[ok]
        entry = dict(
            spec=VARIANTS[v], n_ok=len(ok), errors=[(r["idx"], r["error"]) for r in errs],
            mean_mse=float(mse.mean()) if len(ok) else None,
            base_mean_mse_same=float(base_mse[ok].mean()) if len(ok) else None,
            mse_reduction_of_means=float(1 - mse.mean() / base_mse[ok].mean()) if len(ok) else None,
            mean_ratio=float(ratio.mean()) if len(ok) else None,
            ratio_ci95=boot_ci(ratio) if len(ok) else None,
            improved=int((ratio < 1).sum()),
            per=[dict(idx=rr[k]["idx"], mse=rr[k]["mse"], ratio=float(ratio[j]),
                      layers=rr[k]["layers"]) for j, k in enumerate(ok)],
        )
        report["variants"][v] = entry
        print(f"{v:10s} n={len(ok):2d} meanMSE={entry['mean_mse']:.3e} "
              f"red={entry['mse_reduction_of_means']:+.3f} ratio={entry['mean_ratio']:.3f} "
              f"CI={entry['ratio_ci95']} improved={entry['improved']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
