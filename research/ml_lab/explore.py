"""収集済み特徴で Δω・Δτ の予測可能性を素早く見る(リッジ、MLP 単位の 2 分割交差検証)。

重み "sens" は ReLU 平均の感度 (φ(α)(α²-1))²(ω)・(φ(α)α)²(τ)。
DEV r0 で、感度重み付き R² ≈ 鎖内での MSE 改善 / オラクル改善 の比になっていた。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_corrector import FEATS, FEATURE_NAMES, feature_matrix  # noqa: E402


def sens(x, target):
    al = x[:, FEATURE_NAMES.index("alpha")]
    g = (al * al - 1.0) if target == "omega" else al
    return (np.exp(-0.5 * al * al) * g) ** 2


def ridge_r2(xs, ys, ws, cols, poly=False):
    def design(x):
        x = x[:, cols]
        if poly:
            x = np.concatenate([x, (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)], 1)
        return x
    out = []
    half = len(xs) // 2
    for tr, te in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        Xtr = np.concatenate([design(x) for x in xs[tr]])
        Ytr = np.concatenate(ys[tr])
        Wtr = np.concatenate(ws[tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
        coef = np.linalg.solve((A * Wtr[:, None]).T @ A + 1e-2 * np.eye(A.shape[1]), (A * Wtr[:, None]).T @ Ytr)
        Xte = np.concatenate([design(x) for x in xs[te]])
        Yte = np.concatenate(ys[te])
        Wte = np.concatenate(ws[te])
        At = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], 1)
        out.append(1 - (Wte * (At @ coef - Yte) ** 2).sum() / (Wte * Yte**2).sum())
    return float(np.mean(out))


def main():
    rows = sorted(int(p.stem.split("_")[1]) for p in FEATS.glob("row_*.npz") if p.stem.count("_") == 1)
    rows = [r for r in rows if not (64 <= r < 256)][:40]
    print("rows", len(rows))
    sets = {"base9": [FEATURE_NAMES.index(f) for f in (
        "omega", "omega_diag", "omega_regen", "metric_over_var", "tv_over_var", "alpha", "tdiag",
        "layer", "lam")], "all17": list(range(17)), "all19": list(range(19))}
    for target in ("omega", "tdiag"):
        xs, ys = [], []
        for r in rows:
            x, y, _ = feature_matrix(np.load(FEATS / f"row_{r:05d}.npz"), target)
            xs.append(x)
            ys.append(y)
        for wname in ("none", "sens"):
            ws = [np.ones(len(x)) if wname == "none" else sens(x, target) for x in xs]
            for name, cols in sets.items():
                print(f"{target} w={wname} {name} lin {ridge_r2(xs, ys, ws, cols):.3f} "
                      f"quad {ridge_r2(xs, ys, ws, cols, poly=True):.3f}", flush=True)
            Y, W = np.concatenate(ys), np.concatenate(ws)
            for f in FEATURE_NAMES:
                X = np.concatenate([x[:, FEATURE_NAMES.index(f)] for x in xs])
                Xc, Yc = X - np.average(X, weights=W), Y - np.average(Y, weights=W)
                c = np.average(Xc * Yc, weights=W) / np.sqrt(np.average(Xc**2, weights=W) * np.average(Yc**2, weights=W))
                print(f"   corr {f:16s} {c:+.3f}")


if __name__ == "__main__":
    main()
