"""手順3: 鎖の ω(事前活性化の標準化 κ4 対角)を真値へ寄せる小さな MLP 補正器を学習する。

入力: collect_features.py が保存した /kaggle/ext/ml_lab_feats/row_XXXXX.npz(層 1〜15)。
目的変数: Δω = ω_true − ω_chain(ニューロンごと)。検証は MLP 単位の分割。
GPU(torch)で学習し、W&B はオフライン、エポックごとにチェックポイントを保存する。
出力: /kaggle/ext/ml_lab_ckpt/<run>/epoch_XXX.pt、best.pt、research/ml_lab/omega_corrector.json
(推論用の重み・正規化定数。flopscope 実装 corrector.py が読む)。

実行: research/gpu_env.sh research/ml_lab/train_corrector.py --run NAME
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

FEATS = Path("/kaggle/ext/ml_lab_feats")
CKPT = Path("/kaggle/ext/ml_lab_ckpt")
ROOT = Path(__file__).resolve().parents[2]
DEV_ROWS = set(range(64, 128))
FINAL_ROWS = set(range(128, 256))


FEATURE_NAMES = ["omega", "omega_diag", "omega_regen", "metric_over_var", "tv_over_var",
                 "alpha", "tdiag", "layer", "lam", "star", "two", "lin", "k31", "diag4", "part",
                 "tri", "diag3", "q22", "q21"]
_RAW = ("star", "two", "lin", "k31", "diag4", "part", "tri", "diag3", "q22", "q21")


SPHERE = False  # True なら真値を球面入力 √n·u の値へ換算してから目的変数を作る(--sphere)


def sphere_truth(z):
    """Gaussian 入力の真値(平均・分散・τ・ω)を、球面入力 √n·u の値へ換算する。

    バイアスなし ReLU 網は 1 次正斉次なので z(x) = r̃ z(√n u)(r̃ = χ_n/√n、u と独立)。
    よって生モーメントは E[z^k] = E[r̃^k] E[z_s^k] で、E[r̃^k] = (2/n)^{k/2} Γ((n+k)/2)/Γ(n/2)。
    球面入力を扱う鎖(sphere_input)の誤差を測るにはこの換算後の値を真値とする。
    """
    import math
    mu, var = z["true_m"], z["true_var"]
    n = mu.shape[-1]
    k3 = z["true_tdiag"] * var ** 1.5
    k4 = z["true_omega"] * var * var
    m2 = var + mu * mu
    m3 = k3 + 3.0 * mu * var + mu ** 3
    m4 = k4 + 4.0 * mu * k3 + 3.0 * var * var + 6.0 * mu * mu * var + mu ** 4
    rho = {k: math.exp(0.5 * k * math.log(2.0 / n) + math.lgamma((n + k) / 2.0)
                       - math.lgamma(n / 2.0)) for k in (1, 2, 3, 4)}
    s1, s2, s3, s4 = mu / rho[1], m2 / rho[2], m3 / rho[3], m4 / rho[4]
    vs = s2 - s1 * s1
    k3s = s3 - 3.0 * s1 * s2 + 2.0 * s1 ** 3
    k4s = s4 - 4.0 * s1 * s3 - 3.0 * s2 * s2 + 12.0 * s1 * s1 * s2 - 6.0 * s1 ** 4
    return dict(m=s1, var=vs, tdiag=k3s / vs ** 1.5, omega=k4s / (vs * vs))


def feature_matrix(z, target="omega", layers=range(1, 16)):
    """(ニューロン, 特徴) 行列と目的変数(真値 − 鎖)。列は FEATURE_NAMES 順で、
    定義は oracle_estimator の feats と一致させる。保存されていない特徴は NaN。"""
    X, Y, L = [], [], []
    truth = sphere_truth(z) if SPHERE else {target: z["true_" + target]}
    for layer in layers:
        var = z["chain_var"][layer]
        vv = var * var
        metric = z["chain_metric"][layer]
        omega = z["chain_omega"][layer]
        od = metric * z["chain_tg"][layer] / vv
        f = dict(
            omega=omega, omega_diag=od, omega_regen=omega - od, metric_over_var=metric / var,
            tv_over_var=z["chain_tv"][layer] / var, alpha=z["chain_m"][layer] / np.sqrt(var),
            tdiag=z["chain_tdiag"][layer], layer=np.full_like(var, layer / 15.0),
            lam=np.full_like(var, z["lam"][layer]),
        )
        for k in _RAW:
            f[k] = z["chain_" + k][layer] if "chain_" + k in z.files else np.full_like(var, np.nan)
        X.append(np.stack([f[k] for k in FEATURE_NAMES], axis=1))
        Y.append(truth[target][layer] - z["chain_" + target][layer])
        L.append(np.full(var.shape, layer))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(L)


def load(rows, target="omega", tags=("",), demean=False):
    """demean=True なら目的変数から (MLP, 層) ごとの平均を引く(形だけを学ぶ)。"""
    xs, ys, ls, gs = [], [], [], []
    for g, (row, tag) in enumerate((r, t) for r in rows for t in tags
                                   if (FEATS / f"row_{r:05d}{t}.npz").exists()):
        z = np.load(FEATS / f"row_{row:05d}{tag}.npz")
        x, y, layer = feature_matrix(z, target)
        if demean:
            for value in np.unique(layer):
                y[layer == value] -= y[layer == value].mean()
        xs.append(x)
        ys.append(y)
        ls.append(layer)
        gs.append(np.full(len(y), g))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(ls), np.concatenate(gs))


class Net(torch.nn.Module):
    def __init__(self, d, hidden, depth):
        super().__init__()
        layers, k = [], d
        for _ in range(depth):
            layers += [torch.nn.Linear(k, hidden), torch.nn.ReLU()]
            k = hidden
        layers.append(torch.nn.Linear(k, 1))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="omega_mlp")
    ap.add_argument("--target", default="omega", choices=("omega", "tdiag"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--features", default=",".join(FEATURE_NAMES[:17]), help="all で q22・q21 も使う")
    ap.add_argument("--export", default=None, help="既定 research/ml_lab/<target>_corrector.json")
    ap.add_argument("--weight", default="none", choices=("none", "sens"),
                    help="sens: ReLU 平均の感度 (φ(α)(α²-1))²(ω)・(φ(α)α)²(τ)で重み付け")
    ap.add_argument("--tags", default="", help="特徴ファイルの接尾辞のカンマ区切り(例 ',_r1' で両方)")
    ap.add_argument("--demean", action="store_true", help="(MLP, 層) ごとの平均を除いた形だけを学ぶ")
    ap.add_argument("--sphere", action="store_true",
                    help="真値を球面入力へ換算してから学ぶ(sphere_input を使う鎖の補正器)")
    a = ap.parse_args()
    global SPHERE
    SPHERE = a.sphere

    if a.features == "all":
        a.features = ",".join(FEATURE_NAMES)
    tags = tuple(a.tags.split(","))
    rows = sorted({int(p.stem.split("_")[1]) for p in FEATS.glob("row_*.npz")
                   if p.stem[9:] in tags})
    rows = [r for r in rows if r not in DEV_ROWS and r not in FINAL_ROWS]
    if "q22" in a.features or "q21" in a.features:
        rows = [r for r in rows if "chain_q22" in np.load(FEATS / f"row_{r:05d}.npz").files]
    # データが増えても変わらない MLP 単位の分割(行番号の剰余、約 1/7 を検証に)
    val_rows = [r for r in rows if r % 7 == 3]
    train_rows = [r for r in rows if r % 7 != 3]
    keep = [FEATURE_NAMES.index(f) for f in a.features.split(",")]

    Xtr, Ytr, Ltr, _ = load(train_rows, a.target, tags, a.demean)
    Xva, Yva, Lva, _ = load(val_rows, a.target, tags, a.demean)
    if a.export is None:
        a.export = str(ROOT / "research" / "ml_lab" / f"{a.target}_corrector.json")

    def sens(X):
        if a.weight == "none":
            return np.ones(len(X), np.float32)
        al = X[:, FEATURE_NAMES.index("alpha")].astype(np.float64)
        g = (al * al - 1.0) if a.target == "omega" else al
        s = (np.exp(-0.5 * al * al) * g) ** 2
        return (s / s.mean()).astype(np.float32)

    Wtr, Wva = sens(Xtr), sens(Xva)
    Wva = Wva / Wva.mean()
    Xtr, Xva = Xtr[:, keep], Xva[:, keep]
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-12
    yscale = float(Ytr.std())
    dev = torch.device("cuda")
    tX = torch.tensor((Xtr - mean) / std, device=dev)
    tY = torch.tensor(Ytr / yscale, device=dev)
    tW = torch.tensor(Wtr, device=dev)
    vX = torch.tensor((Xva - mean) / std, device=dev)
    vY = torch.tensor(Yva / yscale, device=dev)
    vW = torch.tensor(Wva, device=dev)

    # 参考: 線形(リッジ)の検証 R²
    A = np.concatenate([(Xtr - mean) / std, np.ones((len(Xtr), 1), np.float32)], 1).astype(np.float64)
    coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ (Ytr / yscale))
    Av = np.concatenate([(Xva - mean) / std, np.ones((len(Xva), 1), np.float32)], 1)
    lin_r2 = 1 - ((Av @ coef - Yva / yscale) ** 2).mean() / ((Yva / yscale) ** 2).mean()

    os.environ.setdefault("WANDB_DIR", "/kaggle/ext/wandb")
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb
    wandb.init(project="arc-whitebox-phase2", name=a.run, config=dict(
        vars(a), n_train_rows=len(train_rows), n_val_rows=len(val_rows), lin_val_r2=lin_r2))
    print(f"train rows {len(train_rows)} val rows {len(val_rows)} samples {len(Ytr)} "
          f"ridge val R2(vs zero) {lin_r2:.4f}", flush=True)

    torch.manual_seed(0)
    net = Net(len(keep), a.hidden, a.depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(tY) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    out = CKPT / a.run
    out.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), -1)
    for epoch in range(a.epochs):
        net.train()
        order = torch.randperm(len(tY), device=dev)
        tot = 0.0
        for k in range(0, len(tY), a.batch):
            idx = order[k:k + a.batch]
            loss = (tW[idx] * (net(tX[idx]) - tY[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            pv = net(vX)
            vloss = float((vW * (pv - vY) ** 2).mean())
            per_layer = {
                f"val_r2_layer{layer}": float(1 - ((pv - vY)[torch.tensor(Lva == layer, device=dev)] ** 2).mean()
                                              / (vY[torch.tensor(Lva == layer, device=dev)] ** 2).mean())
                for layer in (1, 4, 8, 12, 15)
            }
        r2 = 1 - vloss / float((vW * vY ** 2).mean())
        wandb.log(dict(epoch=epoch, train_loss=tot / len(tY), val_loss=vloss, val_r2=r2, **per_layer))
        state = dict(net=net.state_dict(), mean=mean, std=std, yscale=yscale, keep=keep,
                     features=[FEATURE_NAMES[i] for i in keep], epoch=epoch, val_r2=r2,
                     hidden=a.hidden, depth=a.depth, train_rows=train_rows, val_rows=val_rows)
        torch.save(state, out / f"epoch_{epoch:03d}.pt")
        if vloss < best[0]:
            best = (vloss, epoch)
            torch.save(state, out / "best.pt")
        print(f"epoch {epoch} train {tot/len(tY):.4f} val {vloss:.4f} R2 {r2:.4f}", flush=True)

    state = torch.load(out / "best.pt", weights_only=False)
    layers = [m for m in Net(len(keep), a.hidden, a.depth).f if isinstance(m, torch.nn.Linear)]
    sd = state["net"]
    weights = []
    for k in range(len(layers)):
        weights.append(dict(W=sd[f"f.{2*k}.weight"].cpu().numpy().T.tolist(),
                            b=sd[f"f.{2*k}.bias"].cpu().numpy().tolist()))
    Path(a.export).write_text(json.dumps(dict(
        features=state["features"], mean=state["mean"].tolist(), std=state["std"].tolist(),
        yscale=state["yscale"], layers=weights, epoch=state["epoch"], val_r2=state["val_r2"],
        run=a.run, train_rows=train_rows, val_rows=val_rows)))
    wandb.summary.update(dict(best_epoch=best[1], best_val_loss=best[0], ridge_val_r2=lin_r2))
    wandb.finish()
    print("best epoch", best[1], "exported", a.export, flush=True)


if __name__ == "__main__":
    main()
