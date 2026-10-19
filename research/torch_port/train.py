"""PyTorch 移植の鎖で、b2・再生の係数・学習補正器を最終層 MSE に対して Adam で当てはめる(GPU 環境)。

損失: ミニバッチの MLP ごとの最終層 MSE(真値 = 公式 N=1e9 の all_layer_means[-1])の平均 × 1e8
(--aux-weight > 0 なら他の層の MSE 平均にその重みを掛けて足す。既定 0 = 最終層のみ)。
学習: 公式 full 0〜63(train64)、264〜423(train160)、424〜621(train198、あれば)。
選択: 毎エポック開発 64 網(公式 64〜127)の最終層 MSE 平均。最終確認用の集合は読まない。
W&B はオフライン(project arc-whitebox-adam)、チェックポイントは /kaggle/ext/torch_ckpt/RUN/。

例: WANDB_MODE=offline WANDB_DIR=/kaggle/ext/wandb research/gpu_env.sh research/torch_port/train.py --run adam1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/kaggle/ext/torch_data")
CKPT = Path("/kaggle/ext/torch_ckpt")
LOSS_SCALE = 1e8


class NetSet(torch.utils.data.Dataset):
    def __init__(self, names):
        self.items = []
        for name in names:
            d = DATA / name
            if not (d / "truth.npy").exists():
                print(f"skip {name}: not prepared", flush=True)
                continue
            ids = json.loads((d / "ids.json").read_text())
            truth = np.load(d / "truth.npy")
            for k, gid in enumerate(ids):
                self.items.append((d / f"w_{gid:04d}.npy", truth[k], gid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, truth, gid = self.items[i]
        return torch.from_numpy(np.load(path)), torch.from_numpy(truth), gid


def loader(ds, batch, shuffle, seed, workers=2):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g,
                                       num_workers=workers, pin_memory=True,
                                       persistent_workers=False, prefetch_factor=2)


def net_losses(pred, truth, aux_weight):
    err = (pred.double() - truth) ** 2  # (B, depth, n)
    final = err[:, -1].mean(1)
    if aux_weight > 0:
        return final, final + aux_weight * err[:, :-1].mean((1, 2))
    return final, final


@torch.no_grad()
def evaluate(P, ds, bs, dev):
    mses, gids = [], []
    for W, truth, gid in loader(ds, bs, False, 0):
        pred = chain.predict(P, W.to(dev, non_blocking=True))
        final, _ = net_losses(pred, truth.to(dev), 0.0)
        mses += final.cpu().tolist()
        gids += [int(x) for x in gid]
    return np.array(mses), gids


def save(path, **obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default=str(ROOT / "research/frozen_phase2/combined.py"))
    ap.add_argument("--train-sets", default="train64,train160,train198")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--mb", type=int, default=8, help="MLPs per optimizer step")
    ap.add_argument("--bs", type=int, default=4, help="MLPs per forward batch")
    ap.add_argument("--val-bs", type=int, default=8)
    ap.add_argument("--lr-scalar", type=float, default=1e-2)
    ap.add_argument("--lr-corr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=10, help="steps between latest.pt saves")
    ap.add_argument("--no-ckpt-grad", action="store_true", help="disable per-layer checkpointing")
    ap.add_argument("--freeze", default="", help="comma list: scalars,corrector")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import wandb

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)  # CPU は Python の発行と読み込みだけ(他の作業と CPU を分ける)
    out = CKPT / a.run
    out.mkdir(parents=True, exist_ok=True)
    train = NetSet([s for s in a.train_sets.split(",") if s])
    dev_ds = NetSet(["dev64"])
    if any(g in range(128, 256) for _, _, g in train.items + dev_ds.items):
        raise SystemExit("forbidden rows 128-255 present")
    P = chain.ChainParams(a.source).to(dev)
    freeze = set(x for x in a.freeze.split(",") if x)
    groups = []
    if "scalars" not in freeze:
        groups.append(dict(params=P.scalar_groups(), lr=a.lr_scalar, name="scalars"))
    if "corrector" not in freeze:
        groups.append(dict(params=P.corrector_groups(), lr=a.lr_corr, name="corrector"))
    for p in P.parameters():
        p.requires_grad_(False)
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    opt = torch.optim.Adam(groups)
    steps_per_epoch = math.ceil(len(train) / a.mb)
    total = steps_per_epoch * a.epochs

    def lr_lambda(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - a.warmup) / max(1, total - a.warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    state = dict(epoch=0, step=0, best=None, best_epoch=-1, history=[], baseline=None, bad=0)
    if a.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        P.load_state_dict(ck["params"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        print("resumed", {k: state[k] for k in ("epoch", "step", "best", "best_epoch")}, flush=True)

    run = wandb.init(project="arc-whitebox-adam", name=a.run, id=a.run, resume="allow",
                     dir=os.environ.get("WANDB_DIR", "/kaggle/ext/wandb"),
                     config=dict(vars(a), n_train=len(train), n_dev=len(dev_ds),
                                 loss_scale=LOSS_SCALE, loss="final-layer MSE mean (+aux)"))

    def checkpoint(name):
        save(out / name, params=P.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
             state=state, args=vars(a), export=P.export())

    if state["baseline"] is None:
        t = time.time()
        base, gids = evaluate(P, dev_ds, a.val_bs, dev)
        state["baseline"] = dict(mse=base.tolist(), gids=gids)
        state["best"] = float(base.mean())
        print(f"baseline dev64 mse {base.mean():.4e} ({time.time()-t:.0f}s)", flush=True)
        run.log(dict(dev_mse=float(base.mean()), dev_ratio=1.0, epoch=0), step=0)
        checkpoint("epoch_000.pt")
    base = np.array(state["baseline"]["mse"])

    while state["epoch"] < a.epochs:
        epoch = state["epoch"]
        P.train()
        t0 = time.time()
        start_step = state["step"] - epoch * steps_per_epoch
        dl = loader(train, a.bs, True, a.seed * 1000 + epoch)
        acc_n, acc_loss, acc_final, it = 0, 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        skip_batches = start_step * (a.mb // a.bs)
        for bi, (W, truth, gid) in enumerate(dl):
            if bi < skip_batches:
                continue
            W, truth = W.to(dev, non_blocking=True), truth.to(dev, non_blocking=True)
            pred = chain.predict(P, W, checkpoint=not a.no_ckpt_grad)
            final, loss_net = net_losses(pred, truth, a.aux_weight)
            loss = loss_net.sum() * (LOSS_SCALE / a.mb)
            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} batch {bi} gids {gid.tolist()}; skipped",
                      flush=True)
                continue
            loss.backward()
            acc_n += len(gid)
            acc_loss += float(loss)
            acc_final += float(final.sum())
            it += 1
            if acc_n >= a.mb or bi == len(dl) - 1:
                grads = [p.grad for g in groups for p in g["params"] if p.grad is not None]
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], a.clip))
                if math.isfinite(gnorm):
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                state["step"] += 1
                ex, _ = P.export()
                log = dict(train_loss=acc_loss, train_final_mse=acc_final / acc_n, grad_norm=gnorm,
                           lr_scalar=sched.get_last_lr()[0], **{f"p_{k}": v for k, v in ex.items()},
                           yscale=float(P.corr_yscale), epoch=epoch + bi / len(dl))
                run.log(log, step=state["step"])
                if state["step"] % 5 == 0:
                    print(f"ep {epoch} step {state['step']} loss {acc_loss:.4f} "
                          f"mse {acc_final/acc_n:.3e} gn {gnorm:.3g} "
                          f"b2 {ex['k31_two_term']:.4f} p {ex['regen_p0']:.3f},{ex['regen_p1']:.3f},"
                          f"{ex['regen_p2']:.3f},{ex['regen_p3']:.3f} feed {ex['regen_feed']:.3f} "
                          f"use31 {ex['regen_use31']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                if state["step"] % a.ckpt_every == 0:
                    checkpoint("latest.pt")
                acc_n, acc_loss, acc_final = 0, 0.0, 0.0
                del grads
        # --- 検証 ---
        P.eval()
        t1 = time.time()
        mse, gids = evaluate(P, dev_ds, a.val_bs, dev)
        ratio = mse / base
        rec = dict(epoch=epoch + 1, dev_mse=float(mse.mean()), dev_ratio=float(mse.mean() / base.mean()),
                   dev_paired_ratio=float(np.mean(ratio)), improved=int((mse < base).sum()),
                   train_sec=t1 - t0, val_sec=time.time() - t1)
        state["history"].append(rec)
        state["epoch"] = epoch + 1
        state["step"] = (epoch + 1) * steps_per_epoch
        if rec["dev_mse"] < state["best"]:
            state["best"], state["best_epoch"], state["bad"] = rec["dev_mse"], epoch + 1, 0
            checkpoint("best.pt")
        else:
            state["bad"] += 1
        checkpoint(f"epoch_{epoch + 1:03d}.pt")
        checkpoint("latest.pt")
        run.log(dict(dev_mse=rec["dev_mse"], dev_ratio=rec["dev_ratio"],
                     dev_paired_ratio=rec["dev_paired_ratio"], dev_improved=rec["improved"],
                     epoch=epoch + 1), step=state["step"])
        print("EPOCH", json.dumps(rec), flush=True)
        if state["bad"] >= a.patience:
            print("early stop", flush=True)
            break
    (out / "history.json").write_text(json.dumps(dict(
        history=state["history"], best=state["best"], best_epoch=state["best_epoch"],
        baseline_mse=float(base.mean())), indent=1))
    run.finish()


if __name__ == "__main__":
    main()
