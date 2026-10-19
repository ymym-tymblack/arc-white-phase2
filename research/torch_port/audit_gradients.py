"""決定的な訓練網で FP32 鎖の勾配と再計算経路を監査する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.utils.checkpoint

import chain
from train import NetSet, LOSS_SCALE, net_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--indices", default="0,1")
    ap.add_argument("--fp64-shadow", action="store_true",
                    help="diagnostic only: identical formulas without FP32 rounding; never used for training")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if a.fp64_shadow:
        chain.F32 = torch.float64
    dtype = chain.F32
    P = chain.ChainParams(ck["args"]["source"]).cuda()
    P.load_state_dict(ck["params"])
    ds = NetSet(["train64", "train160"])
    report = dict(checkpoint=str(a.checkpoint), sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  precision="FP64 diagnostic shadow" if a.fp64_shadow else "production mixed FP32/FP64",
                  epsilons=[1e-2, 3e-3, 1e-3], loss="final-layer MSE * 1e8", networks=[],
                  tolerance=dict(relative=0.05, absolute=0.003,
                                 rationale="FP32 predictions near order 1 are differenced before a 1e8-scaled tiny MSE; use 5% plus 0.003 absolute derivative, report every epsilon. Checkpoint path requires 1e-6 relative + 1e-7 absolute gradients."),
                  static_review=dict(detaches="only export serialization; none in predict/layer",
                                     inplace="state is replaced, not mutated across checkpoint recomputation",
                                     qr="reduced QR gradients remain connected; finite differences include QR paths",
                                     rng="predict has no random draws; QR sketches use fixed next-layer weights"))
    gen = torch.Generator(device="cuda").manual_seed(404)
    direction = [torch.randn(p.shape, device="cuda", generator=gen) for p in P.corr_W]
    norm = torch.sqrt(sum((d.double() ** 2).sum() for d in direction))
    direction = [(d / norm).to(dtype) for d in direction]
    targets = [("b2", [(P.b2, torch.ones_like(P.b2))]),
               ("regen_p1", [(P.regen_p, torch.tensor([0., 1., 0., 0.], device="cuda"))]),
               ("regen_feed", [(P.regen_feed, torch.ones_like(P.regen_feed))]),
               ("regen_use31", [(P.regen_use31, torch.ones_like(P.regen_use31))]),
               ("corrector_W_unit_direction", list(zip(P.corr_W, direction)))]
    for index in map(int, a.indices.split(",")):
        W, truth, gid = ds[index]
        W, truth = W[None].cuda().to(dtype), truth[None].cuda()
        assert tuple(W.shape[1:]) == (16, 1024, 1024)
        rec = dict(gid=gid, shape=list(W.shape), differences=[])
        def loss(checkpoint=False):
            pred = chain.predict(P, W, checkpoint=checkpoint)
            return net_losses(pred, truth, 0)[0].mean() * LOSS_SCALE
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone()
        grads = []
        losses = []
        for checkpoint in (False, True, True):
            P.zero_grad(set_to_none=True)
            value = loss(checkpoint)
            value.backward()
            losses.append(float(value.detach()))
            grads.append({n: p.grad.detach().clone() for n, p in P.named_parameters()})
            del value
        rec["path_losses"] = losses
        rec["rng_unchanged"] = bool(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        rec["checkpoint_gradients"] = {}
        for n in grads[0]:
            x, y = grads[0][n], grads[1][n]
            rec["checkpoint_gradients"][n] = dict(max_abs=float((x-y).abs().max()),
                l2_relative=float(torch.linalg.vector_norm(x-y) / torch.linalg.vector_norm(x).clamp_min(1e-30)),
                pass_tolerance=bool(torch.allclose(x, y, rtol=1e-6, atol=1e-7)),
                repeated_equal=bool(torch.equal(y, grads[2][n])), finite=bool(torch.isfinite(x).all()))
        P.zero_grad(set_to_none=True)
        # Keep parameter copies, not a graph, during finite differences.
        named = {id(p): n for n, p in P.named_parameters()}
        with torch.no_grad():
            repeated = [float(loss()), float(loss())]
            rec["repeated_forward_losses"] = repeated
            for label, pieces in targets:
                auto = sum(float((grads[0][named[id(p)]].double() * d.double()).sum()) for p, d in pieces)
                originals = [p.clone() for p, _ in pieces]
                for eps in report["epsilons"]:
                    values = []
                    actual = []
                    for sign in (1, -1):
                        for (p, d), original in zip(pieces, originals):
                            p.copy_(original + sign * eps * d)
                        values.append(float(loss()))
                        actual.append(sum(float(((p-original).double()*d.double()).sum()) for (p,d),original in zip(pieces,originals)))
                    for (p, _), original in zip(pieces, originals):
                        p.copy_(original)
                    fd = (values[0]-values[1])/(2*eps)
                    error = abs(fd-auto)
                    rec["differences"].append(dict(parameter=label, epsilon=eps, autograd=auto,
                        plus_loss=values[0], minus_loss=values[1], actual_directional_steps=actual,
                        centered_difference=fd, absolute_error=error,
                        relative_error=error/max(abs(fd),abs(auto),1e-30),
                        pass_tolerance=error <= .003 + .05*max(abs(fd),abs(auto))))
        rec["pass"] = (all(d["pass_tolerance"] for d in rec["differences"])
                       and all(d["pass_tolerance"] and d["repeated_equal"] and d["finite"] for d in rec["checkpoint_gradients"].values())
                       and rec["rng_unchanged"] and repeated[0] == repeated[1] and losses[0] == losses[1] == losses[2])
        report["networks"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(rec), flush=True)
    report["pass"] = all(r["pass"] for r in report["networks"])
    a.out.write_text(json.dumps(report, indent=2))
    print("AUDIT", report["pass"], flush=True)


if __name__ == "__main__":
    main()
