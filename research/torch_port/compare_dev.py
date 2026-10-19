"""eval_dev_candidate.py の結果 2 つ(候補・基準)を網ごとに対にして比べる。

平均 MSE の比 mean(候補)/mean(基準)、その網単位ブートストラップ 95% CI、網ごとの比の平均、改善した網の数。
例: python research/torch_port/compare_dev.py results/dev64_adam_fit.json results/dev64_head.json
"""
from __future__ import annotations

import json
import random
import sys


def compare(cand, head):
    """同じ網のMSEと課金FLOPから、精度・調整後スコアの対比較を返す。"""
    if cand.get("failures", 0) or head.get("failures", 0):
        raise ValueError("Prediction failures prevent an accuracy comparison")
    a = {p["idx"]: p for p in cand["per"]}
    b = {p["idx"]: p for p in head["per"]}
    if not a or set(a) != set(b) or len(a) != len(cand["per"]) or len(b) != len(head["per"]):
        raise ValueError("Candidate/reference indices must match exactly, without duplicates")
    idx = sorted(a)
    x = [a[i]["mse"] for i in idx]
    y = [b[i]["mse"] for i in idx]
    ax = [a[i]["mse"] * max(0.1, a[i]["flops"] / 2**41) for i in idx]
    ay = [b[i]["mse"] * max(0.1, b[i]["flops"] / 2**41) for i in idx]
    ratio = sum(x) / sum(y)
    rng = random.Random(0)
    boots, adj_boots, paired_adj_boots = [], [], []
    for _ in range(10000):
        pick = [rng.randrange(len(idx)) for _ in idx]
        boots.append(sum(x[k] for k in pick) / sum(y[k] for k in pick))
        adj_boots.append(sum(ax[k] for k in pick) / sum(ay[k] for k in pick))
        paired_adj_boots.append(sum(ax[k] / ay[k] for k in pick) / len(pick))
    boots.sort()
    adj_boots.sort()
    paired_adj_boots.sort()
    out = dict(
        n=len(idx), cand_mse=sum(x) / len(x), head_mse=sum(y) / len(y), ratio=ratio,
        ci95=[boots[249], boots[9749]],
        mean_per_net_ratio=sum(p / q for p, q in zip(x, y)) / len(idx),
        improved=sum(p < q for p, q in zip(x, y)),
        cand_adj=sum(ax) / len(idx), head_adj=sum(ay) / len(idx),
        adj_ratio_of_means=sum(ax) / sum(ay),
        adj_ratio_of_means_ci95=[adj_boots[249], adj_boots[9749]],
        mean_per_net_adj_ratio=sum(p / q for p, q in zip(ax, ay)) / len(idx),
        mean_per_net_adj_ratio_ci95=[paired_adj_boots[249], paired_adj_boots[9749]],
        improved_adj=sum(p < q for p, q in zip(ax, ay)),
        cand_util=cand.get("util"), head_util=head.get("util"),
    )
    return out


def main():
    cand = json.loads(open(sys.argv[1]).read())
    head = json.loads(open(sys.argv[2]).read())
    print(json.dumps(compare(cand, head), indent=1))


if __name__ == "__main__":
    main()
