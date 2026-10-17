"""accuracy_lab 変種を DEV(full_dev64)で対比較するハーネス。

各 (変種, MLP) を独立したジョブとして spawn プールで走らせ、例外(負分散ガード等)は
MLP ごとに記録する。結果は 1 ジョブ終わるごとに results/accuracy_lab.json へ書き出す
(ソース SHA256 が同じものだけ再利用)。要約は基準(全スイッチ無効)との MLP ごとの
最終層 MSE 比(平均・ブートストラップ95%CI・改善数)。

例: uv run python research/accuracy_lab_eval.py --variants base sphere --idx 0-7
    uv run python research/accuracy_lab_eval.py --summary --idx 0-63
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import os  # noqa: E402

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # spawn 子の flopscope 警告を抑える
import eval_mini  # noqa: E402

PATH = ROOT / "research" / "variants" / "accuracy_lab.py"
OUT = ROOT / "results" / "accuracy_lab.json"
DATASET = "/kaggle/ext/full_dev64"
B = 2.0 ** 41

# 層0を厳密にする設定: λ_0=g のとき (3,1) と κ4→κ3 誕生はどちらも 2 倍が厳密(MC 検算済み)。
_EXACT0 = dict(sphere_input=True, sphere_use31_0=2.0, sphere_feed0=2.0, sphere_exact0=True)
_C1024 = math.sqrt(2.0 / 1024) * math.exp(math.lgamma(1025 / 2.0) - math.lgamma(512.0))  # c_n(n=1024)
VARIANTS = {
    "base": {},
    "sphere": dict(sphere_input=True),
    "sphere_exact0": _EXACT0,
    "k31b2_1.0": dict(k31_two_term=1.0),
    "k31b2_1.3": dict(k31_two_term=1.3),
    "kmax4": dict(kmax=4),
    "kmax5": dict(kmax=5),
    "k31b2_1.6": dict(k31_two_term=1.6),
    "k31b2_2.0": dict(k31_two_term=2.0),
    "sphere+b2_1.0": dict(sphere_input=True, k31_two_term=1.0),
    "sphere+b2_1.3": dict(sphere_input=True, k31_two_term=1.3),
    "sphere_exact0+b2_1.0": dict(_EXACT0, k31_two_term=1.0),
    "sphere_exact0+b2_1.3": dict(_EXACT0, k31_two_term=1.3),
    # b2 は burned(公式 full 0〜7)で 1.0〜4.0 を走査し 3.0 が最小(DEV は見ていない)
    "k31b2_2.5": dict(k31_two_term=2.5),
    "k31b2_3.0": dict(k31_two_term=3.0),
    "k31b2_4.0": dict(k31_two_term=4.0),
    "sphere_exact0+b2_2.0": dict(_EXACT0, k31_two_term=2.0),
    "sphere_exact0+b2_3.0": dict(_EXACT0, k31_two_term=3.0),
    "sphere+b2_3.0": dict(sphere_input=True, k31_two_term=3.0),
    "sphere+b2_2.5": dict(sphere_input=True, k31_two_term=2.5),
    "sphere+b2_3.5": dict(sphere_input=True, k31_two_term=3.5),
    "sphere+b2_4.0": dict(sphere_input=True, k31_two_term=4.0),
    # 寄与分解: 最終層の MSE だけを見るので、calib で c_n の掛け算だけを足す/外す(n=1024 固定)。
    "cn_only": dict(calib=1.0 - _C1024),
    "sphere_no_cn": dict(sphere_input=True, calib=1.0 - 1.0 / _C1024),
}


def parse_idx(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}}


def save(db):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=1))
    tmp.replace(OUT)


# dev = 比較用 DEV、burned = 係数の当てはめに使ってよい公式 full 0〜63 番
DATASETS = {"dev": DATASET, "burned": "/kaggle/ext/compute_full64"}
SET = "dev"


def key_of(name, sha):
    suffix = "" if SET == "dev" else f"#{SET}"
    return f"{name}@{sha[:10]}{suffix}"


def run(names, idxs, workers, threads, extra):
    sha = hashlib.sha256(PATH.read_bytes()).hexdigest()
    db = load()
    jobs = []
    for name in names:
        kwargs = dict(VARIANTS[name]) if name in VARIANTS else {}
        kwargs.update(extra.get(name, {}))
        entry = db["runs"].setdefault(key_of(name, sha), dict(name=name, sha=sha, kwargs=kwargs, per={}))
        if entry["kwargs"] != kwargs:
            raise SystemExit(f"{name}: kwargs differ from cached run; rename the variant")
        jobs += [(name, i) for i in idxs if str(i) not in entry["per"]]
    if not jobs:
        return sha
    ident = eval_mini.dataset_identity(DATASETS[SET], "full")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=eval_mini._init, initargs=(threads, ident))
    # 同じ MLP を続けて読むよう MLP 番号順に並べる(ワーカーの MLP キャッシュを活かす)
    jobs.sort(key=lambda j: (j[1], j[0]))
    futs = {}
    for name, i in jobs:
        kw = db["runs"][key_of(name, sha)]["kwargs"]
        futs[pool.submit(eval_mini._run, (str(PATH), kw, i))] = (name, i)
    t0 = time.time()
    for done, fut in enumerate(as_completed(futs), 1):
        name, i = futs[fut]
        try:
            res = fut.result()
        except Exception as exc:  # ガード(ValueError)を MLP ごとに記録する
            res = dict(idx=i, error=f"{type(exc).__name__}: {exc}")
        db["runs"][key_of(name, sha)]["per"][str(i)] = res
        save(db)
        tag = res.get("error") or f"mse={res['mse']:.3e} C/B={res['flops'] / B:.3f} {res['sec']:.0f}s"
        print(f"[{done}/{len(jobs)} {time.time() - t0:.0f}s] {name} #{i}: {tag}", flush=True)
    pool.shutdown()
    return sha


def summarize(names, idxs, sha, boot=10000):
    db = load()
    base = db["runs"].get(key_of("base", sha), {}).get("per", {})
    rows = {}
    for name in names:
        per = db["runs"].get(key_of(name, sha), {}).get("per", {})
        got = [i for i in idxs if str(i) in per]
        errors = {i: per[str(i)]["error"] for i in got if "error" in per[str(i)]}
        ok = [i for i in got if i not in errors]
        if not ok:
            rows[name] = dict(n=0, errors=errors)
            continue
        mse = sum(per[str(i)]["mse"] for i in ok) / len(ok)
        util = sum(per[str(i)]["flops"] for i in ok) / len(ok) / B
        adj = sum(per[str(i)]["mse"] * max(0.1, per[str(i)]["flops"] / B) for i in ok) / len(ok)
        row = dict(n=len(ok), mse=mse, util=util, adj=adj, errors=errors)
        paired = [i for i in ok if str(i) in base and "error" not in base[str(i)]]
        if paired and name != "base":
            ratios = [per[str(i)]["mse"] / base[str(i)]["mse"] for i in paired]
            rng = random.Random(0)
            means = sorted(
                sum(rng.choice(ratios) for _ in ratios) / len(ratios) for _ in range(boot)
            )
            row.update(
                paired_n=len(paired), ratio_mean=sum(ratios) / len(ratios),
                ratio_ci95=[means[int(0.025 * boot)], means[int(0.975 * boot) - 1]],
                improved=sum(r < 1.0 for r in ratios),
                mse_ratio_of_means=(
                    sum(per[str(i)]["mse"] for i in paired) / sum(base[str(i)]["mse"] for i in paired)
                ),
                flops_ratio=(
                    sum(per[str(i)]["flops"] for i in paired) / sum(base[str(i)]["flops"] for i in paired)
                ),
            )
        rows[name] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--idx", default="0-7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extra", default="{}", help='JSON {"name": {kwargs}} で臨時の変種を定義')
    ap.add_argument("--summary", action="store_true", help="評価せず要約だけ")
    ap.add_argument("--tag", default=None, help="要約を summaries[tag] として保存")
    ap.add_argument("--dataset", choices=list(DATASETS), default="dev")
    ap.add_argument("--path", default=None, help="別の推定器ファイル(SHA でキーが分かれる)")
    args = ap.parse_args()
    global SET, PATH
    SET = args.dataset
    if args.path:
        PATH = Path(args.path).resolve()
    idxs = parse_idx(args.idx)
    extra = json.loads(args.extra)
    names = args.variants + [k for k in extra if k not in args.variants]
    sha = hashlib.sha256(PATH.read_bytes()).hexdigest()
    if not args.summary:
        sha = run(sorted(set(names) | {"base"}), idxs, args.workers, args.threads, extra)
    rows = summarize(names, idxs, sha)
    for name, r in rows.items():
        if not r["n"]:
            print(f"{name:24s} no results; errors={r['errors']}")
            continue
        line = f"{name:24s} n={r['n']:2d} mse={r['mse']:.4e} C/B={r['util']:.4f} adj={r['adj']:.4e}"
        if "ratio_mean" in r:
            lo, hi = r["ratio_ci95"]
            line += (f" ratio={r['ratio_mean']:.4f} [{lo:.4f},{hi:.4f}] "
                     f"improved={r['improved']}/{r['paired_n']} flops×{r['flops_ratio']:.4f}")
        if r["errors"]:
            line += f" ERRORS={r['errors']}"
        print(line)
    if args.tag:
        db = load()
        db.setdefault("summaries", {})[args.tag] = dict(sha=sha, dataset=SET, idx=idxs, rows=rows)
        save(db)


if __name__ == "__main__":
    main()
