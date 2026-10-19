"""Phase 2 推定器のクラス属性(ノブ)を開発集合 full64〜127 で調整する(提出物ではない)。

目的関数は開発行の調整後スコア(最終層 MSE × max(0.1, FLOP/2**41))の平均。
既定設定と同じ行での対の比(網ごとの調整後スコア比)を必ず併記する。
最終確認用(full_final128, full_holdout126)には触れない。

評価対象は開始時点の root `estimator.py` を固定したコピー(--candidate)。並行して root の
Strassen 段数が変わっても、同じソース上で比較が閉じるようにする(ソースの SHA256 を記録)。
各 (設定, 行) の結果は JSONL のキャッシュへ逐次追記し(checkpoint)、段階をまたいで再利用する。
Optuna の試行は sqlite に保存するので、中断しても同じコマンドで再開できる。

段階:
  baseline  既定設定を評価(既定 dev 0〜63)
  scan      age_old×rank_old×window の資源走査(dev 0〜7)
  optuna    TPE(seed 固定、op 数制約付き)で dev 0〜15
  promote   dev 0〜15 の上位を dev 0〜31 へ
  final     dev 0〜31 の上位 3 個を dev 0〜63 へ
  report    全試行と最終比較を JSON に出力
例:
  uv run --with optuna --with wandb python research/tune_phase2_knobs.py scan --workers 8 --threads 2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_mini  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = "/kaggle/ext/full_dev64"
STORE = Path("/kaggle/ext/optuna")
CACHE = STORE / "tune_phase2_cache.jsonl"
STUDY_DB = STORE / "tune_phase2.db"
STUDY_NAME = "tune_phase2_v2"  # v1 は窓を含む探索域(窓が無関係と判明して中止、記録は sqlite に残す)
RESULTS = ROOT / "results"
B = 2**41
OPS_LIMIT = 1.10  # 既定の op 数に対する上限(residual 時間の上限 0.4 s を守るため)
WALL_LIMIT = 1800.0  # 評価機の負荷で 120 s を超えても調整は続ける(採点の上限は別途 preflight で確認)

KNOBS = ("window", "age_old", "rank_old", "iik_rank", "k3leg_rank", "kmax", "regen_p0",
         "regen_p1", "regen_p2", "regen_p3", "regen_feed", "k31_two_term", "learned_omega")
SCAN_GRID = dict(age_old=(4, 5, 6, 7), rank_old=(128, 192, 256, 320), window=(16,))
# 窓は若い家族の数の上限だが、若い家族は age_old で共有基底へ移り、共有基底の家族は捨てないので
# age_old < window なら窓は効かない(輸送・読み出しが同じ演算列)。走査の窓 12/14 は確認用に
# age_old=4 の 3 階数だけ評価する(予測が窓 16 と一致することの実測)。
SCAN_WINDOW_CHECK = [dict(age_old=4, rank_old=r, window=w) for r in (128, 192, 256) for w in (12, 14)]


# ---------------------------------------------------------------- worker
def _work(args):
    """eval_mini._run と同じ採点に、op 数(len(ctx._op_log))と失敗時のゼロ予測 MSE を加える。"""
    path, kwargs, idx = args
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext

    mlp, truth = eval_mini._load_mlp(idx)
    zero_mse = float((truth[-1] ** 2).mean())
    try:
        spec = importlib.util.spec_from_file_location("cand_" + Path(path).stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        est = mod.Estimator(**kwargs)
        est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=B,
                               api_version="1.0", submission_dir=str(ROOT), seed=0))
        t = time.time()
        with flops.BudgetContext(flop_budget=B, wall_time_limit_s=WALL_LIMIT, quiet=True) as ctx:
            pred = est.predict(mlp, B)
        dt = time.time() - t
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"bad_prediction:shape={pred.shape}")
        return dict(idx=idx, ok=True, mse=float(((pred[-1] - truth[-1]) ** 2).mean()),
                    flops=int(ctx.flops_used), ops=len(ctx._op_log), sec=dt,
                    resid=float(ctx.residual_wall_time_s), zero_mse=zero_mse)
    except Exception as exc:  # 採点ではゼロ予測になるので、その MSE で罰する
        return dict(idx=idx, ok=False, error=f"{type(exc).__name__}: {exc}"[:400],
                    mse=zero_mse, flops=B, ops=None, sec=None, resid=None, zero_mse=zero_mse)


# ---------------------------------------------------------------- cache / pool
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Runner:
    def __init__(self, candidate: Path, workers: int, threads: int):
        self.candidate = candidate.resolve()
        self.sha = _sha(self.candidate)
        spec = importlib.util.spec_from_file_location("tune_defaults", self.candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.defaults = {k: getattr(mod.Estimator, k) for k in KNOBS}
        self.cache = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                rec = json.loads(line)
                if rec["sha"] == self.sha:
                    self.cache[(rec["key"], rec["idx"])] = rec
        self.workers, self.threads = workers, threads
        self.pool = None

    def canon(self, kwargs: dict) -> dict:
        """既定値と同じ上書きを除き、同じ設定が同じキーになるようにする。"""
        out = {}
        for k in KNOBS:
            if k in kwargs and kwargs[k] != self.defaults[k]:
                v = kwargs[k]
                out[k] = round(v, 6) if isinstance(v, float) else v
        return out

    def key(self, kwargs: dict) -> str:
        return json.dumps(self.canon(kwargs), sort_keys=True)

    def _ensure_pool(self):
        if self.pool is None:
            ident = eval_mini.dataset_identity(DEV, "full")
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=eval_mini._init, initargs=(self.threads, ident))

    def evaluate(self, configs: list[dict], idxs, on_row=None) -> list[dict]:
        """全設定の不足行をまとめて投げ、終わった行から順にキャッシュへ追記する。"""
        idxs = list(idxs)
        todo = [(self.canon(c), i) for c in configs for i in idxs
                if (self.key(c), i) not in self.cache]
        todo = list({(json.dumps(c, sort_keys=True), i): (c, i) for c, i in todo}.values())
        if todo:
            self._ensure_pool()
            futures = {self.pool.submit(_work, (str(self.candidate), c, i)): (c, i) for c, i in todo}
            with CACHE.open("a") as fh:
                for fut in as_completed(futures):
                    c, i = futures[fut]
                    rec = dict(fut.result(), key=json.dumps(c, sort_keys=True), sha=self.sha,
                               time=time.time())
                    self.cache[(rec["key"], i)] = rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if on_row:
                        on_row(rec)
        return [self.summary(c, idxs) for c in configs]

    def rows(self, kwargs, idxs):
        k = self.key(kwargs)
        return [self.cache[(k, i)] for i in idxs]

    def summary(self, kwargs, idxs) -> dict:
        idxs = list(idxs)
        rows = self.rows(kwargs, idxs)
        adj = [r["mse"] * max(0.1, r["flops"] / B) for r in rows]
        ok = [r for r in rows if r["ok"]]
        canon = self.canon(kwargs)
        full = dict(self.defaults, **canon)
        corrector = bool(full["learned_omega"]) and full["k31_two_term"] == 3.0
        out = dict(kwargs=canon, corrector_active=corrector, n=len(rows), failures=len(rows) - len(ok),
                   adj=sum(adj) / len(adj), mse=sum(r["mse"] for r in rows) / len(rows),
                   cb=sum(r["flops"] for r in rows) / len(rows) / B,
                   ops=max((r["ops"] for r in ok), default=None),
                   max_resid_loaded=max((r["resid"] for r in ok), default=None),
                   max_sec_loaded=max((r["sec"] for r in ok), default=None),
                   per_adj=adj, idxs=idxs)
        if (self.key({}), idxs[0]) in self.cache and all((self.key({}), i) in self.cache for i in idxs):
            out.update(paired(self.summary_base(idxs), adj))
            base_ops = self.base_ops()
            out["ops_ratio"] = None if out["ops"] is None or base_ops is None else out["ops"] / base_ops
        return out

    def summary_base(self, idxs):
        return [r["mse"] * max(0.1, r["flops"] / B) for r in self.rows({}, idxs)]

    def base_ops(self):
        ops = [r["ops"] for (k, _), r in self.cache.items() if k == self.key({}) and r["ok"]]
        return max(ops) if ops else None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def paired(base_adj, adj, n_boot=10000, seed=0) -> dict:
    """網ごとの比 adj/base の平均・#改善と、比の平均・平均の比の対ブートストラップ 95% CI。"""
    ratios = [a / b for a, b in zip(adj, base_adj)]
    n = len(ratios)
    rng = random.Random(seed)
    boot_mean, boot_rom = [], []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in range(n)]
        boot_mean.append(sum(ratios[j] for j in s) / n)
        boot_rom.append(sum(adj[j] for j in s) / sum(base_adj[j] for j in s))
    boot_mean.sort()
    boot_rom.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return dict(base_adj=sum(base_adj) / n, ratio_mean=sum(ratios) / n,
                ratio_mean_ci95=[q(boot_mean, 0.025), q(boot_mean, 0.975)],
                ratio_of_means=sum(adj) / sum(base_adj),
                ratio_of_means_ci95=[q(boot_rom, 0.025), q(boot_rom, 0.975)],
                improved=sum(r < 1.0 for r in ratios), per_ratio=ratios)


def feasible(s: dict) -> bool:
    return s["failures"] == 0 and s.get("ops_ratio") is not None and s["ops_ratio"] <= OPS_LIMIT


def parse_range(text: str):
    lo, hi = (int(x) for x in text.split("-"))
    return range(lo, hi + 1)


def short(s: dict) -> str:
    keys = ("adj", "ratio_mean", "improved", "mse", "cb", "ops_ratio", "failures")
    return " ".join(f"{k}={s.get(k):.4g}" if isinstance(s.get(k), float) else f"{k}={s.get(k)}"
                    for k in keys) + " " + json.dumps(s["kwargs"])


def save(name: str, payload) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"tune_phase2_{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.replace(path)
    return path


def wandb_run(phase: str, args, runner: Runner):
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", "/kaggle/ext/wandb")
    import wandb
    return wandb.init(project="arc-whitebox-tune", name=f"{phase}-{time.strftime('%m%d-%H%M')}",
                      config=dict(phase=phase, candidate=str(runner.candidate), sha=runner.sha,
                                  workers=args.workers, threads=args.threads, args=vars(args)))


def log_summary(run, number, s, prefix=""):
    if run is None:
        return
    row = {f"{prefix}{k}": s[k] for k in ("adj", "mse", "cb", "ops", "ratio_mean",
                                          "ratio_of_means", "improved", "failures")
           if s.get(k) is not None}
    run.log(dict(row, **{f"{prefix}number": number}))


# ---------------------------------------------------------------- phases
def phase_baseline(args, runner):
    s = runner.evaluate([{}], parse_range(args.indices))[0]
    print("default", short(s), flush=True)
    save("baseline", dict(candidate=str(runner.candidate), sha=runner.sha, summary=s,
                          rows=runner.rows({}, parse_range(args.indices))))


def scan_configs():
    return [dict(age_old=a, rank_old=r, window=w) for a in SCAN_GRID["age_old"]
            for r in SCAN_GRID["rank_old"] for w in SCAN_GRID["window"]] + SCAN_WINDOW_CHECK


def phase_scan(args, runner):
    idxs = parse_range(args.indices)
    runner.evaluate([{}], idxs)
    run = wandb_run("scan", args, runner)
    configs = scan_configs()
    out = []
    done = [0]

    def on_row(rec):
        done[0] += 1
        if done[0] % 8 == 0:
            print(f"  rows {done[0]}", flush=True)

    sums = runner.evaluate(configs, idxs, on_row=on_row)
    for step, s in enumerate(sorted(sums, key=lambda s: s["adj"])):
        log_summary(run, step, s, "scan/")
        out.append(s)
        print(short(s), flush=True)
    save("scan", dict(candidate=str(runner.candidate), sha=runner.sha, indices=list(idxs),
                      grid=SCAN_GRID, window_check=SCAN_WINDOW_CHECK,
                      default=runner.summary({}, idxs), configs=out))
    if run:
        run.finish()


def suggest(trial, defaults) -> dict:
    """既定値の周りの局所探索(試行1回が数十分かかるので、走査と v1 の打ち切りで外れた域は除く)。
    窓は age_old < 16 では効かないので探索しない(既定 16 のまま)。"""
    kw = dict(
        age_old=trial.suggest_int("age_old", 5, 8),
        rank_old=trial.suggest_int("rank_old", 160, 320, step=32),
        iik_rank=trial.suggest_int("iik_rank", 2, 12, step=2),
        k3leg_rank=trial.suggest_int("k3leg_rank", 2, 12, step=2),
        kmax=trial.suggest_int("kmax", 2, 4),
        regen_p0=trial.suggest_float("regen_p0", 1.0, 2.2),
        regen_p1=trial.suggest_float("regen_p1", 0.7, 1.8),
        regen_p2=trial.suggest_float("regen_p2", 0.7, 1.8),
        regen_p3=trial.suggest_float("regen_p3", 1.0, 2.2),
        regen_feed=trial.suggest_float("regen_feed", 0.2, 0.9),
        learned_omega=trial.suggest_categorical("learned_omega", [True, False]),
    )
    # 学習補正器は k31_two_term == 3.0 でだけ有効。補正器を使う試行では 3.0 に固定し、
    # 使わない試行でだけ二項則の係数を探す(補正器なしは corrector_active=False と記録)。
    if kw["learned_omega"]:
        kw["k31_two_term"] = 3.0
    else:
        kw["k31_two_term"] = trial.suggest_float("k31_two_term", 2.0, 4.0)
    return kw


def to_params(kw: dict, defaults: dict) -> dict:
    """設定を Optuna の分布のパラメータ名へ(enqueue 用)。"""
    full = dict(defaults, **kw)
    p = {k: full[k] for k in KNOBS if k not in ("k31_two_term", "window")}
    if not full["learned_omega"] or full["k31_two_term"] != 3.0:
        p["learned_omega"] = False
        p["k31_two_term"] = full["k31_two_term"]
    return p


def load_study(optuna, runner, seed):
    def constraints(trial):
        return trial.user_attrs.get("constraint", [1.0])

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                         n_startup_trials=8, constant_liar=True,
                                         constraints_func=constraints)
    STORE.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
                               sampler=sampler, direction="minimize", load_if_exists=True)


def trial_records(study, runner):
    recs = []
    for t in study.trials:
        recs.append(dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                         **{k: v for k, v in t.user_attrs.items()}))
    return recs


def _record(t, s):
    t.set_user_attr("kwargs", s["kwargs"])
    for k in ("adj", "mse", "cb", "ops", "ops_ratio", "ratio_mean", "ratio_of_means",
              "improved", "failures", "corrector_active", "max_resid_loaded", "max_sec_loaded"):
        t.set_user_attr(k, s.get(k))
    t.set_user_attr("n_rows", s["n"])
    t.set_user_attr("constraint", [s["ops_ratio"] - OPS_LIMIT if s.get("ops_ratio") else 1.0,
                                   float(s["failures"])])
    t.set_user_attr("feasible", feasible(s))


def phase_optuna(args, runner):
    """2 段の忠実度: dev 0〜7 で既定より ratio_of_means > prune 倍、または op 制約違反なら
    打ち切り(PRUNED、TPE は完了試行より悪い扱い)。残りだけ dev 0〜15 で完了させる。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    idxs = list(parse_range(args.indices))
    low = idxs[:8]
    runner.evaluate([{}], idxs)
    study = load_study(optuna, runner, args.seed)
    for t in study.trials:  # 中断で残った RUNNING は失敗扱いにして再開する
        if t.state.name == "RUNNING":
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if len(study.trials) == 0:
        # 既定設定と資源走査の上位を最初の試行として入れる(同じ行はキャッシュから再利用)
        study.enqueue_trial(to_params({}, runner.defaults))
        scan_path = RESULTS / "tune_phase2_scan.json"
        if scan_path.exists():
            scan = json.loads(scan_path.read_text())
            for s in [s for s in scan["configs"] if feasible(s) and s["kwargs"]
                      and "window" not in s["kwargs"]][:args.warm]:
                study.enqueue_trial(to_params(s["kwargs"], runner.defaults))
    run = wandb_run("optuna", args, runner)
    finished = sum(t.state.is_finished() for t in study.trials)
    while finished < args.trials:
        batch = [study.ask() for _ in range(min(args.batch, args.trials - finished))]
        kws = [suggest(t, runner.defaults) for t in batch]
        first = runner.evaluate(kws, low)
        keep = []
        for t, kw, s in zip(batch, kws, first):
            if feasible(s) and s["ratio_of_means"] <= args.prune:
                keep.append((t, kw))
                continue
            _record(t, s)
            t.report(s["adj"], step=len(low))
            study.tell(t, state=optuna.trial.TrialState.PRUNED)
            log_summary(run, t.number, s, "optuna/")
            print(f"trial {t.number} PRUNE@{len(low)} {short(s)}", flush=True)
        sums = runner.evaluate([kw for _, kw in keep], idxs)
        for (t, _), s in zip(keep, sums):
            _record(t, s)
            study.tell(t, s["adj"])
            log_summary(run, t.number, s, "optuna/")
            print(f"trial {t.number} {'OK ' if feasible(s) else 'REJ'} {short(s)}", flush=True)
        finished = sum(t.state.is_finished() for t in study.trials)
        save("trials", dict(candidate=str(runner.candidate), sha=runner.sha, study=STUDY_NAME,
                            storage=str(STUDY_DB), indices=idxs, low_fidelity=low, prune=args.prune,
                            ops_limit=OPS_LIMIT, default=runner.summary({}, idxs),
                            trials=trial_records(study, runner)))
    if run:
        run.finish()


def ranked_trials(runner, idxs):
    """dev 0〜15 で評価済みの全設定(Optuna・走査)を、制約を満たすものだけ調整後スコア順に。"""
    import optuna
    study = load_study(optuna, runner, 0)
    seen, out = set(), []
    for t in study.trials:
        if t.state.name != "COMPLETE" or not t.user_attrs.get("feasible"):
            continue
        key = runner.key(t.user_attrs["kwargs"])
        if key in seen or key == runner.key({}):
            continue
        seen.add(key)
        out.append((t.value, t.user_attrs["kwargs"], t.number))
    return sorted(out, key=lambda x: x[0])


def phase_promote(args, runner, stage_name, src_idxs, dst_idxs, top):
    run = wandb_run(stage_name, args, runner)
    if stage_name == "promote":
        cands = [kw for _, kw, _ in ranked_trials(runner, src_idxs)[:top]]
    else:
        prev = json.loads((RESULTS / "tune_phase2_promote.json").read_text())
        cands = [s["kwargs"] for s in prev["configs"] if feasible(s)][:top]
    sums = runner.evaluate([{}] + cands, dst_idxs)
    base, rest = sums[0], sorted(sums[1:], key=lambda s: s["adj"])
    print("default", short(base), flush=True)
    for step, s in enumerate(rest):
        log_summary(run, step, s, f"{stage_name}/")
        print(short(s), flush=True)
    save(stage_name, dict(candidate=str(runner.candidate), sha=runner.sha, indices=list(dst_idxs),
                          default=base, configs=rest))
    if run:
        run.finish()


def phase_report(args, runner):
    import optuna
    final = json.loads((RESULTS / "tune_phase2_final.json").read_text())
    best = [s for s in final["configs"] if feasible(s)][0]
    study = load_study(optuna, runner, 0)
    scan = json.loads((RESULTS / "tune_phase2_scan.json").read_text())
    promote = json.loads((RESULTS / "tune_phase2_promote.json").read_text())
    report = dict(
        candidate=str(runner.candidate), sha=runner.sha, dataset=DEV, ops_limit=OPS_LIMIT,
        recommended_kwargs=best["kwargs"], recommended=best, default_dev64=final["default"],
        final_dev64=final["configs"], promote_dev32=promote["configs"],
        scan_dev8=scan["configs"], optuna_dev16=trial_records(study, runner),
        optuna_v1_aborted=trial_records(optuna.load_study(
            study_name="tune_phase2_v1", storage=f"sqlite:///{STUDY_DB}"), runner),
    )
    path = save("report", report)
    print(path)
    for s in [final["default"]] + final["configs"]:
        print(short(s), "CI", s.get("ratio_of_means_ci95"), s.get("ratio_mean_ci95"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("baseline", "scan", "optuna", "promote", "final", "report"))
    p.add_argument("--candidate", type=Path, default=STORE / "estimator_f13be0e5.py")
    p.add_argument("--indices", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--prune", type=float, default=1.02,
                   help="dev 0〜7 の平均の比(既定比)がこれを超えた試行は 0〜15 へ進めない")
    args = p.parse_args()
    if args.workers * args.threads > 16:
        raise SystemExit("この調整に割り当てた CPU は 16 スレッドまで")
    runner = Runner(args.candidate, args.workers, args.threads)
    default_idx = dict(baseline="0-63", scan="0-7", optuna="0-15", promote="0-31", final="0-63")
    args.indices = args.indices or default_idx.get(args.phase, "0-63")
    try:
        if args.phase == "baseline":
            phase_baseline(args, runner)
        elif args.phase == "scan":
            phase_scan(args, runner)
        elif args.phase == "optuna":
            phase_optuna(args, runner)
        elif args.phase == "promote":
            phase_promote(args, runner, "promote", parse_range("0-15"), parse_range(args.indices),
                          args.top or 8)
        elif args.phase == "final":
            phase_promote(args, runner, "final", parse_range("0-31"), parse_range(args.indices),
                          args.top or 3)
        else:
            phase_report(args, runner)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
