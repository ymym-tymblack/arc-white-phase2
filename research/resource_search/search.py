"""固定係数・固定補正器で資源だけを destroy/repair SA 探索する研究用 CLI。

run: 4網でSA、最大8候補を16網へ、上位2候補を64網へ昇格する。
既存64網結果はソース・データ・予算の一致を確認して再利用する。
行キャッシュは逐次 fsync、探索状態と最終設定は atomic replace で保存する。
CPU解放後のみ --cpu-released を付ける。holdout は読み込まない。
例: uv run --with wandb python research/resource_search/search.py run \
  --source research/candidates/adam1_epoch004.py --cpu-released
smoke は標準ライブラリだけで実行でき、データも目的関数も評価しない。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import eval_mini

BUDGET = 2**41
DATASET = "/kaggle/ext/full_dev64"
COEFFICIENTS = ("k31_two_term", "regen_p0", "regen_p1", "regen_p2", "regen_p3",
                "regen_feed", "regen_use31")
GRID = dict(age_old=(4, 5, 6, 7, 8), rank_old=(128, 192, 256, 320),
            window=(16,), iik_rank=(2, 4, 6, 8), k3leg_rank=(2, 4, 6, 8),
            kmax=(3,), strassen_levels=(2, 3, 4), strassen_levels_old=(2, 3, 4))
# 再学習時のtorch鎖と一致させるため、window/kmaxは監査済みの既定値に固定する。
VERSION = 2


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(encode(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_audit(path):
    """ASTのみで既定値と固定学習パラメータを読む(NumPyをimportしない)。"""
    data = path.read_bytes()
    tree = ast.parse(data)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Estimator")
    attrs = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        attrs[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    defaults = {key: attrs[key] for key in GRID if key in attrs}
    if set(COEFFICIENTS) - set(attrs):
        raise ValueError("Source does not expose all seven learned coefficients")
    blobs = [ast.dump(node, include_attributes=False) for node in tree.body
             if isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id.startswith("_OMEGA_CORRECTOR")
                 for target in node.targets)]
    if not blobs:
        raise ValueError("Source has no fixed corrector blob")
    fixed = {key: value for key, value in attrs.items() if key not in GRID}
    return dict(source_sha256=digest_bytes(data), defaults=defaults,
                coefficients={key: attrs[key] for key in COEFFICIENTS},
                corrector_ast_sha256=digest_bytes(encode(blobs).encode()),
                fixed_class_attributes=fixed,
                fixed_parameter_sha256=digest_bytes(encode(dict(attrs=fixed, blobs=blobs)).encode()))


def canon(kwargs, defaults):
    if set(kwargs) - set(defaults):
        raise ValueError("Only supported resource knobs may change")
    return {key: value for key, value in sorted(kwargs.items()) if value != defaults[key]}


def repair(config, defaults):
    out = dict(defaults, **config)
    for key, value in out.items():
        if value not in GRID[key]:
            raise ValueError(f"Resource outside bounded grid: {key}={value}")
    if out["age_old"] > out["window"]:
        out["age_old"] = max(v for v in GRID["age_old"] if v <= out["window"])
    return canon(out, defaults)


def propose(rng, current, defaults, seen, step):
    """2〜3個の資源を壊し、範囲・年齢/窓整合性を修復する。全直積は作らない。"""
    tradeoffs = [dict(age_old=4, rank_old=320), dict(age_old=5, rank_old=320),
                dict(age_old=7, rank_old=192), dict(age_old=8, rank_old=128)]
    if step < len(tradeoffs):
        out = repair(tradeoffs[step], defaults)
        if encode(out) not in seen:
            return out, "coupled_age_rank"
    for _ in range(1000):
        out = dict(defaults, **current)
        keys = rng.sample(sorted(key for key in defaults if len(GRID[key]) > 1),
                          rng.choice((2, 3)))
        for key in keys:
            out[key] = rng.choice([value for value in GRID[key] if value != out[key]])
        out = repair(out, defaults)
        if encode(out) not in seen:
            return out, "destroy_repair"
    raise RuntimeError("No unseen destroy/repair proposal after 1000 attempts")


def temperature(step, budget, start=0.08, end=0.005):
    return start * (end / start) ** (step / max(1, budget - 1))


def acceptance(current_score, proposed_score, temp):
    if proposed_score is None:
        return 0.0, None
    delta = math.log(proposed_score) - math.log(current_score)
    return (1.0 if delta <= 0 else math.exp(-delta / temp)), delta


def worker(job):
    """初期化でBLAS環境変数を設定した後にだけNumPy/flopscopeをimportする。"""
    path, expected_sha, kwargs, idx, coefficients = job
    ctx = None
    started = time.monotonic()
    phase = "load"
    try:
        import numpy as np
        import flopscope as flops
        from whestbench import SetupContext
        if digest_bytes(Path(path).read_bytes()) != expected_sha:
            raise ValueError("Immutable source digest changed")
        mlp, truth = eval_mini._load_mlp(idx)
        spec = importlib.util.spec_from_file_location("resource_candidate", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        est = module.Estimator(**kwargs)
        if any(getattr(est, key) != value for key, value in coefficients.items()):
            raise ValueError("Fixed learned coefficient changed")
        phase = "setup"
        setup_start = time.monotonic()
        est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=BUDGET,
                               api_version="1.0", submission_dir=str(ROOT), seed=0))
        setup_sec = time.monotonic() - setup_start
        phase = "predict"
        predict_start = time.monotonic()
        with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120.0, quiet=True) as ctx:
            pred = est.predict(mlp, BUDGET)
        sec = time.monotonic() - predict_start
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction shape/values: {pred.shape}")
        mse = float(((pred[-1] - truth[-1]) ** 2).mean())
        return dict(idx=idx, ok=True, mse=mse, flops=int(ctx.flops_used),
                    adj=mse * max(0.1, ctx.flops_used / BUDGET),
                    ops=len(ctx._op_log), sec=sec, setup_sec=setup_sec,
                    resid_loaded=float(ctx.residual_wall_time_s))
    except Exception as error:
        return dict(idx=idx, ok=False, phase=phase, error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(), elapsed=time.monotonic() - started,
                    flops=int(ctx.flops_used) if ctx is not None else None,
                    ops=len(ctx._op_log) if ctx is not None else None)


class Runner:
    def __init__(self, args, audit, store, identity):
        self.args, self.audit, self.store, self.identity = args, audit, store, identity
        self.defaults = audit["defaults"]
        self.rows = {}
        self.pool = None
        self.cache = store / "rows.jsonl"
        if self.cache.exists():
            for line in self.cache.read_text().splitlines():
                record = json.loads(line)
                if record["identity"] != identity:
                    raise ValueError("Row cache identity differs")
                key = (record["kwargs_key"], record["row"]["idx"])
                if key in self.rows:
                    raise ValueError(f"Duplicate cache row: {key}")
                self.rows[key] = record["row"]

    def save_row(self, kwargs, row):
        key = encode(canon(kwargs, self.defaults))
        record = dict(identity=self.identity, kwargs_key=key, row=row, recorded_at=time.time())
        append(self.cache, record)
        self.rows[(key, row["idx"])] = row

    def seed_baseline(self, path):
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        identity = record["identity"]
        if (identity["dataset"] != self.identity["dataset"] or
                identity["wall_limit"] != 120.0 or identity["flop_budget"] != BUDGET):
            raise ValueError("Stored baseline dataset/budget identity differs")
        matches = [run for run in record["runs"].values()
                   if run["sha256"] == self.audit["source_sha256"] and run.get("kwargs") == {}]
        if not matches:
            return False
        run = matches[0]
        if run["failures"] or sorted(row["idx"] for row in run["per"]) != list(range(64)):
            raise ValueError("Stored baseline is incomplete/failed")
        for source_row in run["per"]:
            if source_row["sec"] > 120.0 or source_row["flops"] > BUDGET:
                raise ValueError("Stored baseline violates prediction limits")
            if ("{}", source_row["idx"]) not in self.rows:
                row = dict(source_row, ok=True, ops=None,
                           adj=source_row["mse"] * max(0.1, source_row["flops"] / BUDGET),
                           resid_loaded=source_row.get("resid"), reused_from=str(path.resolve()),
                           ops_note="Historical paired evaluator did not record operation counts")
                self.save_row({}, row)
        return True

    def evaluate(self, configs, count):
        todo = [(canon(config, self.defaults), idx) for config in configs for idx in range(count)
                if (encode(canon(config, self.defaults)), idx) not in self.rows]
        todo = list({(encode(config), idx): (config, idx) for config, idx in todo}.values())
        if todo:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.args.workers,
                    mp_context=mp.get_context("spawn"), initializer=eval_mini._init,
                    initargs=(self.args.threads, self.identity["dataset"]))
            jobs = {self.pool.submit(worker, (str(self.store / "source.py"),
                    self.audit["source_sha256"], config, idx, self.audit["coefficients"])):
                    (config, idx) for config, idx in todo}
            for future in as_completed(jobs):
                config, idx = jobs[future]
                try:
                    row = future.result()
                except Exception as error:
                    row = dict(idx=idx, ok=False, phase="process",
                               error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                self.save_row(config, row)
                print(encode(dict(event="row", kwargs=config, idx=idx, ok=row["ok"],
                                  adj=row.get("adj"), error=row.get("error"))), flush=True)
        return [self.summary(config, count) for config in configs]

    def summary(self, kwargs, count):
        rows = [self.rows[(encode(canon(kwargs, self.defaults)), idx)] for idx in range(count)]
        failed = [row for row in rows if not row["ok"]]
        good = [row for row in rows if row["ok"]]
        return dict(kwargs=canon(kwargs, self.defaults), n=count, failures=len(failed),
                    adj=None if failed else sum(row["adj"] for row in rows) / count,
                    mse=None if failed else sum(row["mse"] for row in rows) / count,
                    cb=None if failed else sum(row["flops"] for row in rows) / count / BUDGET,
                    max_ops=max((row["ops"] for row in rows if row.get("ops") is not None), default=None),
                    max_resid_loaded=max((row["resid_loaded"] for row in good
                                          if row.get("resid_loaded") is not None), default=None),
                    max_sec_loaded=max((row["sec"] for row in good), default=None),
                    per_adj=None if failed else [row["adj"] for row in rows])

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def paired(base, candidate, seed, draws=10000):
    rng = random.Random(seed)
    n = len(base)
    ratios, mean_ratios = [], []
    for _ in range(draws):
        sample = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(candidate[j] for j in sample) / sum(base[j] for j in sample))
        mean_ratios.append(sum(candidate[j] / base[j] for j in sample) / n)
    ratios.sort()
    mean_ratios.sort()
    return dict(ratio_of_means=sum(candidate) / sum(base),
                ratio_of_means_ci95=[ratios[int(draws * .025)], ratios[int(draws * .975)]],
                mean_paired_ratio=sum(a / b for a, b in zip(candidate, base)) / n,
                mean_paired_ratio_ci95=[mean_ratios[int(draws * .025)], mean_ratios[int(draws * .975)]],
                improved=sum(a < b for a, b in zip(candidate, base)), n=n,
                bootstrap_draws=draws, bootstrap_seed=seed)


def tuple_tree(value):
    return tuple(tuple_tree(item) for item in value) if isinstance(value, list) else value


def smoke(args):
    audit = source_audit(args.source)
    defaults = audit["defaults"]
    def sequence():
        rng = random.Random(args.seed)
        seen = {"{}"}
        out = []
        for step in range(args.proposals):
            proposal, kind = propose(rng, {}, defaults, seen, step)
            seen.add(encode(proposal))
            changed = sum(dict(defaults, **proposal)[key] != defaults[key] for key in defaults)
            assert changed in (2, 3), (proposal, changed)
            out.append((proposal, kind))
        return out
    assert sequence() == sequence()
    assert acceptance(1.0, 0.5, .05)[0] == 1.0
    assert acceptance(1.0, None, .05) == (0.0, None)
    assert 0.0 < acceptance(1.0, 1.1, .05)[0] < 1.0
    rng = random.Random(args.seed)
    saved = json.loads(encode(rng.getstate()))
    restored = random.Random()
    restored.setstate(tuple_tree(saved))
    assert rng.random() == restored.random()
    assert "numpy" not in sys.modules
    result = dict(passed=True, mode="objective_independent_smoke", proposals=sequence(),
                  source_audit=audit, no_dataset_loaded=True, numpy_imported=False)
    if args.out:
        atomic(args.out, result)
    print(json.dumps(result, indent=2))


def run(args):
    if not args.cpu_released:
        raise SystemExit("CPU release required; do not evaluate while parent profiles residual safety")
    if args.workers * args.threads > 30:
        raise SystemExit("Resource search must reserve 2 of 32 CPU threads")
    audit = source_audit(args.source)
    identity = dict(version=VERSION, source_sha256=audit["source_sha256"],
                    dataset=eval_mini.dataset_identity(DATASET, "full"), flop_budget=BUDGET,
                    wall_limit=120.0, execution=dict(workers=args.workers, threads=args.threads))
    parameters = dict(seed=args.seed, proposals=args.proposals, max_promotions=args.max_promotions,
                      screen_n=4, promote_n=16, final_n=64, final_top=2,
                      promotion_ratio=1.02, temperature_start=.08, temperature_end=.005)
    tag = digest_bytes(encode(dict(identity=identity, parameters=parameters)).encode())[:20]
    store = args.store / tag
    store.mkdir(parents=True, exist_ok=True)
    provenance = dict(identity=identity, parameters=parameters, source_audit=audit,
                      original_source=str(args.source.resolve()),
                      objective="mean network MSE * max(0.1, FLOPs / 2**41)",
                      energy="log(adjusted score)", fixed_theta=True,
                      evaluator="exact flopscope; no GPU surrogate",
                      torch_note="ChainConfig lacks window/Strassen parity; no surrogate used",
                      timing_note="Loaded residual diagnostic only; parent owns quiet final safety")
    if (store / "provenance.json").exists():
        if json.loads((store / "provenance.json").read_text()) != provenance:
            raise ValueError("Immutable search provenance differs")
    else:
        atomic(store / "provenance.json", provenance)
        (store / "source.py").write_bytes(args.source.read_bytes())
    if digest_bytes((store / "source.py").read_bytes()) != audit["source_sha256"]:
        raise ValueError("Immutable snapshot mismatch")
    if (store / "recommendation.json").exists():
        print((store / "recommendation.json").read_text())
        return
    runner = Runner(args, audit, store, identity)
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_DIR", "/kaggle/ext/wandb")
    import wandb
    wb = wandb.init(project="arc-fixed-resource-search", name=f"sa-{tag}", config=provenance)
    started = time.time()
    try:
        reused = runner.seed_baseline(args.baseline)
        baseline = runner.evaluate([{}], 64)[0]
        if baseline["failures"]:
            raise RuntimeError("Fixed-source baseline failed; rows persisted, no candidate selected")
        state_path = store / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = dict(step=0, current={}, current_score=runner.summary({}, 4)["adj"],
                         best={}, best_score=runner.summary({}, 4)["adj"], proposals=[],
                         rng_state=random.Random(args.seed).getstate())
            atomic(state_path, state)
        rng = random.Random()
        rng.setstate(tuple_tree(state["rng_state"]))
        while state["step"] < args.proposals:
            step = state["step"]
            seen = {"{}"} | {encode(record["kwargs"]) for record in state["proposals"]}
            proposal, kind = propose(rng, state["current"], audit["defaults"], seen, step)
            screen = runner.evaluate([proposal], 4)[0]
            temp = temperature(step, args.proposals)
            probability, delta = acceptance(state["current_score"], screen["adj"], temp)
            draw = rng.random()
            accepted = draw < probability
            record = dict(step=step, kwargs=proposal, kind=kind, screen=screen,
                          temperature=temp, delta_energy=delta, acceptance_probability=probability,
                          acceptance_draw=draw, accepted=accepted,
                          decision="failed" if screen["failures"] else ("accepted" if accepted else "rejected"))
            if accepted:
                state.update(current=proposal, current_score=screen["adj"])
            if screen["adj"] is not None and screen["adj"] < state["best_score"]:
                state.update(best=proposal, best_score=screen["adj"])
            prior_promotions = sum("promote" in previous for previous in state["proposals"])
            if (screen["adj"] is not None and prior_promotions < args.max_promotions and
                    (step < 4 or screen["adj"] <= state["best_score"] * 1.02)):
                record["promote"] = runner.evaluate([proposal], 16)[0]
            state["proposals"].append(record)
            state["step"] += 1
            state["rng_state"] = rng.getstate()
            atomic(state_path, state)
            wb.log(dict(step=step, screen_adj=screen["adj"], accepted=int(accepted),
                        failures=screen["failures"], best_screen_adj=state["best_score"],
                        temperature=temp, ops=screen["max_ops"]))
            print(encode(dict(event="proposal", **record)), flush=True)
        promoted = [record["promote"] for record in state["proposals"]
                    if "promote" in record and not record["promote"]["failures"]]
        promoted.sort(key=lambda record: record["adj"])
        finalists = runner.evaluate([record["kwargs"] for record in promoted[:2]], 64)
        for finalist in finalists:
            if not finalist["failures"]:
                finalist["paired"] = paired(baseline["per_adj"], finalist["per_adj"], args.seed)
        valid = sorted([record for record in finalists if not record["failures"]], key=lambda r: r["adj"])
        winner = valid[0] if valid and valid[0]["adj"] < baseline["adj"] else None
        result = dict(provenance=provenance, baseline_reused=reused, baseline=baseline,
                      proposals=state["proposals"], finalists=finalists,
                      outcome="improved" if winner else "no_improvement",
                      recommended_kwargs=winner["kwargs"] if winner else {}, winner=winner,
                      store=str(store.resolve()), seconds=time.time() - started,
                      final_holdout_used=False,
                      selection_note="Development-selected result; no claim of holdout improvement")
        atomic(store / "recommendation.json", result)
        if args.out:
            if args.out.exists():
                raise FileExistsError(f"Refusing to overwrite report: {args.out}")
            atomic(args.out, result)
        wb.summary.update(dict(outcome=result["outcome"], baseline_adj=baseline["adj"],
                               winner_adj=winner["adj"] if winner else baseline["adj"]))
        print(encode(dict(event="complete", store=str(store), outcome=result["outcome"],
                          kwargs=result["recommended_kwargs"])), flush=True)
    finally:
        runner.close()
        wb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--source", type=Path, default=ROOT / "research/candidates/adam1_epoch004.py")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/dev64_adam1_epoch004_pair.json")
    parser.add_argument("--store", type=Path, default=Path("/kaggle/ext/resource_search"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=2119)
    parser.add_argument("--proposals", type=int, default=24)
    parser.add_argument("--max-promotions", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--cpu-released", action="store_true")
    args = parser.parse_args()
    if args.proposals < 4 or args.max_promotions < 2 or args.workers < 1 or args.threads < 1:
        parser.error("Need >=4 proposals, >=2 promotions and positive worker/thread counts")
    (smoke if args.mode == "smoke" else run)(args)


if __name__ == "__main__":
    main()
