"""Offline measured, uncalibrated SA/GA/hill search and fixed-config batches.

Default selection uses mini0..3 and the original directional-k4 search space.
Compute search: --space results/compute_search_space.json --indices 0 1 2 3 4 5 6 7
               --methods sa ga hill --wandb-offline --output results/compute_search.json
Fixed-config confirmation: --evaluate-configs configs.json --split validation.
Local schema-3.0/prepared-Arrow confirmation also takes --dataset-source PATH,
--dataset-split NAME and explicit --indices. No final dataset is selected implicitly.
Existing explicit output is a source/space/dataset/indices-validated evaluation
cache, not an exact mid-algorithm RNG resume. Without --output, a new report is
created; existing checkpoints are never implicitly overwritten.
Parallel residual is diagnostic only; run a separate single-thread preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

from eval_mini import DEFAULT_DATASET_SOURCE, DEFAULT_DATASET_SPLIT, Evaluator, ROOT, dataset_identity

SPACE_PATH = ROOT / "results/directional_search_space.json"
BUDGET = 2**41
WALL_LIMIT = 120.0
WORKERS, THREADS = 8, 4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity(candidate, space_path=SPACE_PATH, space=None):
    paths = (candidate, ROOT / "research/eval_mini.py", Path(__file__).resolve(), space_path)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    identity = dict(sources=sources, search_space=space)
    return hashlib.sha256(canonical(identity).encode()).hexdigest(), sources


def safe_json(value):
    """Keep failed measurements explicit without writing nonstandard NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def nonfinite_paths(value, prefix="metrics"):
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        return [path for key, item in value.items()
                for path in nonfinite_paths(item, f"{prefix}.{key}")]
    if isinstance(value, (list, tuple)):
        return [path for i, item in enumerate(value)
                for path in nonfinite_paths(item, f"{prefix}[{i}]")]
    return []


def normalize_config(config, space):
    allowed = set(space["grid"]) | set(space["fixed_kwargs"])
    if unknown := set(config) - allowed:
        raise ValueError(f"Unknown config knobs: {sorted(unknown)}")
    kwargs = {**space["fixed_kwargs"], **space["initial"], **config}
    if kwargs["calib"] != 0.0:
        raise ValueError("Fitted or nonzero calibration is not allowed")
    if type(kwargs["directional_k4"]) is not bool:
        raise ValueError("directional_k4 must be a JSON boolean")
    for name, values in space["grid"].items():
        if kwargs[name] not in values:
            raise ValueError(f"{name}={kwargs[name]!r} is outside the search grid")
        kwargs[name] = values[values.index(kwargs[name])]
    return kwargs


class Objective:
    """One bounded 8-process x 4-thread evaluator shared by all algorithms."""

    def __init__(self, candidate, indices, output, space, run_config, fresh=False, wandb_run=None,
                 dataset_source=None, dataset_split=None, space_path=SPACE_PATH):
        self.candidate, self.indices, self.output = candidate, indices, output
        self.space, self.wandb_run = space, wandb_run
        self.space_path = Path(space_path).resolve()
        self.dataset = dataset_identity(dataset_source, dataset_split)
        # The original space fixes True and keeps False as a measured comparator.
        # When mode is a gene, both modes participate in every search stage.
        self.directional_only = ("directional_k4" not in space["grid"]
                                 and space["fixed_kwargs"].get("directional_k4") is True)
        self.source_hash, sources = source_identity(candidate, self.space_path, space)
        self.cache, self.events, self.runs = {}, [], []
        self.counts = Counter(dict.fromkeys((
            "evaluations", "feasible_evaluations", "infeasible_evaluations", "cache_hits",
            "sa_accepted", "sa_uphill_accepted", "ga_crossovers", "ga_mutated_genes",
            "ga_elites_retained", "hill_improvements",
        ), 0))
        self.ev = None
        self.metadata = dict(schema_version=2, source_hash=self.source_hash, sources=sources,
                             search_space=space, space_path=str(self.space_path), dataset=self.dataset,
                             indices=indices, workers=WORKERS, threads_per_worker=THREADS,
                             flop_budget=BUDGET, wall_limit_s=WALL_LIMIT,
                             objective="mean(mse * max(0.1, flops / flop_budget))",
                             calibration="fixed 0.0; no fitted calibration",
                             parallel_residual="recorded, not a feasibility threshold")
        if output.exists() and not fresh:
            previous = json.loads(output.read_text())
            meta = previous.get("metadata", {})
            valid = (meta.get("schema_version") == 2
                     and meta.get("source_hash") == self.source_hash
                     and meta.get("dataset") == self.dataset
                     and meta.get("search_space") == self.space
                     and meta.get("indices") == indices)
            if valid:
                for record in previous.get("evaluations", []):
                    kwargs = normalize_config(record["kwargs"], space)
                    expected = self.key(kwargs)
                    if (record.get("source_hash") != self.source_hash
                            or record.get("dataset") != self.dataset
                            or record.get("indices") != indices or record.get("key") != expected):
                        raise ValueError("Invalid cache record identity; use --fresh to discard")
                    self.cache[expected] = record
                self.events = previous.get("events", [])
                self.runs = previous.get("runs", [])
                self.counts.update(previous.get("counts", {}))
            else:
                self.events.append(dict(event="cache_invalidated", previous_source=meta.get("source_hash"),
                                        previous_indices=meta.get("indices"),
                                        previous_dataset=meta.get("dataset")))
        self.run_id = len(self.runs)
        self.runs.append(dict(id=self.run_id, config=run_config, status="running", started=time.time()))
        self.save()

    def key(self, kwargs):
        payload = dict(source_hash=self.source_hash, dataset=self.dataset, search_space=self.space,
                       indices=self.indices, kwargs=kwargs)
        return hashlib.sha256(canonical(payload).encode()).hexdigest()

    def save(self):
        best = self.ranked()
        document = dict(metadata=self.metadata, runs=self.runs, counts=dict(self.counts),
                        best=best[0] if best else None, evaluations=list(self.cache.values()),
                        events=self.events)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + ".tmp")
        temporary.write_text(json.dumps(safe_json(document), indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output)

    def event(self, event, **payload):
        self.counts[event] += 1
        self.events.append(dict(event=event, run=self.run_id, time=time.time(), **payload))
        self.save()

    def ranked(self, directional_only=False, search_only=False):
        rows = (row for row in self.cache.values() if row["feasible"]
                and (not directional_only or row["kwargs"]["directional_k4"])
                and (not search_only or all(row["kwargs"][name] == value
                                            for name, value in self.space["fixed_kwargs"].items())))
        return sorted(rows, key=lambda row: row["metrics"]["adj"])

    def __call__(self, config, stage):
        kwargs = normalize_config(config, self.space)
        if source_identity(self.candidate, self.space_path, self.space)[0] != self.source_hash:
            self.event("source_changed", stage=stage)
            raise RuntimeError("Candidate/evaluator/search/space source changed during search; restart safely")
        if dataset_identity(self.dataset["source"], self.dataset["split"]) != self.dataset:
            self.event("dataset_changed", stage=stage)
            raise RuntimeError("Dataset metadata changed during search; restart safely")
        key = self.key(kwargs)
        if key in self.cache:
            self.counts["cache_hits"] += 1
            row = self.cache[key]
            return row["metrics"]["adj"] if row["feasible"] else math.inf
        if self.ev is None:
            self.ev = Evaluator(self.indices, workers=WORKERS, threads=THREADS,
                                dataset_source=self.dataset["source"], dataset_split=self.dataset["split"])
        started = time.monotonic()
        metrics, reasons = None, []
        try:
            metrics = self.ev.score(str(self.candidate), **kwargs)
            reasons.extend(f"nonfinite:{path}" for path in nonfinite_paths(metrics))
            rows = metrics["per"]
            if [row["idx"] for row in rows] != self.indices:
                reasons.append("per_mlp_indices_mismatch")
            for row in rows:
                failures = []
                if row["flops"] > BUDGET:
                    failures.append("flop_budget_exceeded")
                if row["sec"] > WALL_LIMIT:
                    failures.append("wall_limit_exceeded")
                if nonfinite_paths(row):
                    failures.append("nonfinite_output_or_measurement")
                row["adj"] = row["mse"] * max(0.1, row["flops"] / BUDGET)
                row["feasible"] = not failures
                row["status"] = "feasible" if not failures else "infeasible"
                row["reasons"] = failures
                reasons.extend(f"idx{row['idx']}:{failure}" for failure in failures)
            # These are the real evaluator's uncalibrated scores, not a surrogate.
            if not reasons and (metrics["mse"] < 0 or metrics["adj"] < 0):
                reasons.append("negative_mse_or_adjusted_metric")
        except Exception as exc:
            reasons.append(f"exception:{type(exc).__name__}:{exc}")
        feasible = not reasons
        row = dict(key=key, source_hash=self.source_hash, dataset=self.dataset,
                   indices=self.indices, kwargs=kwargs,
                   stage=stage, run=self.run_id, feasible=feasible,
                   status="feasible" if feasible else "infeasible", reasons=reasons,
                   elapsed_s=time.monotonic() - started, metrics=safe_json(metrics),
                   per_mlp=(safe_json(metrics["per"]) if metrics is not None else
                            [dict(idx=idx, status="unavailable", feasible=False,
                                  reasons=["Evaluator.score raised; no numeric result returned"])
                             for idx in self.indices]))
        self.cache[key] = row
        self.counts["evaluations"] += 1
        self.counts["feasible_evaluations" if feasible else "infeasible_evaluations"] += 1
        self.event("evaluation", key=key, stage=stage, feasible=feasible,
                   adj=metrics["adj"] if metrics is not None else None, reasons=reasons)
        if self.wandb_run is not None:
            log = {"evaluations": self.counts["evaluations"], "feasible": int(feasible)}
            if metrics is not None:
                log.update({name: value for name, value in metrics.items()
                            if name != "per" and isinstance(value, (int, float)) and math.isfinite(value)})
            self.wandb_run.log(log)
        print(f"{stage}: {row['status']} adj={metrics['adj'] if metrics else None} "
              f"evaluations={self.counts['evaluations']} {kwargs}", flush=True)
        return metrics["adj"] if feasible else math.inf

    def close(self):
        if self.ev is not None:
            self.ev.pool.shutdown(wait=True, cancel_futures=True)


def neighbors(config, grid):
    for name, values in grid.items():
        index = values.index(config[name])
        for offset in (-1, 1):
            other = index + offset
            if 0 <= other < len(values):
                yield {**config, name: values[other]}


def mutate(config, grid, rng, rate=None):
    """At least one genuinely changed gene; return actual changed knob names."""
    names = [name for name in grid if rate is not None and rng.random() < rate]
    if not names:
        names = [rng.choice(list(grid))]
    result = dict(config)
    for name in names:
        result[name] = rng.choice([value for value in grid[name] if value != config[name]])
    return result, names


def anneal(objective, start, grid, rng, steps, t0, t1):
    current, score = dict(start), objective(start, "sa_start")
    best, best_score = current, score
    for step in range(steps):
        temperature = t0 * (t1 / t0) ** (step / max(1, steps - 1))
        proposal, genes = mutate(current, grid, rng)
        proposal_score = objective(proposal, "sa")
        uphill = math.isfinite(score) and proposal_score > score
        if not math.isfinite(proposal_score):
            probability = 0.0
        elif proposal_score <= score or not math.isfinite(score):
            probability = 1.0
        else:
            delta = math.log(max(proposal_score, sys.float_info.min)) - math.log(max(score, sys.float_info.min))
            probability = math.exp(-delta / temperature)
        draw = rng.random()
        accepted = draw < probability
        previous_score = score
        if accepted:
            current, score = proposal, proposal_score
            objective.counts["sa_accepted"] += 1
            if uphill:
                objective.counts["sa_uphill_accepted"] += 1
        if score < best_score:
            best, best_score = dict(current), score
        objective.event("sa_step", step=step, temperature=temperature, genes=genes,
                        proposal=proposal, previous_score=previous_score, proposal_score=proposal_score,
                        acceptance_probability=probability, draw=draw, accepted=accepted,
                        uphill=uphill, current=current, score=score, best_score=best_score)
    return best


def genetic(objective, start, grid, rng, generations, population_size, mutation_rate):
    population = [dict(start)]
    # Seed from measured finalists in every searched mode, then explore.
    for row in objective.ranked(directional_only=objective.directional_only, search_only=True):
        if row["kwargs"] not in population and len(population) < population_size:
            population.append(row["kwargs"])
    while len(population) < population_size:
        config = {**start, **{name: rng.choice(values) for name, values in grid.items()}}
        if config not in population:
            population.append(config)
    scores = [objective(config, "ga_initial") for config in population]
    objective.event("ga_initial_population", population=population, scores=scores)
    for generation in range(generations):
        order = sorted(range(population_size), key=lambda index: scores[index])
        elite_count = max(1, population_size // 4)
        next_population = [dict(population[index]) for index in order[:elite_count]]
        next_scores = [scores[index] for index in order[:elite_count]]

        def tournament():
            return min(rng.sample(range(population_size), min(3, population_size)),
                       key=lambda index: scores[index])

        while len(next_population) < population_size:
            first = tournament()
            remaining = [index for index in range(population_size) if index != first]
            second = min(rng.sample(remaining, min(3, len(remaining))), key=lambda index: scores[index])
            # Uniform crossover with genes from both parents, followed by mutation.
            mask = [rng.random() < 0.5 for _ in grid]
            if all(mask) or not any(mask):
                mask[rng.randrange(len(mask))] = not mask[0]
            child = dict(population[first])
            from_second = []
            for (name, _), use_second in zip(grid.items(), mask):
                if use_second:
                    child[name] = population[second][name]
                    from_second.append(name)
            crossed = dict(child)
            child, mutated = mutate(child, grid, rng, mutation_rate)
            objective.counts["ga_crossovers"] += 1
            objective.counts["ga_mutated_genes"] += len(mutated)
            objective.event("ga_offspring", generation=generation, parents=[population[first], population[second]],
                            from_second=from_second, crossed=crossed, mutated_genes=mutated, child=child)
            next_population.append(child)
            next_scores.append(objective(child, "ga"))
        population, scores = next_population, next_scores
        objective.counts["ga_elites_retained"] += elite_count
        objective.event("ga_generation", generation=generation, elite_count=elite_count,
                        population=population, scores=scores, best_score=min(scores))
    return population[min(range(population_size), key=lambda index: scores[index])]


def hill_climb(objective, start, grid, steps):
    current, score = dict(start), objective(start, "hill_start")
    for step in range(steps):
        proposals = list(neighbors(current, grid))
        scores = [objective(config, "hill") for config in proposals]
        best_index = min(range(len(proposals)), key=lambda index: scores[index])
        improved = scores[best_index] < score
        if improved:
            current, score = proposals[best_index], scores[best_index]
            objective.counts["hill_improvements"] += 1
        objective.event("hill_step", step=step, neighbors=len(proposals), improved=improved,
                        current=current, score=score, neighbor_scores=scores)
        if not improved:
            break
    return current


def load_configs(path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "configs" in data:
            data = data["configs"]
        elif "evaluations" in data:
            data = [row["kwargs"] for row in data["evaluations"]]
        elif "kwargs" in data:
            data = [data["kwargs"]]
        else:
            data = [data]
    if not isinstance(data, list) or not data or not all(isinstance(row, dict) for row in data):
        raise ValueError("--evaluate-configs needs a nonempty config list, {configs: [...]}, or search report")
    return [row.get("kwargs", row) for row in data]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", "--stages", nargs="+", choices=("sa", "ga", "hill"), default=["sa", "ga", "hill"])
    parser.add_argument("--seed", type=int, default=260515)
    parser.add_argument("--sa-steps", type=int, default=24)
    parser.add_argument("--temperature-start", type=float, default=0.25)
    parser.add_argument("--temperature-end", type=float, default=0.02)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--hill-steps", type=int, default=3)
    parser.add_argument("--split", default="fast4", help="Named index set in the selected search space")
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--space", type=Path, default=SPACE_PATH)
    parser.add_argument("--dataset-source", default=DEFAULT_DATASET_SOURCE, help="Whestbench local directory or Hub repo")
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT, help="Dataset split, distinct from --split index sets")
    parser.add_argument("--candidate", type=Path, default=ROOT / "estimator.py")
    parser.add_argument("--output", type=Path, help="Checkpoint/report path; omitted creates a new uniquely named report")
    parser.add_argument("--evaluate-configs", type=Path, help="Fixed-config batch; skips all search algorithms")
    parser.add_argument("--fresh", action="store_true", help="Discard any cache in the output file")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default="directional-k4-search")
    args = parser.parse_args()
    if min(args.sa_steps, args.generations, args.hill_steps) < 0 or args.population < 2:
        parser.error("Step counts must be nonnegative and population must be >= 2")
    if not (0 < args.temperature_start < math.inf and 0 < args.temperature_end < math.inf):
        parser.error("SA temperatures must be finite and positive")
    if not 0 <= args.mutation_rate <= 1:
        parser.error("Mutation rate must lie in [0, 1]")
    args.space = args.space.resolve()
    space = json.loads(args.space.read_text())
    if args.indices is None and args.split not in space["splits"]:
        parser.error(f"Unknown --split {args.split!r}; choose from {sorted(space['splits'])} or use --indices")
    dataset = dataset_identity(args.dataset_source, args.dataset_split)
    is_mini = dataset["source"] == DEFAULT_DATASET_SOURCE and dataset["split"] == DEFAULT_DATASET_SPLIT
    if not is_mini and args.indices is None:
        parser.error("Non-mini dataset confirmation requires explicit --indices")
    indices = args.indices if args.indices is not None else space["splits"][args.split]
    if not indices or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
        parser.error("Indices must be nonempty, unique nonnegative integers")
    if is_mini and any(index >= 100 for index in indices):
        parser.error("Mini indices must lie in 0..99")
    if args.evaluate_configs is None and (not is_mini or any(index >= 12 for index in indices)):
        parser.error("Search selection is restricted to mini0..11; use --evaluate-configs for confirmation")
    if args.output is None:
        args.output = ROOT / "results" / f"{args.space.stem}_{time.time_ns()}.json"
    configs = ([normalize_config(config, space) for config in load_configs(args.evaluate_configs)]
               if args.evaluate_configs else None)
    run_config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    wandb_run = None
    if args.wandb_offline:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, mode="offline", config=run_config)
    objective = None
    try:
        objective = Objective(args.candidate.resolve(), indices, args.output.resolve(), space,
                              run_config, args.fresh, wandb_run, args.dataset_source,
                              args.dataset_split, args.space)
        rng = random.Random(args.seed)
        if configs is not None:
            for config in configs:
                objective(config, "confirmation")
            objective.event("batch_complete", configs=len(configs))
        else:
            start = normalize_config({}, space)
            if "baseline" in space:
                objective({**start, **space["baseline"]}, "baseline")
            objective(start, "initial")
            for method in args.methods:
                ranked = objective.ranked(directional_only=objective.directional_only, search_only=True)
                if ranked:
                    start = dict(ranked[0]["kwargs"])
                objective.event("stage_start", method=method, start=start)
                if method == "sa":
                    start = anneal(objective, start, space["grid"], rng, args.sa_steps,
                                   args.temperature_start, args.temperature_end)
                elif method == "ga":
                    start = genetic(objective, start, space["grid"], rng, args.generations,
                                    args.population, args.mutation_rate)
                else:
                    start = hill_climb(objective, start, space["grid"], args.hill_steps)
                objective.event("stage_complete", method=method, result=start)
        objective.runs[objective.run_id]["status"] = "complete"
        objective.runs[objective.run_id]["finished"] = time.time()
        objective.save()
    except BaseException as exc:
        if objective is not None:
            objective.runs[objective.run_id].update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                   error=f"{type(exc).__name__}: {exc}")
            objective.save()
        raise
    finally:
        if objective is not None:
            objective.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
