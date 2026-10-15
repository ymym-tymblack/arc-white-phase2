"""Offline directional-k4 search, with measured, uncalibrated Evaluator.score.

Default selection uses mini0..3; recheck finalists with --split development.
Example: --methods sa ga hill --sa-steps 24 --generations 3 --population 6
         --hill-steps 3 --indices 0 1 2 3 --output results/directional_search.json
Fixed-config confirmation: --evaluate-configs configs.json --split validation.
Existing output is a source/indices-validated evaluation cache, not an exact
mid-algorithm RNG resume. Repeated invocations append new algorithm trajectories.
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

from eval_mini import Evaluator, ROOT

SPACE_PATH = ROOT / "results/directional_search_space.json"
BUDGET = 2**41
WALL_LIMIT = 120.0
WORKERS, THREADS = 8, 4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity(candidate):
    paths = (candidate, ROOT / "research/eval_mini.py", Path(__file__).resolve(), SPACE_PATH)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(canonical(sources).encode()).hexdigest(), sources


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

    def __init__(self, candidate, indices, output, space, run_config, fresh=False, wandb_run=None):
        self.candidate, self.indices, self.output = candidate, indices, output
        self.space, self.wandb_run = space, wandb_run
        self.source_hash, sources = source_identity(candidate)
        self.cache, self.events, self.runs = {}, [], []
        self.counts = Counter(dict.fromkeys((
            "evaluations", "feasible_evaluations", "infeasible_evaluations", "cache_hits",
            "sa_accepted", "sa_uphill_accepted", "ga_crossovers", "ga_mutated_genes",
            "ga_elites_retained", "hill_improvements",
        ), 0))
        self.ev = None
        self.metadata = dict(schema_version=1, source_hash=self.source_hash, sources=sources,
                             indices=indices, workers=WORKERS, threads_per_worker=THREADS,
                             flop_budget=BUDGET, wall_limit_s=WALL_LIMIT,
                             objective="mean(mse * max(0.1, flops / flop_budget))",
                             calibration="fixed 0.0; no fitted calibration",
                             parallel_residual="recorded, not a feasibility threshold")
        if output.exists() and not fresh:
            previous = json.loads(output.read_text())
            meta = previous.get("metadata", {})
            valid = (meta.get("schema_version") == 1
                     and meta.get("source_hash") == self.source_hash
                     and meta.get("indices") == indices)
            if valid:
                for record in previous.get("evaluations", []):
                    kwargs = normalize_config(record["kwargs"], space)
                    expected = self.key(kwargs)
                    if (record.get("source_hash") != self.source_hash
                            or record.get("indices") != indices or record.get("key") != expected):
                        raise ValueError("Invalid cache record identity; use --fresh to discard")
                    self.cache[expected] = record
                self.events = previous.get("events", [])
                self.runs = previous.get("runs", [])
                self.counts.update(previous.get("counts", {}))
            else:
                self.events.append(dict(event="cache_invalidated", previous_source=meta.get("source_hash"),
                                        previous_indices=meta.get("indices")))
        self.run_id = len(self.runs)
        self.runs.append(dict(id=self.run_id, config=run_config, status="running", started=time.time()))
        self.save()

    def key(self, kwargs):
        payload = dict(source_hash=self.source_hash, indices=self.indices, kwargs=kwargs)
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

    def ranked(self, directional_only=False):
        rows = (row for row in self.cache.values() if row["feasible"]
                and (not directional_only or row["kwargs"]["directional_k4"]))
        return sorted(rows, key=lambda row: row["metrics"]["adj"])

    def __call__(self, config, stage):
        kwargs = normalize_config(config, self.space)
        if source_identity(self.candidate)[0] != self.source_hash:
            self.event("source_changed", stage=stage)
            raise RuntimeError("Candidate/evaluator/search source changed during search; restart safely")
        key = self.key(kwargs)
        if key in self.cache:
            self.counts["cache_hits"] += 1
            row = self.cache[key]
            return row["metrics"]["adj"] if row["feasible"] else math.inf
        if self.ev is None:
            self.ev = Evaluator(self.indices, workers=WORKERS, threads=THREADS)
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
                reasons.extend(f"mini{row['idx']}:{failure}" for failure in failures)
            # These are the real evaluator's uncalibrated scores, not a surrogate.
            if not reasons and (metrics["mse"] < 0 or metrics["adj"] < 0):
                reasons.append("negative_mse_or_adjusted_metric")
        except Exception as exc:
            reasons.append(f"exception:{type(exc).__name__}:{exc}")
        feasible = not reasons
        row = dict(key=key, source_hash=self.source_hash, indices=self.indices, kwargs=kwargs,
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
    # Seed with already-measured directional finalists, then explore distinct genomes.
    for row in objective.ranked(directional_only=True):
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
    parser.add_argument("--split", choices=("fast4", "development", "validation", "previous-development"), default="fast4")
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--candidate", type=Path, default=ROOT / "candidates/k3_factored.py")
    parser.add_argument("--output", type=Path, default=ROOT / "results/directional_search.json")
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
    space = json.loads(SPACE_PATH.read_text())
    indices = args.indices if args.indices is not None else space["splits"][args.split]
    if len(set(indices)) != len(indices) or any(index < 0 or index >= 100 for index in indices):
        parser.error("Indices must be unique mini indices in 0..99")
    if args.evaluate_configs is None and any(index >= 12 for index in indices):
        parser.error("Search selection is restricted to mini0..11; use --evaluate-configs for confirmation")
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
                              run_config, args.fresh, wandb_run)
        rng = random.Random(args.seed)
        if configs is not None:
            for config in configs:
                objective(config, "confirmation")
            objective.event("batch_complete", configs=len(configs))
        else:
            start = normalize_config({}, space)
            objective({**start, **space["baseline"]}, "baseline")
            objective(start, "initial")
            for method in args.methods:
                ranked = objective.ranked(directional_only=True)
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
