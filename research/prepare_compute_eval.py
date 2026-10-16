"""Pin an official full-split subset without sampling or evaluating predictions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

REPO = "aicrowd/arc-whestbench-public-2026"
REVISION = "aa99830fdc09fad15407b10e8e3459d3e18bba0a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--shards", type=int, default=9)
    args = parser.parse_args()
    if args.rows < 1 or args.shards < 1:
        parser.error("rows and shards must be positive")

    from datasets import Dataset, concatenate_datasets
    from huggingface_hub import HfApi, snapshot_download
    import pyarrow as pa
    import whestbench

    # Eight save workers inherit four native threads each; no input MC is run.
    pa.set_cpu_count(4)
    started = time.monotonic()
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        provenance = metadata.get("compute_subset", {})
        if (
            provenance.get("repo") != REPO
            or provenance.get("revision") != REVISION
            or provenance.get("rows") != args.rows
        ):
            raise ValueError(
                "Existing output is a different dataset; choose another output directory"
            )
        subset = whestbench.load_dataset(str(args.output), split="full")
        if len(subset) != args.rows:
            raise ValueError("Cached subset row count differs from provenance")
    else:
        files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        arrows = sorted(
            p for p in files if p.startswith("prepared/full/") and p.endswith(".arrow")
        )[: args.shards]
        if len(arrows) != args.shards:
            raise ValueError("Requested more prepared shards than the pinned repository provides")
        root = Path(
            snapshot_download(
                REPO,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=[*arrows, "metadata.json"],
                max_workers=8,
            )
        )
        metadata_bytes = (root / "metadata.json").read_bytes()
        source_metadata = json.loads(metadata_bytes)
        pieces = [Dataset.from_file(str(root / name)) for name in arrows]
        available = concatenate_datasets(pieces)
        if len(available) < args.rows:
            raise ValueError(f"Only {len(available)} rows in selected shards; increase --shards")
        subset = available.select(range(args.rows))
        seeds = [int(value) for value in subset["mlp_seed"]]
        mini = whestbench.load_dataset(REPO, revision=REVISION, split="mini")
        mini_seeds = {int(value) for value in mini["mlp_seed"]}
        if len(set(seeds)) != args.rows or set(seeds) & mini_seeds:
            raise ValueError("The proposed final subset has duplicate or mini-overlapping seeds")
        args.output.mkdir(parents=True, exist_ok=True)
        subset.save_to_disk(str(args.output / "prepared/full"), max_shard_size="500MB", num_proc=8)
        metadata = copy.deepcopy(source_metadata)
        metadata["splits"] = {"full": {**metadata["splits"]["full"], "n_mlps": args.rows}}
        metadata["default_split"] = "full"
        metadata["prepared_splits"] = {"full": {"path": "prepared/full", "format": "save_to_disk"}}
        metadata["compute_subset"] = dict(
            repo=REPO,
            revision=REVISION,
            split="full",
            rows=args.rows,
            source_indices=list(range(args.rows)),
            source_arrow_files=arrows,
            source_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            mlp_seeds=seeds,
            mini_seed_overlap=0,
            predictions_evaluated_during_preparation=False,
            sampling_performed=False,
        )
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(metadata_path)
        subset = whestbench.load_dataset(str(args.output), split="full")
        if len(subset) != args.rows:
            raise ValueError("Prepared subset did not load through the official loader")

    report = dict(
        source=str(args.output.resolve()),
        split="full",
        rows=len(subset),
        n_samples=metadata["n_samples"],
        width=metadata["width"],
        depth=metadata["depth"],
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        provenance=metadata["compute_subset"],
        wall_seconds=time.monotonic() - started,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("source", "rows", "n_samples", "metadata_sha256", "wall_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
