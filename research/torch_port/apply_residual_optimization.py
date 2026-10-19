"""学習済み係数を触らず、検証済みの作業領域・Strassen再構成・λ特徴の実装を移す。"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.resource_search.search import source_audit  # noqa: E402

TARGETS = ("_Strassen", "_omega_shift")


def implementations(text):
    nodes = {node.name: node for node in ast.parse(text).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in TARGETS}
    if set(nodes) != set(TARGETS):
        raise ValueError("Source must contain the expected two implementation blocks")
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path,
                        default=ROOT / "research/candidates/adam1_residual_optimized.py")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output already exists; never overwrite a frozen candidate")
    source, donor = args.source.read_text(), args.template.read_text()
    nodes, replacements = implementations(source), implementations(donor)
    lines = source.splitlines(keepends=True)
    for name, node in sorted(nodes.items(), key=lambda item: item[1].lineno, reverse=True):
        replacement = ast.get_source_segment(donor, replacements[name])
        lines[node.lineno - 1:node.end_lineno] = [replacement + "\n"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines))
    before, after = source_audit(args.source), source_audit(args.out)
    if before["fixed_parameter_sha256"] != after["fixed_parameter_sha256"]:
        raise ValueError("Learned/fixed parameters changed during implementation transfer")
    if before["defaults"] != after["defaults"]:
        raise ValueError("Resource configuration changed during implementation transfer")
    print(after["source_sha256"], flush=True)


if __name__ == "__main__":
    main()
