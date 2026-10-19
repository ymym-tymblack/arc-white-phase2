"""学習値・資源配分を固定し、Strassen・λ特徴・ペア係数packingを安全化する。"""
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


def _pack_pairs(text):
    """既知の３つの文だけをASTで確認し、不要なstack/concat/gatherを除く。"""
    cls = next(node for node in ast.parse(text).body
               if isinstance(node, ast.ClassDef) and node.name == "Estimator")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "predict")
    assignments = {}
    for node in ast.walk(method):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"c_head", "coefficients", "pair"}):
            name = node.targets[0].id
            if name in assignments:
                raise ValueError(f"Duplicate pair-packing assignment: {name}")
            assignments[name] = node
    head, pair = assignments["c_head"], assignments["pair"].value
    head_new = "fnp.stack(c[:kmax + 5])"
    args_new = ("fnp.stack((c_head, c2p, c2p), axis=1)",
                "fnp.stack((c_head, c_head, c2p), axis=1)")

    def same(node, expression):
        return ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body)

    if (not isinstance(pair, ast.Call) or not same(pair.func, "_pair_terms_batched")
            or len(pair.args) != 9 or pair.keywords):
        raise ValueError("Unexpected pair contraction")
    if ("coefficients" not in assignments and same(head.value, head_new)
            and all(same(node, expression) for node, expression in zip(pair.args[:2], args_new))):
        return text
    coefficient = assignments["coefficients"]
    if not (same(head.value, "fnp.concatenate([fnp.stack(c[:2]), c_tail[:kmax + 3]])")
            and same(coefficient.value, "fnp.stack((c_head, c2p), axis=1)")
            and same(pair.args[0], "coefficients[:, [0, 1, 1], :]")
            and same(pair.args[1], "coefficients[:, [0, 0, 1], :]")):
        raise ValueError("Pair packing differs from the audited implementation")
    uses = [node for node in ast.walk(method)
            if isinstance(node, ast.Name) and node.id == "coefficients"]
    if len(uses) != 3:
        raise ValueError("Intermediate coefficients have additional uses")
    if not (head.lineno == head.end_lineno and coefficient.lineno == coefficient.end_lineno
            and pair.args[0].lineno == pair.args[0].end_lineno
            == pair.args[1].lineno == pair.args[1].end_lineno
            and all(node.lineno > pair.args[1].end_lineno for node in pair.args[2:])):
        raise ValueError("Expected separate packing statements and argument rows")
    lines = text.splitlines(keepends=True)
    lines[head.lineno - 1] = " " * head.col_offset + f"c_head = {head_new}\n"
    lines[coefficient.lineno - 1] = ""
    indent = " " * pair.args[0].col_offset
    lines[pair.args[0].lineno - 1] = "".join(indent + expression + ",\n" for expression in args_new)
    return "".join(lines)


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
    args.out.write_text(_pack_pairs("".join(lines)))
    before, after = source_audit(args.source), source_audit(args.out)
    if before["fixed_parameter_sha256"] != after["fixed_parameter_sha256"]:
        raise ValueError("Learned/fixed parameters changed during implementation transfer")
    if before["defaults"] != after["defaults"]:
        raise ValueError("Resource configuration changed during implementation transfer")
    print(after["source_sha256"], flush=True)


if __name__ == "__main__":
    main()
