"""学習済みチェックポイントの値を estimator.py の写しへ書き込み、候補ファイルを作る。

クラス属性の既定値(k31_two_term・regen_p0..p3・regen_feed・regen_use31)、補正器の base64
(_OMEGA_CORRECTOR_B64 と同じ並び)を差し替え、補正器の適用条件も新しい値へそろえる。
資源設定はcheckpointから復元し、Strassen段数などは --resources で明示できる。

例: research/gpu_env.sh research/torch_port/export_candidate.py /kaggle/ext/torch_ckpt/adam1/best.pt \
        --source research/frozen_phase2/combined.py --out research/candidates/adam_fit.py
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import torch

RESOURCE_KEYS = {"age_old", "rank_old", "iik_rank", "k3leg_rank", "window", "kmax",
                 "strassen_levels", "strassen_levels_old"}


def sub_once(pattern, repl, text, flags=0):
    new, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise ValueError(f"pattern not found: {pattern}")
    return new


def _replace_attributes(text, attrs):
    tree = ast.parse(text)
    estimator = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Estimator")
    replacements, found = [], set()
    source_lines = text.splitlines(keepends=True)
    for node in estimator.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        selected = set(names) & set(attrs)
        if not selected:
            continue
        if selected != set(names) or found & selected:
            raise ValueError("Unexpected grouped or duplicate learned attribute assignment")
        found.update(selected)
        tail = source_lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode()
        replacement = "\n".join(f"    {name} = {attrs[name]!r}" for name in names) + tail
        replacements.append((node.lineno, node.end_lineno, replacement))
    if found != set(attrs):
        raise ValueError(f"Missing learned attributes: {set(attrs) - found}")
    lines = text.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1:end] = [replacement]
    text = "".join(lines)
    gates = [node.comparators[0] for node in ast.walk(ast.parse(text))
             if isinstance(node, ast.Compare) and len(node.ops) == 1
             and isinstance(node.ops[0], ast.Eq) and isinstance(node.left, ast.Attribute)
             and isinstance(node.left.value, ast.Name) and node.left.value.id == "self"
             and node.left.attr == "k31_two_term"]
    if len(gates) != 1 or not isinstance(gates[0], ast.Constant):
        raise ValueError("Expected one literal corrector activation gate")
    gate = gates[0]
    lines = text.splitlines(keepends=True)
    line = lines[gate.lineno - 1].encode()
    lines[gate.lineno - 1] = (line[:gate.col_offset]
                             + repr(float(attrs["k31_two_term"])).encode()
                             + line[gate.end_col_offset:]).decode()
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resources", type=Path, help="探索で固定した資源設定のJSON object")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    attrs, blob = ck["export"]
    resources = dict(ck.get("chain_config", {}))
    explicit = json.loads(a.resources.read_text()) if a.resources else {}
    if not isinstance(explicit, dict) or (set(resources) | set(explicit)) - RESOURCE_KEYS:
        raise ValueError("Only the eight supported resource attributes may be exported")
    for key, value in explicit.items():
        if type(value) is not int:
            raise ValueError(f"Resource setting must be an integer: {key}")
        if key in resources and value != resources[key]:
            raise ValueError(f"Resource setting differs from training checkpoint: {key}")
    attrs = dict(attrs, **dict(resources, **explicit))
    state = ck["state"]
    text = a.source.read_text()
    text = _replace_attributes(text, attrs)
    lines = "\n".join(f'    "{blob[i:i + 96]}"' for i in range(0, len(blob), 96))
    text = sub_once(r"_OMEGA_CORRECTOR_B64 = \(\n.*?\n\)", lambda _: f"_OMEGA_CORRECTOR_B64 = (\n{lines}\n)",
                    text, re.S)
    hist = state.get("history", [])
    best = next((h for h in hist if h["epoch"] == state.get("best_epoch")), None)
    note = (f"# research/torch_port/train.py で鎖ごと Adam で再当てはめ。\n"
            f"# train_sets={ck['args']['train_sets']}、開発64〜127で選択。\n"
            f"# run={ck['args']['run']} epoch={state.get('best_epoch')} "
            f"dev64_torch_mse={best['dev_mse'] if best else float('nan'):.4e}。")
    text = re.sub(r"^# research/torch_port/train.py で鎖ごと Adam で再当てはめ。\n"
                  r"# train_sets=[^\n]*\n", "", text, flags=re.M)
    text = sub_once(r"^# run=[^\n]*$", note, text, re.M)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print("wrote", a.out, attrs)


if __name__ == "__main__":
    main()
