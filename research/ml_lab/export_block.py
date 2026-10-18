"""学習済み補正器 JSON を estimator.py に貼り付けられる定数ブロック(Python の float タプル)に変換する。

例: python research/ml_lab/export_block.py omega=research/ml_lab/omega_corrector.json \
        tdiag=research/ml_lab/tdiag_corrector.json > research/ml_lab/corrector_block.py
出力の _CORRECTOR_<NAME> は (特徴名, 平均, 1/標準偏差, ((W 行優先, b, 入力次元, 出力次元), ...), 出力倍率)。
"""
from __future__ import annotations

import json
import sys


def fmt(values):
    return "(" + ", ".join(f"{float(v):.8g}" for v in values) + ",)"


def block(name, path):
    spec = json.loads(open(path).read())
    lines = [f"# {name}: run={spec.get('run')} epoch={spec.get('epoch')} val_r2={spec.get('val_r2'):.4f}",
             f"_CORRECTOR_{name.upper()} = ("]
    lines.append("    " + repr(tuple(spec["features"])) + ",")
    lines.append("    " + fmt(spec["mean"]) + ",")
    lines.append("    " + fmt(1.0 / s for s in spec["std"]) + ",")
    lines.append("    (")
    for layer in spec["layers"]:
        W = layer["W"]
        flat = [x for row in W for x in row]
        lines.append(f"        ({fmt(flat)}, {fmt(layer['b'])}, {len(W)}, {len(W[0])}),")
    lines.append("    ),")
    lines.append(f"    {float(spec['yscale']):.8g},")
    lines.append(")")
    return "\n".join(lines)


def main():
    parts = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        parts.append(block(name, path))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
