"""学習済み補正器 JSON を estimator.py に埋め込む定数(float32 の base64)へ変換する。

並びは mean(d), 1/std(d), 層ごとに W(行優先 in×out), b(out), 最後に yscale(1)。
例: python research/ml_lab/export_b64.py /kaggle/ext/ml_lab_ckpt/v2s_omega_dm.json
"""
from __future__ import annotations

import base64
import json
import struct
import sys


def main():
    spec = json.loads(open(sys.argv[1]).read())
    values = list(spec["mean"]) + [1.0 / s for s in spec["std"]]
    dims = [len(spec["mean"])]
    for layer in spec["layers"]:
        W = layer["W"]
        values += [x for row in W for x in row] + list(layer["b"])
        dims.append(len(W[0]))
    values.append(float(spec["yscale"]))
    blob = base64.b64encode(struct.pack("<%df" % len(values), *values)).decode("ascii")
    print(f"# run={spec.get('run')} epoch={spec.get('epoch')} val_r2={spec.get('val_r2'):.4f} "
          f"train_rows={len(spec.get('train_rows', []))} features={spec['features']}")
    print(f"_OMEGA_CORRECTOR_DIMS = {tuple(dims)}")
    print("_OMEGA_CORRECTOR_B64 = (")
    for start in range(0, len(blob), 96):
        print(f'    "{blob[start:start + 96]}"')
    print(")")


if __name__ == "__main__":
    main()
