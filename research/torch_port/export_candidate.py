"""学習済みチェックポイントの値を estimator.py の写しへ書き込み、候補ファイルを作る。

クラス属性の既定値(k31_two_term・regen_p0..p3・regen_feed・regen_use31)、補正器の base64
(_OMEGA_CORRECTOR_B64 と同じ並び)を差し替え、補正器を使う条件の k31_two_term == 3.0 を
当てはめた値に置き換える。それ以外のコードは元のまま。

例: research/gpu_env.sh research/torch_port/export_candidate.py /kaggle/ext/torch_ckpt/adam1/best.pt \
        --source research/frozen_phase2/combined.py --out research/candidates/adam_fit.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def sub_once(pattern, repl, text, flags=0):
    new, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise ValueError(f"pattern not found: {pattern}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    attrs, blob = ck["export"]
    state = ck["state"]
    text = a.source.read_text()
    b2 = repr(float(attrs["k31_two_term"]))
    text = sub_once(r"^    regen_p0 = .*$", f"    regen_p0 = {attrs['regen_p0']!r}", text, re.M)
    text = sub_once(r"^    regen_p1 = regen_p2 = .*$",
                    f"    regen_p1 = {attrs['regen_p1']!r}\n    regen_p2 = {attrs['regen_p2']!r}",
                    text, re.M)
    text = sub_once(r"^    regen_p3 = .*$", f"    regen_p3 = {attrs['regen_p3']!r}", text, re.M)
    text = sub_once(r"^    regen_feed = .*$", f"    regen_feed = {attrs['regen_feed']!r}", text, re.M)
    text = sub_once(r"^    regen_use31 = .*$", f"    regen_use31 = {attrs['regen_use31']!r}", text, re.M)
    text = sub_once(r"^    k31_two_term = 3\.0$", f"    k31_two_term = {b2}", text, re.M)
    text = sub_once(r"and self\.sphere_input and self\.k31_two_term == 3\.0\)",
                    f"and self.sphere_input and self.k31_two_term == {b2})", text)
    lines = "\n".join(f'    "{blob[i:i + 96]}"' for i in range(0, len(blob), 96))
    text = sub_once(r"_OMEGA_CORRECTOR_B64 = \(\n.*?\n\)", lambda _: f"_OMEGA_CORRECTOR_B64 = (\n{lines}\n)",
                    text, re.S)
    hist = state.get("history", [])
    best = next((h for h in hist if h["epoch"] == state.get("best_epoch")), None)
    note = (f"# research/torch_port/train.py で鎖ごと Adam で再当てはめ。\n"
            f"# train_sets={ck['args']['train_sets']}、開発64〜127で選択。\n"
            f"# run={ck['args']['run']} epoch={state.get('best_epoch')} "
            f"dev64_torch_mse={best['dev_mse'] if best else float('nan'):.4e}。")
    text = sub_once(r"^# run=v2s_omega_dm .*$", note, text, re.M)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print("wrote", a.out, attrs)


if __name__ == "__main__":
    main()
