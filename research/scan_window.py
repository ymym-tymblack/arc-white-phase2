import sys; sys.path.insert(0, "research")
from eval_mini import Evaluator, ROOT
ev = Evaluator(range(8))
p = str(ROOT / "research/archive/k3_factored_v1.py")
for w in (16, 14, 12, 10, 8, 6, 4):
    r = ev.score(p, window=w)
    print(f"window={w:2d} adj={r['adj']:.3e} mse={r['mse']:.3e} util={r['util']:.3f} resid={r['resid']:.2f}", flush=True)
