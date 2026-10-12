import sys; sys.path.insert(0, "research")
from eval_mini import Evaluator, ROOT
if __name__ == "__main__":
    ev = Evaluator(range(100)); p = str(ROOT / "research/archive/k3_factored_v1.py")
    for name, kw in (("calib=0 ", dict(calib=0.0)), ("calib既定", dict())):
        r = ev.score(p, **kw)
        print(f"{name}: 調整後={r['adj']:.3e} 生MSE={r['mse']:.3e} C/B={r['util']:.3f} residual最大={r['resid']:.2f}s 1MLP最大={r['sec']:.1f}s", flush=True)
