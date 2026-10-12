import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
d = np.load("/tmp/stage5.npz")
pred, truth = d["pred"], d["truth"]            # (100, 16, 1024)
r = truth[:, -1] - pred[:, -1]                  # 最終層の残差 (100, 1024)
base = (r**2).mean()
print(f"最終層 MSE (100 MLP 平均) = {base:.3e}  各 MLP の MSE の変動係数 = {(r**2).mean(1).std()/ (r**2).mean(1).mean():.2f}")
bias = r.mean(1, keepdims=True)
print(f"MLP ごとの平均残差(バイアス)を引いた場合の MSE = {((r-bias)**2).mean():.3e}  (バイアスが占める割合 {(bias**2).mean()/base:.1%})")

# 特徴量(最終層、ニューロンごと)
L = -1
X = np.stack([d["f_m"][:, L], d["f_sigma"][:, L], d["f_alpha"][:, L], d["f_tdiag"][:, L],
              d["f_mu_gauss"][:, L], d["f_mu"][:, L], d["f_var_post"][:, L],
              d["wnorm2"][:, L], d["wsum"][:, L]], axis=-1)   # (100, 1024, 9)
names = ["m", "sigma", "alpha", "tau", "mu_gauss", "mu", "var_post", "wnorm2", "wsum"]
print("残差と各特徴の相関:", {n: round(float(np.corrcoef(X[..., i].ravel(), r.ravel())[0, 1]), 3) for i, n in enumerate(names)})
# 予測値と正解の関係(相対残差)
print(f"残差と予測値 mu の相関 {np.corrcoef(pred[:, -1].ravel(), r.ravel())[0,1]:.3f}")

# MLP 単位の 5 分割 CV
folds = np.arange(100) % 5
def cv(model_fn, feat):
    out = np.zeros_like(r)
    for f in range(5):
        tr, te = folds != f, folds == f
        m = model_fn()
        m.fit(feat[tr].reshape(-1, feat.shape[-1]), r[tr].ravel())
        out[te] = m.predict(feat[te].reshape(-1, feat.shape[-1])).reshape(te.sum(), -1)
    return out
mu_ = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-12
Xs = (X - mu_) / sd
for nm, fn, feat in (("ridge(線形)", lambda: Ridge(alpha=1.0), Xs),
                     ("GBM", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05), X)):
    o = cv(fn, feat)
    new = ((r - o) ** 2).mean()
    print(f"{nm:12s}: CV 後の MSE = {new:.3e}  (比 {new/base:.3f})  R² = {1-new/base:.3f}")
# 全層で学習して最終層に適用 (層を特徴に追加)
