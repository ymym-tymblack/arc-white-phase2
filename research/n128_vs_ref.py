"""幅 128 で、密版(全項目)と参照実装の 3 次キュムラントを成分ごとに比べる(GPU 研究環境)。

  research/gpu_env.sh research/n128_vs_ref.py <seed> <層数>
h(ReLU 後)の κ3 を添字の型ごと(全部異なる / 2 つ同じ iik / 全部同じ iii)に分けて相対差を出す。
"""
import sys

import numpy as np
import torch
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.mlp import MLP

sys.path.insert(0, "/kaggle/arc-white-phase2/research")
import k3_dense as k3  # noqa: E402

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    seed, L = int(sys.argv[1]), int(sys.argv[2])
    n = 128
    r = np.random.default_rng(100 + seed)
    W = [(r.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32) for _ in range(L)]
    dev = torch.device("cuda")
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).double().to(dev)
    for j in range(L):
        mlp.Ws[j].weight.data = torch.from_numpy(W[j].astype(np.float64).T).contiguous().to(dev)
    K_in = {1: torch.zeros(n, dtype=torch.float64, device=dev), 2: torch.eye(n, dtype=torch.float64, device=dev)}
    K = mlp_kprop(mlp, K_in, k_max=3, kind=Kind.SIMPLE, factor=True, output_all=True, up_to_layer=f"act{L-1}")
    tr = []
    groups = sys.argv[3].split(",") if len(sys.argv) > 3 else ["rho", "tau1", "tau1e", "txxy", "txxx", "pairtau", "meantau", "iik", "iii"]
    kmax = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    k3.GROUPS = set(groups)
    kap4 = None
    if len(sys.argv) > 5 and sys.argv[5] == "refk4":
        # 参照実装の 4 次キュムラント(放射部分: (4,) と (2,2) の切り口)を密版に渡す
        kap4 = []
        for l in range(1, L + 1):
            pre = K[f"pre{l-1}"]
            var = pre[2].get_dslice((2,)).cpu().numpy()
            if 4 in pre:
                k4d = pre[4].get_dslice((4,)).cpu().numpy() / var**2
                k22 = pre[4].get_dslice((2, 2)).cpu().numpy() / np.outer(var, var)
            else:
                k4d, k22 = np.zeros(n), np.zeros((n, n))
            kap4.append((k4d, k22, np.zeros((n, n))))
        print("参照実装の κ4 を密版に入力")
    k3.propagate([w.astype(np.float64) for w in W], kmax=kmax, emax=2, trace=tr, kap4=kap4)
    print("密版の項:", sorted(groups), "kmax", kmax)
    ii = np.arange(n)
    I, J, Kx = np.meshgrid(ii, ii, ii, indexing="ij")
    distinct = (I != J) & (J != Kx) & (I != Kx)
    rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)  # noqa: E731
    print(f"seed {seed}: 密版(全項目) と 参照実装 の差(相対)")
    print(f"{'層':>2} | {'z: m':>8} {'z: var':>8} {'z: T 異':>9} {'z: T iik':>9} {'z: T iii':>9} | "
          f"{'h: μ':>8} {'h: C対角外':>10} {'h: U 異':>9} {'h: U iik':>9} {'h: U iii':>9}")
    for l in range(1, L + 1):
        m, S, T, mu, C, U = tr[l - 1]
        pre, act = K[f"pre{l-1}"], K[f"act{l-1}"]
        m_r = pre[1].to_tensor().cpu().numpy()
        S_r = pre[2].to_tensor().cpu().numpy()
        mu_r = act[1].to_tensor().cpu().numpy()
        C_r = act[2].to_tensor().cpu().numpy()
        U_r = act[3].to_tensor().cpu().numpy() if 3 in act else np.zeros((n, n, n))
        T_r = pre[3].to_tensor().cpu().numpy() if 3 in pre else np.zeros((n, n, n))
        T = T if T is not None else np.zeros((n, n, n))
        off2 = ~np.eye(n, dtype=bool)
        iik_T = np.array([[T[i, i, k] for k in range(n)] for i in range(n)])[off2]
        iik_Tr = np.array([[T_r[i, i, k] for k in range(n)] for i in range(n)])[off2]
        iik_U = np.array([[U[i, i, k] for k in range(n)] for i in range(n)])[off2]
        iik_Ur = np.array([[U_r[i, i, k] for k in range(n)] for i in range(n)])[off2]
        print(f"{l:>2} | {rel(m, m_r):8.1e} {rel(np.diag(S), np.diag(S_r)):8.1e} "
              f"{rel(T[distinct], T_r[distinct]):9.1e} {rel(iik_T, iik_Tr):9.1e} "
              f"{rel(np.einsum('iii->i', T), np.einsum('iii->i', T_r)):9.1e} | "
              f"{rel(mu, mu_r):8.1e} {rel(C[off2], C_r[off2]):10.1e} {rel(U[distinct], U_r[distinct]):9.1e} "
              f"{rel(iik_U, iik_Ur):9.1e} {rel(np.einsum('iii->i', U), np.einsum('iii->i', U_r)):9.1e}",
              flush=True)
