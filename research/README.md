# K=3 キュムラント伝播(Phase 2 向け推定器)

提出候補は `candidates/k3_factored.py`(単一ファイル、flopscope のみ)。

```bash
uv run whest run --estimator candidates/k3_factored.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
uv run whest package --estimator candidates/k3_factored.py --output submission.tar.gz
```

## 結果(mini、N=1e9 の正解)

評価分割(計画0 で固定): 探索用 0〜11 番、検証用 12〜39 番、最終確認用 40〜99 番(当てはめに使わない)。

| 版 | 評価 | 最終層 MSE | C/B | 調整後スコア |
|---|---|---:|---:|---:|
| 公式 共分散伝播 (`examples/03`) | 10 MLP | 4.4e-6 | 0.0235(0.1 に切り上げ) | 4.4e-7 |
| 因子化 K=3(段階2) | 3 MLP | 6.3e-7 | 0.449 | 2.8e-7 |
| + κ4 対角(段階3) | 3 MLP | 3.6e-7 | 0.449 | 1.6e-7 |
| + 探索した window/倍率(段階4) | 検証 28 | 2.7e-7 | 0.340 | 9.3e-8 |
| 段階5 の既定(縮小係数あり) | 最終確認 60 | 2.24e-7 | 0.324 | 7.26e-8 |
| **計画3 の既定(現在)** | **最終確認 60** | **1.82e-7** | **0.324** | **5.88e-8** |

1 スレッドで 1 MLP 約 9 s、residual 0.10 s、最大メモリ 0.39 GB(`research/preflight.py`)。

## 計画0〜4(改善の第2ラウンド)

| 計画 | 内容 | 結果 |
|---|---|---|
| 0 | 評価分割の固定、縮小係数を 0〜39 番だけで当てはめ直し | 最終確認用で MSE 0.82 倍(係数は分割間で安定) |
| 1 | 同一添字の 3 次切り口(iik/iii)の補正源 | n=128 で密版と一致し MSE 1.1〜1.3 倍改善。n=1024 では生 MSE は下がるが FLOP 増を上回らず、既定では不使用(`window_iik=0`) |
| 2 | κ4[z_i,z_j,z_k,z_k] を 3 次の誕生に反映 | ρ_ikρ_jk 比例の R² 0.03〜0.38、真の値の注入でも 1.15 倍悪化 → 不採用 |
| 3 | 窓・倍率・縮小係数・kmax を貪欲法 → TPE で探索 | 検証 28 MLP で 6.79e-8 → 6.06e-8、最終確認 60 MLP で 7.26e-8 → 5.88e-8 |
| 4 | 形状(1024×16/24/32、256×8/32)、FLOP、時間、residual、メモリ、公式ランナー | すべて上限内 |

副次効果: 窓の外に出る源を輸送前に捨てるよう直し、C/B が 0.340 → 0.324 に下がった(精度は同じ)。

## 構成

| ファイル | 内容 |
|---|---|
| `research/k3_dense.py` | 密テンソル版(n=128〜256 用)。論文の diagram summation formula(Theorem A.2)を τ 1次まで実装。アブレーション用スイッチ付き |
| `research/check_k3_dense.py`, `ablate_k3.py` | MC との照合、項ごとのアブレーション |
| `research/mc_cumulants.py`, `oracle_k4.py`, `oracle_k4b.py`, `k4_birth.py` | κ4 を MC で測って密版に注入する実験(計画2 を含む) |
| `candidates/k3_factored.py` | 因子化版(提出用)。密版の最小構成 {rho, tau1, pairtau, meantau} と一致 |
| `research/check_factored.py`, `check_factored_iik.py` | 因子化版と密版の一致確認(最大差 8e-7、iik あり 7e-7) |
| `research/eval_mini.py`, `scan.py` | mini 上の並列評価ハーネス(公式スコアと同じ式)と設定の一括評価 |
| `research/tune_k3.py`, `tune_k3_v2.py` | 段階4の TPE 探索、計画3の貪欲法 → TPE 探索 |
| `research/preflight.py` | 提出前の安全確認(形状・FLOP・時間・residual・メモリ) |
| `research/stage5_collect.py`, `stage5_analyze.py` | 学習補正の有効性検証 |
| `scripts/measure_layer_error.py`, `measure_nongaussian.py` | 層別誤差・非ガウス性の測定 |

## 数式の要点

- 事前活性化 `z` の状態 `(m, S, T)`: 平均・共分散・3次キュムラント。
- ReLU 後の共分散は Hermite 展開の接続図の和(Mehler 項 `Σ_k c_k c_k ρ^k/k!` + τ の 1 次補正)。
- T は源(source)の和で持つ: `T_abc = Σ_b Σ_q w_q (A_aq A_bq P_cq + 置換)`。
  ReLU が読むのは `D3 = T_aaa` と `D21 = T_aab` だけ(n³ の行列積 2 回で読み出す)。
  輸送は `A ← Wᵀ diag(Φ(α)) A`、`P ← Wᵀ diag(Φ(α)) P`。誕生は `A = Wᵀ diag(c1) ρ, P = Wᵀ, w = c2`。
- κ4 は対角の超過尖度 `ω_ℓ = G[ℓ]/n` だけを平均・分散の補正に使う(`G` は MC 実測の表、`ω ∝ 1/n`)。
  共分散側(κ4_iijj など)だけ入れても、3 次の誕生(κ4_ijkk)に入れても悪化する(オラクルで確認)。

## 測って分かったこと(要点)

- 周辺の非ガウス性(Edgeworth)は誤差の約 5%。残りは μ_z, σ_z の伝播誤差で、ニューロン間の
  交差キュムラントが主因(独立和の近似は κ4 を 60 倍過小評価)。
- 密版の項のうち効くのは `rho`, `tau1`, `pairtau`, `meantau`。`tau1e`, `txxx` は不要、`txxy` は約 5%。
- 古い源を捨てる(window)と、MSE がほぼ変わらず FLOP が減る(7〜10 付近が最良)。
- 学習補正: 9 特徴の ridge(0.816)・GBM(0.83〜0.85)は 1 パラメータの縮小(0.825)を超えない。

## 未実装・既知の制約

- `iik` 補正は実装済みだが、幅 1024 ではスコアに効かないため既定で無効。
- 深さが 16 を超える場合の ω 表は外挿。FLOP ガードで予算内には収まる(深さ 32 で 0.77B)。
- Strassen 型の分割は、公平性の扱いが不明なので使っていない。
- 定数(`omega_scale`, `tau_scale`, `calib`, `window`)は理論値ではなく当てはめ値。MLP の形状が
  変われば再調整が必要。
