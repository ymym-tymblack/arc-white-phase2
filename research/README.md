# K=3 キュムラント伝播(Phase 2 向け推定器)

提出候補は `candidates/k3_factored.py`(単一ファイル、flopscope のみ)。

```bash
uv run whest run --estimator candidates/k3_factored.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
uv run whest package --estimator candidates/k3_factored.py --output submission.tar.gz
```

## 結果(mini 100 MLP、N=1e9 の正解)

| 版 | 最終層 MSE | C/B | 調整後スコア |
|---|---:|---:|---:|
| 公式 共分散伝播 (`examples/03`、10 MLP) | 4.4e-6 | 0.0235(0.1 に切り上げ) | 4.4e-7 |
| 因子化 K=3(段階2、3 MLP) | 6.3e-7 | 0.449 | 2.8e-7 |
| + κ4 対角(段階3、3 MLP) | 3.6e-7 | 0.449 | 1.6e-7 |
| + 探索した window/倍率(段階4、検証 28 MLP) | 2.7e-7 | 0.340 | 9.3e-8 |
| + 出力縮小係数(段階5、100 MLP) | 2.3e-7 | 0.340 | **7.7e-8** |

residual 時間は 1 MLP あたり約 0.17 s 以下(上限 0.4 s)、`--max-threads 1` でも 1 MLP 約 7 s。

注意: 段階4の定数は mini の 12 MLP で探索し 28 MLP で検証した。段階5の縮小係数は 100 MLP 全部で
当てはめた(5 分割 CV では MSE 比 0.825 で、定数の安定性は確認済み)。最終の 7.7e-8 は
定数を当てはめた MLP を含むため、未知の MLP ではもう少し悪い可能性がある。

## 構成

| ファイル | 内容 |
|---|---|
| `research/k3_dense.py` | 密テンソル版(n=128〜256 用)。論文の diagram summation formula(Theorem A.2)を τ 1次まで実装。アブレーション用スイッチ付き |
| `research/check_k3_dense.py`, `ablate_k3.py` | MC との照合、項ごとのアブレーション |
| `research/mc_cumulants.py`, `oracle_k4.py`, `oracle_k4b.py` | κ4 を MC で測って密版に注入する実験 |
| `candidates/k3_factored.py` | 因子化版(提出用)。密版の最小構成 {rho, tau1, pairtau, meantau} と一致 |
| `research/check_factored.py` | 因子化版と密版の一致確認(最大差 8e-7) |
| `research/eval_mini.py` | mini 上の並列評価ハーネス(公式スコアと同じ式) |
| `research/tune_k3.py` | Optuna(TPE)による探索 |
| `research/stage5_collect.py`, `stage5_analyze.py` | 学習補正の有効性検証 |
| `scripts/measure_layer_error.py`, `measure_nongaussian.py` | 層別誤差・非ガウス性の測定 |

## 数式の要点

- 事前活性化 `z` の状態 `(m, S, T)`: 平均・共分散・3次キュムラント。
- ReLU 後の共分散は Hermite 展開の接続図の和(Mehler 項 `Σ_k c_k c_k ρ^k/k!` + τ の 1 次補正)。
- T は源(source)の和で持つ: `T_abc = Σ_b Σ_q w_q (A_aq A_bq P_cq + 置換)`。
  ReLU が読むのは `D3 = T_aaa` と `D21 = T_aab` だけ(n³ の行列積 2 回で読み出す)。
  輸送は `A ← Wᵀ diag(Φ(α)) A`、`P ← Wᵀ diag(Φ(α)) P`。誕生は `A = Wᵀ diag(c1) ρ, P = Wᵀ, w = c2`。
- κ4 は対角の超過尖度 `ω_ℓ = G[ℓ]/n` だけを平均・分散の補正に使う(`G` は MC 実測の表、`ω ∝ 1/n`)。
  共分散側(κ4_iijj など)だけ入れると悪化する(オラクルで確認)。(2,1,1) の U への反映は未実装。

## 測って分かったこと(要点)

- 周辺の非ガウス性(Edgeworth)は誤差の約 5%。残りは μ_z, σ_z の伝播誤差で、ニューロン間の
  交差キュムラントが主因(独立和の近似は κ4 を 60 倍過小評価)。
- 密版の項のうち効くのは `rho`, `tau1`, `pairtau`, `meantau`。`tau1e`, `txxx` は不要、`txxy` は約 5%。
- 古い源を捨てる(window)と、MSE がほぼ変わらず FLOP が減る(7〜10 付近が最良)。
- 学習補正: 9 特徴の ridge(0.816)・GBM(0.83〜0.85)は 1 パラメータの縮小(0.825)を超えない。

## 未実装・既知の制約

- `iik` 補正(U の同一添字スライス)は未実装。n=256 のアブレーションで約 1.3 倍の差。
- 深さが 16 を超える場合の ω 表は外挿。FLOP ガードで予算内には収まる(window なしで測った値: 深さ 32 で 0.82B)。
- Strassen 型の分割は、公平性の扱いが不明なので使っていない。
- 定数(`omega_scale`, `tau_scale`, `calib`, `window`)は理論値ではなく当てはめ値。MLP の形状が
  変われば再調整が必要。
