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
| 計画3 の既定 | 最終確認 60 | 1.82e-7 | 0.324 | 5.88e-8 |
| 計画3改の既定(旧版、`research/archive/k3_factored_v1.py`) | 最終確認 60 | 1.69e-7 | 0.324 | 5.47e-8 |
| ARC 公式参照実装 K=3 SIMPLE(GPU、提出不可の参考値) | MLP 0〜3 | 3.7〜4.6e-8 | — | — |
| 家族版(定数の当てはめなし) | 最終確認 60 | 3.25e-8 | 0.459 | 1.49e-8 |
| ARC 参照実装 K=3 AUGMENT(GPU、参考値、1 MLP 35 分) | MLP 0〜1 | 2.0e-8 | — | — |
| **+ 古い家族の共有基底 + k3 脚の誕生(現在)** | **最終確認 60** | **3.01e-8** | **0.425** | **1.28e-8** |

現在の版は 1 スレッドで 1 MLP 約 15 s、residual 0.19〜0.24 s(上限 0.4 s)、最大メモリ 1.2 GB(`research/preflight.py`)。
深さ 17〜24 は軽い設定(窓 8、k3 脚なし)、深さ 25 以上は共分散伝播 + κ4 放射だけに落とす(residual 時間が層数に比例するため)。

## 残り誤差の内訳と改善(第3ラウンド: C → A → B)

| 手順 | 分かったこと / 結果 |
|---|---|
| 参照実装 AUGMENT・K4(`gpu_reference.py`) | AUGMENT は SIMPLE の約 2 倍良い(2.0e-8)。K4 は幅 1024 で GPU のメモリが足りず実行できない |
| AUGMENT の項を 1 つずつ外す(`aug_ablate.py`、参照実装に研究用スイッチを追加) | 層 4 の平均誤差への寄与: 追加 Edgeworth 項 43%(大半は家族版に既にある)、k3 脚の誕生 12%、κ4 (2,1,1) の誕生 2%、κ4 脚・二重辺はほぼ 0 |
| κ4 (2,1,1) の誕生(λ ρ_ij)、`gpu_k4_lambda.py` | GPU MC で λ_ℓ ≈ 0.3 ω_ℓ を確認。入れても最終層 MSE は変わらず FLOP が増える → 不採用(`beta_k4=0`) |
| κ4 放射 → 誕生(`k4_birth_rad`) | 効果ほぼなし → 不採用 |
| A: 古い家族の共有基底(`age_old=6`, `rank_old=256`) | 年齢 6 以上の家族の伝播脚 M を、運び続ける共有基底 U に射影(M ≈ U F、輸送は U だけで厳密)。検証 28 MLP で 1.540e-8 → 1.382e-8 |
| B: k3 脚の誕生(`k3leg`, `k3leg_rank=16`) | 中心 q の脚 X = M B、B_iq = (c2_i c2_q τ_iiq + c1_i c3_q τ_iqq)/2。B を階数 16 で近似すると密な版と同じ精度で費用はわずか。検証 28 MLP で 1.382e-8 → 1.313e-8 |

## 構造の差の特定(GPU、RTX 5090)

定数の調整では最終層 MSE が約 1.7e-7 で止まったため、ARC の公式参照実装
(github.com/alignment-research-center/mlp_cumulant_propagation、`/kaggle/ext` に取得)と
GPU の MC 真値(2e8 サンプル)を作り、層ごとに内部量を照合した。

| 手順 | 分かったこと |
|---|---|
| 参照実装を mini で実行(`gpu_reference.py`) | 同じ K=3 でも最終層 MSE 3.7e-8。私の版(定数なし)は 4.8e-7 で 13 倍の差。差は層 2 から開く |
| 内部量を MC 真値と照合(`gpu_mc_stats.py`, `compare_internals.py`, `compare_three.py`) | 層 2 では私の版(iik 補正あり)と参照は一致。層 3 以降、私の版の D3・D21 の誤差が 16〜26% に膨らむ(参照は 4%) |
| 真値の差し込み(オラクル)と幅 128 の成分比較(`n128_vs_ref.py`) | 抜けていたのは κ4 の放射部分(κ̃4_iijj = ω/3)の、共分散と同一添字 3 次切り口への寄与。入れると内部量が参照と一致 |
| 同一添字切り口の補正 Δ の構造(`/tmp` の試験、コミット e16a767) | Δ ≈ 主要項(誕生脚 A の列スケール)+ 階数 16 の残差で精度が変わらない → 別の密な脚が不要 |
| 家族版(`candidates/k3_factored.py`) | 補正つきで FLOP は補正なしと同じ。窓 16 で最終層 MSE 3.2e-8(参照実装より良い) |

GPU 研究環境は `research/gpu_env.sh`(提出用とは別の Python 3.12 環境、torch 2.14 + CUDA 13)。wandb はオフライン
記録(`/kaggle/ext/wandb`)。

## 計画0〜4(改善の第2ラウンド)

| 計画 | 内容 | 結果 |
|---|---|---|
| 0 | 評価分割の固定、縮小係数を 0〜39 番だけで当てはめ直し | 最終確認用で MSE 0.82 倍(係数は分割間で安定) |
| 1 | 同一添字の 3 次切り口(iik/iii)の補正源 | n=128 で密版と一致し MSE 1.1〜1.3 倍改善。n=1024 では生 MSE は下がるが FLOP 増を上回らず、既定では不使用(`window_iik=0`) |
| 2 | κ4[z_i,z_j,z_k,z_k] を 3 次の誕生に反映 | ρ_ikρ_jk 比例の R² 0.03〜0.38、真の値の注入でも 1.15 倍悪化 → 不採用 |
| 3 | 窓・倍率・縮小係数・kmax を貪欲法 → TPE で探索 | 検証 28 MLP で 6.79e-8 → 6.06e-8、最終確認 60 MLP で 7.26e-8 → 5.88e-8 |
| 4 | 形状(1024×16/24/32、256×8/32)、FLOP、時間、residual、メモリ、公式ランナー | すべて上限内 |
| 3改 | 探索用を 40 MLP に拡大、calib を解析的に最適化、焼きなまし法 → ビームサーチ | 探索用 6.06e-8 → 5.79e-8、最終確認用 5.88e-8 → 5.47e-8 |

評価分割の感度(`research/split_sensitivity.py`): 旧/新 2 設定を 100 MLP で評価し、ランダムな分割
5000 通りで改善率を測ると、12 個で 0.812 ± 0.026、60 個で 0.812 ± 0.008(95% 区間)。改善は分割の
選び方に依らず、全 100 MLP で新設定が良かった。分割そのものを最適化すると、最終確認用の値が
本番より楽観的になるだけなので行わない。

副次効果: 窓の外に出る源を輸送前に捨てるよう直し、C/B が 0.340 → 0.324 に下がった(精度は同じ)。

## 構成

| ファイル | 内容 |
|---|---|
| `research/k3_dense.py` | 密テンソル版(n=128〜256 用)。論文の diagram summation formula(Theorem A.2)を τ 1次まで実装。アブレーション用スイッチ付き |
| `research/check_k3_dense.py`, `ablate_k3.py` | MC との照合、項ごとのアブレーション |
| `research/mc_cumulants.py`, `oracle_k4.py`, `oracle_k4b.py`, `k4_birth.py` | κ4 を MC で測って密版に注入する実験(計画2 を含む) |
| `candidates/k3_factored.py` | 家族版(提出用) |
| `research/archive/k3_factored_v1.py` | 旧版(計画0〜4 と探索スクリプトの再現用) |
| `research/check_factored.py`, `check_factored_iik.py` | 旧版と密版の一致確認(最大差 8e-7、iik あり 7e-7) |
| `research/gpu_env.sh`, `gpu_reference.py`, `gpu_ref_internals.py`, `gpu_mc_stats.py` | GPU での参照実装・MC 真値 |
| `research/export_mini.py`, `dump_ours.py`, `compare_internals.py`, `compare_three.py`, `n128_vs_ref.py`, `oracle_mc.py`, `oracle_refomega.py` | 層別照合とオラクル実験 |
| `research/eval_mini.py`, `scan.py`, `scan_fit.py` | mini 上の並列評価ハーネス(公式スコアと同じ式)と設定の一括評価(`scan_fit` は縮小係数を解析的に最適化) |
| `research/tune_k3.py`, `tune_k3_v2.py`, `search_k3.py` | 段階4の TPE、計画3の貪欲法 → TPE、計画3改の焼きなまし法 → ビームサーチ |
| `research/split_sensitivity.py` | 評価分割の選び方で改善率がどれだけ揺れるか |
| `research/preflight.py` | 提出前の安全確認(形状・FLOP・時間・residual・メモリ) |
| `research/stage5_collect.py`, `stage5_analyze.py` | 学習補正の有効性検証 |
| `scripts/measure_layer_error.py`, `measure_nongaussian.py` | 層別誤差・非ガウス性の測定 |

## 数式の要点

- 事前活性化 `z` の状態 `(m, S, T)`: 平均・共分散・3次キュムラント。κ4 は放射部分だけ(κ̃4_iiii = ω、κ̃4_iijj = ω/3)。
- ReLU 後の共分散は Hermite 展開の接続図の和(Mehler 項 + τ の 1 次 + κ4 放射の 1 次)。
- T は家族の和で持つ: `T = Σ_q c2_q Sym(A A M) + Σ_q Sym(M M PΔ)`、`PΔ = A diag(e) + M diag(dd) + Z Qᵀ`。
  ReLU が読むのは `D3 = T_aaa` と `D21 = T_aab` だけ(家族全体で n³ の行列積 2 回)。
  輸送は `X ← Wᵀ diag(Φ(α)) X`(X = A, M, Z)。家族は横並びの行列で持ち、家族数によらず演算回数が一定。
- ω の層平均は表 `ω_ℓ = G[ℓ]/n`(MC 実測、参照実装の解析値と 2〜15% で一致)。

## 測って分かったこと(要点)

- 周辺の非ガウス性(Edgeworth)は誤差の約 5%。残りは μ_z, σ_z の伝播誤差で、ニューロン間の
  交差キュムラントが主因(独立和の近似は κ4 を 60 倍過小評価)。
- 密版の項のうち効くのは `rho`, `tau1`, `pairtau`, `meantau`。`tau1e`, `txxx` は不要、`txxy` は約 5%。
- 古い源を捨てる(window)のは旧版では得だったが、家族版では窓 16(全部持つ)が最良。窓 10 では最終層 MSE が 2 倍以上悪い。
- 学習補正: 9 特徴の ridge(0.816)・GBM(0.83〜0.85)は 1 パラメータの縮小(0.825)を超えない。

## 未実装・既知の制約

- 深さが 16 を超える場合の ω 表は外挿。FLOP ガードで予算内には収まる(深さ 24 で 0.78B、32 で 0.69B)。
- Strassen 型の分割は、公平性の扱いが不明なので使っていない。
- residual 時間は 0.22 s 前後で、上限 0.4 s までの余裕は約 1.8 倍。評価機が遅い場合の余裕は小さい。
- 家族版の定数は ω の表だけ(倍率 1、縮小係数 0)。縮小係数の解析的最適値は -1.4e-6 でほぼ 0。
