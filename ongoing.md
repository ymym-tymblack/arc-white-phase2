# 方向依存4次閉包を採用済み

`candidates/k3_factored.py` の既定と `submission.tar.gz` を更新した。外部への提出はしていない。
アーカイブ自体は既存の `.gitignore` 方針で追跡せず、固定ソースと生成来歴をGit管理する。
推論入力のモンテカルロは0。GPUのMCは独立した評価用の正解平均の作成に限る。

## 保持する成分と安定化

- 4次テンソル全体ではなく、κ4=(1/6)Sym6(G⊗I)、G=diag(g)+λC_off の調和成分を保持する。
- raw対角を線形輸送し、非対角を既存の共分散方向から再生。ReLU後の小さな分散では割らない。
- K22・K31・K211を2次および3次の更新へ反映。4次→3次の脚は既存の家族の積へ折り込み、家族ごとの新しいn³積を増やさない。
- 符号付きEdgeworth周辺分布では実測でE[h²]<0になったため、その候補は不採用。1+u+u²/2の正値Hermite密度を正規化・標準化して解析積分する方式へ変更した。
- rawな前後の分散を検査し、実質的な負値は拒否する。負の分散を黙って丸めて通す方式ではない。丸め誤差だけを許容する。
- 深い負の裾は32段Mills比で評価。FP64は周辺rawモーメント、密な状態はFP32。

採用値は再生係数(1.6,1.2,1.2,1.6)、4→3倍率0.5、K31倍率1、再生範囲1〜16(0始まり・終端を含む)、
家族年齢7、共有階数256、κ3脚階数16、窓16、calib=0。
`directional_k4=False` は対照用に残すが、旧版の再現には旧保持設定も明示する。

## 探索と最終比較

正値版はSA・GA・山登りで計94設定を評価した。SA24手で18回受理(うち悪化方向8回)、
GAは15交叉・26遺伝子変異、山登りは2回改善。8プロセス×4スレッドで評価し、各試行・探索イベントでcheckpointを保存。
wandb記録は `results/directional_execution.json` から辿れる。
開発12個の同じ保持資源による対照でも改善したため、κ3脚の階数変更だけによる改善ではない。

mini12〜39の28個で少数候補を選択した後、単一スレッド安全検査を通したソースと設定を固定。
その後、未使用だった幅1024・16層のseed26051591〜26051594に候補を1つだけ評価した。

| 指標(未使用4個の平均) | 従来 | 採用版 |
|---|---:|---:|
| 調整後スコア(参照雑音を差し引かない公式形式) | 1.32384e-8 | 1.02848e-8 |
| 調整後スコア(参照平均の推定雑音を差し引く) | 1.28390e-8 | 9.87163e-9 |
| C/B | 0.424705 | 0.439334 |

公式形式で22.31%、参照雑音補正後で23.11%改善、FLOP増は3.44%。4個すべてが改善し、独立な参照2系列の平均改善も両方正。
事前の採用条件を満たした。最終結果を使った再調整は行っていない。
参照はRTX5090で各ネットワーク4000万入力×独立2系列。雑音補正は信頼区間ではなく、4個での結果を任意のMLPへ一般化はしない。
この4個は今後の「未使用評価集合」には再利用しない。

## 実行上限と検証

初回はresidual上限を超えたため、切り口・Hermite基底・rawモーメントの演算をnativeな積へ集約した。
非ゼロの4次成分、分散ガード、安全上限は削っていない。既存の深さ別3次家族ポリシー以外の形状別fallbackは追加していない。

- 数式恒等式・独立した正値密度の数値積分・実ネットワークの146検査を通過。
- 最適化前後の4例で予測がbitwise一致。旧保持設定を明示した無効モードではFLOPも一致。
- 1スレッドの7形状すべてで安全検査を通過。最大residual0.338秒(上限0.4)、最大wall21.2秒、観測最大RSS1.37GiB。
- 公式subprocessでmini2個を実行し、予算・wall・residual・推定器エラーは0件。
- `whest validate`、関連4ファイルのRuff F検査を通過。
- packagerでアーカイブを生成し、manifestのソースSHA256が固定候補と一致することを確認。

固定ソースSHA256: `258bb19fbdcb6579c1bf8c7b8f908c3eb167e16efaab1d29517df30d1c7dbaae`。

## 保存先

- 数式・失敗原因・寄与分解・採否: `research/README.md`
- 数式/正値性監査: `research/regen_audit.py`、`results/regen_selected_audit.json`
- 探索: `research/search_directional_k4.py`、`results/directional_positive_search.json`
- 選択: `results/directional_positive_confirmation.json`、`results/directional_positive_validation.json`
- 固定条件/最終比較: `results/directional_protocol.json`、`results/directional_frozen.json`、`results/directional_final.json`
- 安全検査/公式実行: `results/directional_preflight.json`、`results/directional_official.json`
- 演算集約の同値比較: `results/directional_perf_parity.json`。一時比較スクリプトは検証後に削除済み。
- 実行来歴・wandb・参照生成・採用: `results/directional_execution.json`
- GPU参照とwandb実体: `/kaggle/ext/closure`、`/kaggle/ext/wandb`

前ラウンドのGamma共通尺度・標準化行列コア・強トレース縮約は依然不採用。
今回のraw方向依存閉包・正値周辺密度とは区別し、失敗の記録は研究READMEと既存結果に残している。
