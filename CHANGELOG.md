# Changelog

Participant-visible changes to the starter kit. The kit is the published spec of the
live round, so anything that moves a rule, a parameter, a FLOP figure, or the estimator
contract is announced here.

The kit is not published to PyPI and has never carried a version tag, so entries are
dated and grouped by the dependency baseline they shipped against. **Unreleased** is
what has landed on `main` since the last dated entry. Read it before you tune anything.

## Unreleased

### 未使用126網での確認・提出用rootの採用（2026-10-19）

- 凍結後の公式full874〜999で3ソース計378予測が完走し、失敗0。優先候補は提出済み版より平均adjusted scoreが5.2195%低下、116/126網で改善した。
- 網ごとのscore比の平均0.94894619・対bootstrap95%区間[0.94231738, 0.95562123]で事前の採否条件を通過。holdoutを見た候補の入替はせず、この126網は消費済みとした。
- `estimator.py` を凍結済みage5/rank320・422網epoch16版へbyte一致で切替。切替前のrootと採否記録を保存し、実提出・既存tar更新はしていない。
- 切替後のroot自体もSDK validateとsubprocess runnerを通過（wall33.39661秒、residual0.32801秒、失敗0）。凍結候補と全16層MSE・FLOPsが一致し、全回帰テスト54件も通過した。

### 422網の再学習・安全化・最終候補凍結（2026-10-19）

- 選択したage5/rank320だけをepoch19から新規Adamで422網に再学習し、epoch16を選択。元224網の継続実験と区別してcheckpoint・W&B・初期値の来歴を保存した。
- 422網の最後の6網が8分母で過小重みになる勾配蓄積を修正。実6網の比較で補正後gradは最大絶対差1.733e-7、元224網への影響なしを確認した。
- 再学習候補の深さ24は最初residual0.419秒で失敗。学習値を変えずペア係数のstack/gatherを削減し、同一入力で全層bitwise一致、7ケース全通過へ修正した。過去の失敗は保持する。
- exact開発64網では平均adjusted scoreが提出済み版より5.5362%低下、60/64網で改善。SDK隔離実行・4GiB client/server・GPU照合も通過し、優先候補と比較2候補を最終集合の予測前に固定した。
- `export_candidate.py` はcheckpoint資源設定・初期学習と現在学習の注記を整合し、`verify_port.py` も保存資源を復元する。`apply_residual_optimization.py` はペアpackingをAST確認して転写し、安全化済み入力への適用も不変とした。

### 係数固定の資源配分探索（2026-10-19）

- destroy/repair焼きなまし24提案から5配分を開発16網、上位2配分を64網へ昇格し、exact flopscopeで完走。係数7個・補正器は固定、窓16・kmax3を維持した。
- `age_old=5, rank_old=320` が固定epoch4に対し平均adjusted scoreを0.7550%低下。MSEのみさらに良い別配分はFLOP増でscore悪化のため不採用。
- 探索は4プロセス×8スレッド、網単位cache・SA乱数状態・offline W&Bを保存。開発で選んだ有望配分であり、未知網での改善や時間適格性は別の凍結評価・安全検査で扱う。

### Adam継続・勾配監査（2026-10-19）

- 元224網・元560step cosineを保ってepoch20まで継続し、epoch19を選択。開発PyTorch MSEはepoch4から追加0.3406%低下したが、計画上限停止であり収束の証明とはしない。追加198網はこの実験には未使用。
- checkpoint有無・反復計算は全勾配までbitwise一致。一方FP32有限差分は不合格、高精度shadowにも最大約1.47%差が残る。数学的な鎖の修正やMuon優位の主張は行わない。
- `train.py` のresumeで引数・データ順・batch位置・Adam・RNG・scheduler長を復元し、明示した資源設定を学習・評価・checkpointへ一貫適用。再学習済みソースの再exportにも対応した。
- epoch19安全化版のexact開発64網評価は平均adjusted scoreが提出済み版より4.4565%低下、62/64網で改善。7ケース安全検査・4GiB client/server照合・SDK validateを通過した。基準64件はhash等を検証して再利用し、未使用holdoutとは区別する。

### 学習値を固定したresidual安全化（2026-10-19）

- `research/candidates/adam1_residual_optimized.py` でλのscalar同期・定数配列をbroadcast viewへ変更し、Strassen作業viewを再利用、内部段の右側再構成を融合した。係数・補正器・資源配分は不変。
- 同じ入力の対測定で全層bitwise一致、residual0.3515→0.3277秒、演算呼出し11428→11311。独立7形状検査は全件通過し、1024×16のresidual最大0.337秒、RSS最大3.38GiB。
- 公式4GiB配列上限のclient/serverでもsetup1.062秒、residual0.206/0.188秒、保存済みインプロセス予測と完全一致。過去の安全検査失敗は保持する。
- 学習値・資源配分の指紋を照合しつつ再学習候補へ同じ実装だけを移す生成器と、再現用smokeを追加。rootや提出アーカイブはまだ置き換えていない。

### Adam学習値の提出実装での再現確認（2026-10-19）

- `adam1/best.pt`（epoch4、学習224網）を単一flopscope候補 `research/candidates/adam1_epoch004.py` へ書き出し、checkpoint・ソースのhashと実際の学習範囲を記録した。
- PyTorchとflopscopeの全層予測を開発4網で照合し、最大絶対差4.95e-6・最終層MSE相対差0.208%以内を確認した。
- 同一開発64網で提出済み4段版と比較し、FLOP不変で平均adjusted scoreが5.250894e-9から5.033271e-9へ4.14%低下、61/64網で改善した。未知MLP・本番スコアの改善確認とは区別する。
- 網別checkpoint付き対比較ハーネスを追加し、MSEだけでなくFLOPと0.1床を含むスコア・信頼区間を計算する。比較網の欠落・重複・予測失敗を黙って除外しない回帰検査5件を通過した。
- rootの作業中ファイルや提出済み版は置き換えず、未使用holdoutと実提出は行っていない。
- 候補単独の安全検査は1024×16でresidual0.497/0.475秒となり0.4秒上限を超過。精度改善の再現と提出適格性を分け、現状の候補は提出非推奨として失敗結果も保持する。

### この作業ツリーの最終候補（2026-10-18）

- 大きな因子化行列積をStrassen–Winogradで計算し、球面入力の動径分離・(3,1)二項則・学習済みω補正器を統合。候補と比較条件を `research/frozen_phase2_final.json` に固定した。
- `estimator.py` のsetup事前構築を1秒目安に制限。クライアント方式でsetupを2.10秒から1.08秒へ短縮し、残りのviewは初回予測で構築する。予測値は変更前と一致し、初回のみreshapeのFLOPが加わりうる。
- 公式の単一配列上限は4GiBであり、ローカルflopscope-server既定の100MiBとは異なる。公式上限でクライアント方式の実行を確認した。
- 階数上限の回帰検査をGaussian・球面の両入力モードに拡張。Gaussian初層の解析平均はGaussianモードで検算する。
- 修正前ハーネスの最終128網比較は基準126件・計算削減128件・統合128件が時間超過。失敗結果は保持し、その実行からの改善主張は行わない。
- 最終評価ハーネスに網単位checkpoint・失敗の文字列記録を追加し、spawn先でBLASスレッド設定より先にNumPyを読む初期化順を修正。最終集合の追加実行には明示承認を要する。
- ユーザーが固定候補の失敗分追加評価を明示承認。修正後は3候補とも128網を完走し例外・予算超過・非有限0件、統合版の平均調整後スコアは元の自作版より44.98%低下（9.720188e-9→5.348038e-9）。本番の非公開スコア予測ではない。
- 統合版は同じソースhashの1スレッド安全検査を満たすため提出候補として推奨する。比較用の計算削減のみの旧ソースはresidual0.35秒の自己基準未達のため提出非推奨。実提出は別途承認が必要。

### この作業ツリーの推定器改善（2026-10-16）

- 提出入口をrootの `estimator.py` に統一し、ゼロ出力テンプレートと研究候補の誤梱包経路を解消。競技規則の変更ではない。
- 古い家族の残差・κ3脚を薄い因子で再利用し、不要な密行列積と最終層の全共分散を削減。同じ保持設定でFLOP8.53%減。
- SA・GA・山登りで資源配分を選択し、保持年齢6・補助階数4/4を固定。主共有階数256、4次の式と係数は維持。
- 凍結後の公式full64で、最適化前の方向依存版に対してFLOP15.80%減・MSE6.20%増・調整後スコア10.58%減、推定失敗0件を観測。
- 1スレッド7形状・公式subprocess・146項目の物理監査を通過。数値収集器の診断辞書混入と幅超過階数のreshape失敗も修正。
- 枝刈り・小行列SVDによる階数配分は動作検証済みだが、今回の採用版では無効。本番スコアの予測・外部提出は行わない。

### BREAKING CHANGE

- **Phase 2 is the live round.** The graded shape is now **width 1024 x depth 16**
  (Phase 1 was 256 x 32), so `predict()` returns a `(16, 1024)` array. The per-MLP FLOP
  budget `B_m` is **`2**41` = 2,199,023,255,552 FLOPs**, 8.085x the Phase 1 budget of
  ~2.72e11. Effective compute is now **`C_m = F_m`**: residual wall time is no longer
  priced and the residual-penalty rate λ does not apply in Phase 2. A Phase 1 estimator
  that reads `mlp.width` / `mlp.depth` still runs, but its cost and its score both change;
  anything that hardcoded the Phase 1 shape or budget now returns the wrong array.
- **Residual wall time is capped, not billed.** A hard cap of **400 ms per MLP** replaces
  the λ pricing; if you exceed it, the grader substitutes zero predictions for that MLP.
  The residual bucket covers control flow, bookkeeping, and data handling, not computation.
  Meaningful compute there is disqualifiable. See
  [docs/concepts/allowed-code.md](docs/concepts/allowed-code.md).
- **Phase 2 limits.** Wall-clock cap **120 s per MLP** (was 60 s). Solution process memory
  **8 GB** (the Phase 1 box was 64 GB). `setup()` is capped at **5 s** and runs once.
  Exceeding it fails the **whole submission**, not one MLP. **10 submissions per team per
  UTC day** (was 50). Unchanged from Phase 1: 2 vCPUs pinned to the solution with 14 to
  the flopscope backend, a 100-MLP test suite (50 public + 50 holdout), and the score
  shape `s_m = MSE_final,m * max(0.1, C_m / B_m)`.
- **Allowed code is now an explicit, enforced rule.** A submission may use the grader's
  Python interpreter, the flopscope client API, and the pure-Python standard library —
  nothing else. Vendored numpy/scipy/BLAS, compiled kernels of any kind, ctypes/cffi/FFI,
  asyncio/threads/subprocess/multiprocessing, compute while a flopscope op is in flight,
  and touching the flopscope client/transport/accounting are prohibited. Data files
  (weights, lookup tables, precomputed artifacts) remain permitted. Automated checks run on every
  submission, flagged submissions go through agent-assisted validation, and a person
  reviews anything still unresolved. Review continues after grading, and a submission
  that does not conform is invalidated once identified.

### Feat

- **docs**: add [Allowed Code](docs/concepts/allowed-code.md) — what a submission may use,
  the prohibition list, the data-file carve-out, what residual time is for, and how the
  rule is enforced.
- **docs**: add [Competition Rounds](docs/reference/rounds.md) — every round the challenge
  has run, side by side, with the flags needed to reproduce a score from an earlier one.
- **docs**: add this changelog, so the kit has somewhere to announce a parameter
  change.
- **deps**: pin `whestbench>=0.16.0,<0.17.0` and `flopscope>=0.12.0,<0.13.0`. whestbench 0.16.0
  is the first release whose *defaults* are the graded Phase 2 round (1024x16, `2**41`, a
  120 s predict cap, a 5 s `setup()` cap and a gated 400 ms residual), so under 0.15.0 the kit
  documented rules the local CLI did not enforce. flopscope 0.12.0 reprices ~40 operations
  relative to 0.11.0, but none that these examples use: `matmul`, tagged `einsum`,
  `as_symmetric`/`is_symmetric`, `fill_diagonal`, `stats.norm.*` and `astype` bill identically
  under both, so every FLOP figure in `docs/` is unchanged (`tests/test_flopscope_cost_docs.py`
  is the gate). The evaluator has since moved to `whestbench@v0.16.0` +
  `flopscope[server]==0.12.0`.
- **harness (whestbench 0.16.0)**: the local CLI now defaults to the graded round, so local
  numbers are the grader's numbers. New/changed on `whest run`: `--setup-timeout SECONDS`
  (new; default 5.0, and breaching it fails the *whole* submission, so rehearse it),
  `--wall-time-limit` default 60.0 -> 120.0, and `--no-residual-wall-time-limit` (new; needed
  only to re-score a Phase 1 round). `whestbench.budget.ROUNDS` / `CURRENT_ROUND` expose every
  round's settings as data.
- **harness (whestbench 0.16.0), scoring-affecting**: the harness now replaces a subprocess
  worker that crashes mid-suite and continues the suite; previously one crash failed every
  remaining MLP, so a submission that crashes now scores better across the suite. MLP weights
  are float32 on every construction path.
- **local_engine**: re-default `compare_against_monte_carlo` to the real Phase 2 budget —
  `estimator_budget=2**41`, and a `5e12` sampling budget applied per sample-count row.
- **meter (flopscope 0.12.1), scoring-affecting in one narrow case**: the pinned floor moves
  to `flopscope>=0.12.1` and `whestbench>=0.16.1`. A symmetry tag now describes the buffer it
  is attached to: `flops.as_symmetric()` validates within a tolerance, and it now makes the
  data exactly match the claim before tagging, copying each orbit's first entry over the
  orbit. Data that is already exactly symmetric — an identity, or a covariance built as
  `X.T @ X` — is untouched and costs the same, which is nearly every use. `flops.symmetrize()`
  gains `mode="canonical-copy"` beside the unchanged default, and it stays available for
  symmetry groups too large for the default to enumerate. **What moves a FLOP figure:**
  `fnp.full` and `fnp.full_like` no longer infer a symmetry from shape when `fill_value` is
  an array rather than a scalar, so results built that way are priced as the dense arrays
  they are. A scalar fill is unaffected at every shape.

### Docs

- Sync Phase 2 fair-accounting and packing guidance with upstream starter-kit
  commit [`5eb9aa1`](https://github.com/AIcrowd/whest-starterkit/commit/5eb9aa1455fcb3216af55994bdf25dc242b95797).
  Metering every operation does not permit packing independent values into one
  machine element for an accounting advantage. Clarify that the official
  Challenge Rules govern eligibility, not library pricing or a successful score.
- Phase 2 content pass across `docs/` and `examples/`: every cost figure re-measured at
  1024 x 16 against whestbench 0.16.0 and flopscope 0.12.0 — and identical under flopscope
  0.11.0, since 0.12.0's repricing covers no operation these examples use. The bundled
  examples now cost **86,639,616 FLOPs (0.0039% of `B_m`)** for `02_mean_propagation` and
  **51,709,240,799 FLOPs (2.3515% of `B_m`)** for `03_covariance_propagation` — a 597x
  ratio between them, up from 137.6x at the Phase 1 shape, because the covariance step is
  cubic in width.

## 2026-08-12

### Fix

- **docs(setup)**: correct the `setup()` budget contract after evaluator #203 (#35).

## 2026-08-06

### Docs

- **troubleshooting**: document the aggregate evaluation time limits, the per-MLP time
  limit, and the graded MLP count (#34).

## 2026-08-03

### Fix

- stop claiming scoring is hardware-independent — it never was (#31).
- correct the ground-truth default, the covariance symmetry idiom, the `as_symmetric`
  cost, and a demo path leak (#31).

## 2026-07-31

### Feat

- **deps**: bump to flopscope 0.10.0 + whestbench 0.14.0.

### Fix

- **examples**: re-validate covariance symmetry each layer under flopscope 0.10.0.

### Docs

- re-baseline cost numbers and meter notes for flopscope 0.10.0 / whestbench 0.14.0.

## 2026-07-24

### BREAKING CHANGE

- flopscope 0.9 moves to four-factor billing, so absolute FLOP counts change: sorts bill
  `4*N*ceil(log2 N)` rather than a flat 4x, and a reshape bills even where it returns a
  view. Every cost figure in the kit was re-derived; a Phase 1 estimator's `flops_used`
  moves without any change on your side.

### Feat

- **deps**: adopt whestbench 0.13.0 and flopscope 0.9.1.
- **local_engine**: mirror the upstream float32 sampling idiom and `MLP.seed` passthrough.

### Docs

- rewrite the flopscope primer's cost sections for four-factor billing; re-baseline the
  code-patterns cost guidance; re-profile example 02; correct the on-the-fly `n-samples`
  default to `width*width*256` and re-derive the noise floor; document `time_source=bake`
  sampling attribution in the score report; describe `whest profile-simulation` as the
  timing/correctness benchmark it is.

### Test

- drift-gate the flopscope 0.9.x billing facts the docs state.

## 2026-06-27

### Feat

- **deps**: bump whestbench to 0.12.0rc3 and the flopscope floor to 0.8.0rc5 (#29).

## 2026-06-24

### Docs

- **troubleshooting**: document common participant failure modes (#28).

## 2026-06-23

### Feat

- **deps**: pin `whestbench>=0.12.0rc2` (#25).

### Docs

- document the 4 GiB per-array memory cap (#26).
- correct the grader dependency model — there is no `requirements.txt` install step (#24).
- add the MIT LICENSE and fix the broken license link (#27).

## 2026-06-19

### Docs

- clarify `whest package` / `whest submit` ergonomics: file vs folder, credential
  exclusion, preview (#21).
- rename the README heading to "ARC WhiteBox Estimation Challenge 2026 - Starter Kit" and
  add the challenge badge hub (#23).

### Chore

- refresh `demo.cast` / `demo.gif` on 0.12.0rc0 and add the headless cast pipeline (#22).

## 2026-06-18

### Feat

- **phase 1 release**: the `v1-phase1` dataset revision, a 256 x 32 shape at a 2.72e11
  per-MLP budget, on flopscope 0.8.0rc0 / whestbench 0.11.0rc0 (#17).
- add `.whestignore` so a packaged submission stays under the 50-file cap (#18).

### Fix

- correct stale warmup-era numbers missed in the phase-1 re-profile (#19).

## 2026-06-08

### Feat

- **deps**: bump whestbench 0.10.0 / flopscope 0.5.0 and re-baseline the FLOP docs (#15).

### Docs

- document the GPU/torch flags for baking datasets (#16).

## 2026-06-04

### Docs

- align the staged ladder with the public-dataset workflow and re-record the demo against
  the public Mini split (#14).
- correct ground-truth precision and the noise floor; align scoring and defaults with
  whestbench 0.9.2 (#13).

## 2026-06-01

### Feat

- **deps**: lock whestbench 0.9.2 + flopscope 0.4.2 (#11).

### Docs

- remove the fictional Docker runner stage and renumber the ladder to five stages (#10).

## 2026-05-27

### Feat

- migrate the starters to the PyPI `whestbench` release and Hugging Face datasets (#4).

### Docs

- mini-first quickstart and modernised CLI references (#6); mention the prepared-Arrow
  fast path on Hugging Face (#7).

## 2026-04-24

### Feat

- scaffold the standalone starter kit: examples 01-04, the `local_engine.py` helpers
  (`build_mlp`, `monte_carlo_layer_means`, `compare_against_monte_carlo`), a root
  `estimator.py` template, the five-stage tutorial ladder, the reference / how-to /
  troubleshooting doc tree, and CI that runs every command the README prints.

Earlier history for the examples and docs predates this repository; it was imported
with history preserved when the kit was split out into its own repo.
