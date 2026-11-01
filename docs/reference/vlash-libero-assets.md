# VLASH 専用 LIBERO モデルと互換環境

状態: **厳密復元、API smoke、H50同期診断2 episode、実時間3条件×6 episodeまで完了**。formal は init1..5 の15 episode、development5 episodeとstartup失敗1件は分ける。実時間測定との CPU・RAM・GPU 競合を避け、モデル構築・監査・タスクは排他的な実行枠で行った。

## 固定した資産

| 対象 | 固定値 | 確認事項 |
|---|---|---|
| 作者モデル | `mit-han-lab/vlash-pi05-libero-async5` | LIBERO 用に公開された専用学習済み重み。手元では学習しない |
| モデル revision | `c85d5962a7b823f9f391f97d60227d6ca8bd7bc7` | config と重みだけの公開スナップショット |
| 重み | 7,472,953,952 bytes | SHA256 `8dd3bf4766dac481933ced7c8de5fbfc31c486e665cca33e6ddf970792a46a2e` |
| 保存要素 | 3,616,757,594 | 保存された 824 tensor の要素数。実行時の固有 Parameter 数とは区別する |
| 実行時の固有 Parameter | 3,616,757,594 | 作者の共有構造を保持。255 alias は重複して数えない。normalizer74を含む |
| 作者コード | `mit-han-lab/vlash`, `sim/libero` | revision `d2da223e2251a4a5020b667e77cf8ad57e594144` |
| 作者コード取得 | 35 files / 391,755 bytes | 元のライセンス・copyright を保持し、全ファイルを SHA256 照合 |
| 推論設定 | native horizon 50 / flow 10 / BF16 と F32 の混在 | state 条件は言語トークンへの離散化。追加量子化・compile・fusion は無効 |

モデルの所在は [固定した HF スナップショット](https://huggingface.co/mit-han-lab/vlash-pi05-libero-async5/tree/c85d5962a7b823f9f391f97d60227d6ca8bd7bc7)、実装は [作者コードの固定 revision](https://github.com/mit-han-lab/vlash/tree/d2da223e2251a4a5020b667e77cf8ad57e594144) を参照する。

## コードと重みのライセンスを分ける

作者コードは Apache-2.0。取得したモデルには README/model card がなく、`config.json` の `license` は null だった。**コードの Apache-2.0 を重みへ適用してはいけない**。PaliGemma/Gemma に由来する条件も別に確認する必要がある。再現用コードには重みや tokenizer バイナリを同梱せず、公開元・固定 revision・検証値を残す。

一次資料: [作者 LICENSE](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/LICENSE)、[公開 config](https://huggingface.co/mit-han-lab/vlash-pi05-libero-async5/blob/c85d5962a7b823f9f391f97d60227d6ca8bd7bc7/config.json)。

## 互換性の課題と対処

```text
task base: .venv-realtime (完全な140pin、torch2.7.1+cu128 / LeRobot0.6.2 / Transformers5.4)
                                      ↓ 読み取り専用の fallback
追加する:   .venv-vlash
             ├ LeRobot0.4.1
             ├ 作者指定 Transformers commit dcddb970... (4.53.3)
             ├ HF Hub0.34.4 / tokenizers0.21.4 / draccus0.10.0
             └ 固定された元の VLASH Policy
```

現行 Transformers5.4 では作者実装が要求する `_gated_residual` が import できなかった。モデル数式を推測で置き換えず、[作者 pyproject](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/pyproject.toml) に記載された旧 backend を別環境へ導入した。共有物理環境は従来のものを使い、VLASH 推論だけを別プロセスへ置く構成である。

公開版の再構築は `bash scripts/setup_vlash.sh` から開始できる。task base `.venv-realtime` が無ければ、独立した `scripts/setup_realtime_vla.sh` が完全な `requirements-realtime.lock` を構築する。別プロジェクトや別の基底venvは必要ない。`requirements-vlash.lock` は**このtask baseへ重ねる追加overlayのみ**を固定する。

測定に使った既存workspaceには、torchを別の既存site-packagesから読む歴史的構成があった。新bootstrapは `find_spec('torch')` で実際の配置だけを検出し、その場合に限って追加の読み取り先を作る。公開版の直接インストールでは `.venv-realtime` だけを読む。実験時の14sourceは先に `vlash-run-source-snapshot/` へ SHA一致で保存し、この変更は再現packagingだけである。runtime/evaluate/weightsと測定結果は変更していない。

```bash
# 共通環境と既存 ROS image の再構築手順は realtime-vla の guide を参照。
bash scripts/setup_vlash.sh
.venv-vlash/bin/python experiments/realtime_vla/download_vlash.py
.venv-vlash/bin/python experiments/realtime_vla/smoke_vlash.py --schema-only
.venv-vlash/bin/python experiments/realtime_vla/smoke_vlash.py
```

タスク評価では、別 terminal に `bash experiments/realtime_vla/run_ros_bridge.sh artifacts/realtime-vla/vlash-ros-system` を起動したうえで、新しい出力先を指定して実行する。GPU・meta 構築・大容量監査は他のモデルや実時間評価と重ねない。

```bash
# 作者の画像/state時刻操作を参照した独自H50同期診断。公式評価protocol完全再現ではない。
.venv-vlash/bin/python experiments/realtime_vla/evaluate_vlash.py \
  --mode artificial --condition current --artificial-delay-ticks 0 \
  --initial-states 0 --output artifacts/realtime-vla/development-vlash-artificial-d0
# 実時間3条件は coherent / current / predicted を同じ設定で各1回ずつ。
.venv-vlash/bin/python experiments/realtime_vla/evaluate_vlash.py \
  --mode realtime --condition coherent --initial-states 0 1 2 3 4 5 \
  --output artifacts/realtime-vla/formal-vlash-coherent
```

`init0` は development、`init1..5` だけを formal に集計する。全3条件で各 episode の held initial に同じ **timing calibration2** を行い、最後に RNG を reset して bootstrap を作る。最初の遅延推定はこの calibration2 だけから計算し、cold first warmup は混ぜない。元の事前 protocol を保全し、この修正・実行順・新 source SHA を `vlash-protocol-amendment-01.json` に記録した。

人工遅延 d0/d3 は **作者の画像/state時刻操作を参照した独自H50同期診断**である。[固定された作者の実行例](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/libero-eval-scripts/run.sh) は E5・500 episodes・batch32・8GPU・4 suites・seed42・d1..4。一方、こちらはチェックポイントの native H50 を E50 全実行、task0/init0・1環境・1GPU・seed20261102・14sim秒・実ROSを追加した条件である。公式ベンチマークの完全再現や、そのスコアとの直接比較は主張しない。差分は `vlash-reference-condition-notes.json` に固定した。

さらにソース確認で、robosuite の実測 quaternion は `w>=0` の表現を選ぶことを確認した。予測がπを跨いだ場合も、同じ物理姿勢を表す `q/-q` の符号だけをこの規約にそろえる。axis-angle state token の表現が実測とずれる問題を防ぎ、PD の数式や物理姿勢は変えない。π跨ぎの単体検査と事前 `amendment-02` に根拠・source SHA を記録した。

## 重みを確実に使うための復元設計

作者 `from_pretrained` は missing keys を許容していたため、その経路は使わない。新しい `vlash_runtime.py` は次の検査を実装し、**実重みを使った全段階が API smoke で通過した**。

1. 元コード・config・checkpoint の内容を照合する。
2. 元 constructor を meta device 上で作り、ランダムな大容量初期値の確保を避ける。
3. safetensors の 824 tensor と 255 の明示的な共有 alias で全 state key を埋める。共有は元モデルでも同一 Parameter であることを要求する。
4. strict load 後に元の共有を保持し、全 tensor の値を native dtype への変換後に byte 単位で比較する。未復元・余剰・形状不一致・残存 meta を拒否する。
5. 評価の前後で全 state の SHA256 を比較し、勾配対象 0・未学習を監査する。

1079 state keys（824保存 +255共有）を全値照合し、dtype cast 0・未復元 0・余剰 0・残存 meta 0 だった。チェックポイントに保存されない RoPE/position の決定論的 cache 21個は、元の作者/HF constructor と元の position arange 式だけで再生成した。学習済み値の代用は行っていない。

通常 pi0.5 の保存要素数との差 **74** は normalization の値で説明できる。state8 の mean/std が16、action7 の正規化・逆正規化が28、追加 ee6/joint7/gripper2 が30である。この差を無視して通常 pi0.5 へロードすると復元検査を弱めることになる。

## 入力と遅延の単位

| 項目 | 作者の実装に合わせる処理 |
|---|---|
| 画像 | raw RGB256×256 を 180°回転 **1 回**、BCHW/[0,1]、元 Policy が224へ resize/pad 後 [-1,1] |
| ロボット状態 | 実測 EEF XYZ + quaternion から axis-angle3 + finger2 の state8 |
| 正規化・prompt | 保存 mean/std → 作者の `torch.bucketize` → `Task: ..., State: ...;\nAction: `。`right=False` を含め元コードを維持 |
| tokenizer | Google が公開する同じ PaliGemma SentencePiece を元 converter で変換。全257152 IDs・6 prompt を確認する。FAST action tokenizer は使わない |
| 出力 | native 50×7 の relative EEF pose + gripper。クリップやシミュレータ step は別層 |

作者の学習 augmentation は `idx + delta + offset` であり、`max_delay_steps=4` は **dataset の sample index 単位**である。公開学習 YAML は dataset ID が placeholder のため、この checkpoint の fps や学習遅延秒数は確定できない。`async5` という名前だけで「20Hz の0～4制御 tickで学習済み」とは判断しない。

一方、[作者評価コード](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/vlash/eval_libero.py) の `async_delay` は環境ループ観測を buffer する人工遅延で、古い画像と現在の state を組み合わせる。学習 sample offset と評価 loop step は分けて記録する。今後の H50/R30 は **20行動を消費し、残量30で query** の意味である。

比較案は「画像と同時刻の state」「古い画像と query 時点の実測 state」「古い画像と推論完了時の予測 state」。予測は実測ロボットセンサと既知の制御列だけを使い、物体真値や未来の実測を入力へ渡さない。画像取得から query までの age、query から予測完了までの horizon、生成 chunk の初回実行時刻を別々に測る。action7 を state8 に単純加算すると単位・次元が一致しないため採用しない。

## 現段階の検証

- 元ソース35 files のサイズ・SHA256: 一致。
- 新 Python files の `py_compile`、setup shell の構文検査: 通過。
- torch・モデルをロードしない schema/alias/header/時間原点/予測器と既存coreの単体検査: **36 passed / 0.16 s**（新規18 +既存18）。
- 実 LIBERO init0 の保存済み raw センサを使った GPU API smoke: warm2 +測定3 が通過。モデル常駐 startup 34.14 s は別計測。
- warm 推論: 1110.03 / 426.00 ms。測定: 336.57 / 352.78 / 314.51 ms、平均 **334.62 ms**（model 部分平均333.04 ms）。native50×7/flow10。
- peak CUDA allocated: **7.211 GiB**。固有 Parameter bytes:7,472,805,000。全 state の before/after SHA一致、勾配対象0。
- 上記時間はシミュレータ step・ROS・タスク成功判定を含まない API 検証である。実タスク測定は次表に分ける。各系列の終了後はプロセスを閉じ、全終了時に GPU lease を返却した。

証跡: `artifacts/realtime-vla/models/vlash/{provenance,model-api,schema-smoke,api-smoke,restoration}.json`、`docs/reference/vlash-source-manifest.json`、`tests/test_vlash_{restoration,timing}.py`。

## 実タスク結果と得られた知見

| 実時間入力条件 | formal成功 | hold | RTF中央値 | query間隔のepisode平均 | 成功時の実時間中央値 |
|---|---:|---:|---:|---:|---:|
| 画像と同時刻のstate（coherent） | 3/5 | 0% | 1.007 | 18.21 ticks | 4.07s |
| 古い画像＋query時の実測state（current） | 1/5 | 0% | 1.003 | 20.08 ticks | 4.12s |
| 古い画像＋完了見込み時刻の予測state（predicted） | 2/5 | 0% | 1.003 | 30.46 ticks | 11.04s |

H50同期診断の d0/d3 はそれぞれ init0 成功（101/104 ticks）。実時間の dev0 は全3条件で未達だった。devの成否で設定やformal初期状態を選び直していない。formalの同じ初期状態について、入力5センサの全SHAが3条件で一致した。

同じ H50/R30 でも、stateが表す原点を正しく使うと、expired prefixと再推論の間隔も変わる。したがって上表は **入力state・原点・query頻度を含むsystem条件の比較**であり、「古いstate自体が優れる」という単独の因果は主張できない。待機の解消は達成できたが、今回の単純予測で coherent を上回る成功率は確認できなかった。

predictedの位置誤差はepisode平均の平均22.06mm、pooled中央値19.81mm/p95 39.64mm。finger位置の誤差平均は11.50mmだった。接触・torque飽和・全身ダイナミクスを含まない予測である。学習時のfpsが公開されていないため、遅延範囲の外挿についても条件付きで扱う。

切り分けで、init5/tick178のsensor RPCが875.3msに延びた一方、DDS往復は5.23ms、モデルは316.4msだった。init2にもRPC646.3ms/DDS4.98msの外れ値があり、p95推定horizonは一部17–20ticksになった。モデルだけの速度ではこの尾部遅延を説明できない。future原点より早着したchunkは旧queueを継続して待つため、formalのpending待ちは計133ticksだったが、物理停止やempty holdとは区別している。

最初の実タスク起動は、`.pth` でpackage importは成功しても `sys.prefix` 固定のasset配置が存在しない問題で、最初のtask tick前に停止した。共有ソースを変えず、専用overlayのpackage aliasだけを追加し、585assets/422,313,213bytesを全SHA照合して再実行した。このstartup失敗は成功率の分母に入れない。**import/API成功だけでは、シミュレータのasset配置まで検証したことにならない**という知見である。

全20 task episodeの計243モデル推論（warm/calibration含む）・486実ROS交換を独立nodeログと照合した。全1079stateの値復元とbefore/after不変、勾配対象0、旧45試行の243 JSON/JSONL不変を確認。結果と監査は `vlash-comparison-summary.json` と `vlash-report/` に保存した。
