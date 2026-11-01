# 固定観測によるVLA推論時間の分解

同じ保存済み観測をSmolVLA、π0.5、π0-FASTへ与え、この環境でのnative推論時間を処理段階ごとに調べる独立実験。[既存の実時間VLA実験](realtime-vla.md)と[追加の接続手法評価](realtime-continuity.md)にある成功試行数には加えない。

SmolVLAとπ0.5は測定・出力照合・重み不変・計時整合性を確認した。今回の総時間差が大きく現れた箇所はflowの行動生成だった。π0-FASTは出力の対応照合に失敗したため、正式なstage比較から除外し、参考記録を保持する。事前条件は[inference-profile-protocol.json](../../artifacts/realtime-vla/inference-profile-protocol.json)を正本とする。

## 問いと比較の範囲

以前の実行で観測されたπ0.5の短い推論時間について、画像処理、視覚encoder、VLM prefixの計算、flowまたは自己回帰生成、出力処理のどこに時間が使われるかを確認する。パラメータ数、保存dtype、ある一つの実装差だけから主因を決めない。

対象は保存済み初期観測に対する独立した推論要求。行動列をシミュレータへ適用せず、物理・撮像・ROS・制御待ち行列を推論測定中に実行しない。操作成功率、途中の場面に対する平均性能、実時間制御の改善を測る実験ではない。

## 完了したSmolVLA・π0.5の測定

各行は25要求のfull-chunk wall elapsed。単位はms、warmupを除く。両モデルともH50・flow10のまま生成した。

| モデル | block | 平均 | 中央値 | p95 |
| --- | --- | ---: | ---: | ---: |
| SmolVLA | baseline before | 585.06 | 581.80 | 651.30 |
| SmolVLA | instrumented | 637.73 | 634.07 | 694.90 |
| SmolVLA | baseline after | 622.09 | 613.12 | 678.49 |
| π0.5 | baseline before | 360.30 | 357.29 | 409.13 |
| π0.5 | instrumented | 341.74 | 346.40 | 380.88 |
| π0.5 | baseline after | 375.49 | 372.14 | 420.87 |

参考として、前後のuninstrumented blockを連結した50要求の平均はSmolVLA **603.57ms**、π0.5 **367.90ms**。この連結値は今回の要求集合の記述値であり、前後の変動を消した推定値ではない。上表のblockを主比較として保持する。

対応要求ごとの`instrumented / bracket baseline`の平均はSmolVLA 1.059、π0.5 0.930。後者を「負のinstrumentation overhead」と解釈しない。前後baseline比の対応要求ごとの中央値はSmolVLA +6.2%、π0.5 +5.9%であり、block間に時間変動がある。平均同士の比で表す前後差はそれぞれ+6.3%、+4.2%。これらは異なる統計なので区別する。5〜10%程度の小差だけでは特定の処理を速度差の原因と断定しない。

### 大きな差はvisionよりflow生成に現れた

次はinstrumented 25要求での平均CUDA stream elapsed。1要求中の同名stage呼出しを合計したmsであり、kernel busy timeではない。

| stage | 集計の意味 | SmolVLA | π0.5 |
| --- | --- | ---: | ---: |
| request root | inclusive | 583.12 | 311.96 |
| vision tower | exclusive、2回/1回の要求内合計 | 18.17 | 18.94 |
| VLM prefix prefill | exclusive、1回 | 58.61 | 55.41 |
| flow denoise | inclusive、10回の要求内合計 | 497.57 | 230.97 |
| flow action expert forward | exclusive、10回の要求内合計 | 484.93 | 205.81 |

flow denoiseにはaction expert forwardが含まれるため、この2行を加算しない。expert forwardのexclusiveは、root stream elapsedに対してSmolVLA 83.2%、π0.5 66.0%を占めた。host側でもexpert forwardがそれぞれ528.60ms、223.46msで大きい。今回の時間差を局所化する証拠はflow/action expertにあり、512×512画像のvision towerを主因とする説明は支持されなかった。visionとprefillの小差を個別の優劣として扱わない。

SmolVLAは登録parameter約0.605B、π0.5は約4.143Bである。一方、今回復元したexpertの層数は32対18、代表的なexpert projection入力は`[1,50,480]`対`[1,50,1024]`。層数、幅、attention経路、nativeソース、画像/token shapeと精度が同時に異なる。総parameter数だけではこの時間差を説明できず、32層対18層も実装上の背景であり、それ単独の因果効果を測った対照ではない。

保存parameterはSmolVLAがFP32 149.570M要素/BF16 455.364M要素、π0.5がFP32 533.551M要素/BF16 3609.854M要素。別probeでは両モデルのvision attentionの`q_proj`出力がBF16で、visionにBF16 autocastが有効だった。SmolVLAを「全てFP32で計算した」とは解釈しない。vision入力の代表shapeはSmolVLAが`[1,1024,768]`を2回、π0.5が`[3,256,1152]`を1回。後者は2画像と空camera slotのbatchである。保存dtype、activation dtype、実際の呼出しbatchは別の情報として読む。

root host elapsedとCUDA stream elapsedの差は、SmolVLAで約54.57ms、π0.5で約29.75msだった。この差を純粋なCPUコスト、hook overhead、GPU idle時間と名付けない。hostとCUDA eventは別の時計・範囲を測り、非同期launchや待ちがあるためである。対応baselineから計算するinstrumentedの増減とも別の量である。

### π0-FASTは出力対応不成立の参考記録

元の[FAST report.json](../../artifacts/realtime-vla/inference-profile-v1/pi0fast-native/report.json)は`all_paired_outputs_exact=false`、`all_paired_outputs_allclose=false`のまま保持する。同じsample/repeat/seedでも、uninstrumented beforeとafterの行動hashが22/25組で異なり、instrumentedとbeforeでは23/25組が異なった。instrumentationだけによる変化と切り分けられておらず、3モデルの正式な対応比較・速度原因の結論へ含めない。

| FAST参考値 | 平均wall elapsed |
| --- | ---: |
| baseline before、25要求 | 851.03ms |
| instrumented、25要求 | 849.44ms |
| baseline after、25要求 | 869.44ms |
| uninstrumented前後連結、50要求 | 860.24ms |

有限なH10×7出力、重み不変、nested accountingは成立したが、出力対応の失敗を打ち消すものではない。JSON/CSVには除外理由を付けて残し、primary図はSmolVLAとπ0.5だけを描く。greedy経路の変動原因は未確定であり、backend差などの仮説を原因と断定しない。

SmolVLA・π0.5の出典：[SmolVLA report.json](../../artifacts/realtime-vla/inference-profile-v1/smolvla-native/report.json)、[π0.5 report.json](../../artifacts/realtime-vla/inference-profile-v1/pi05-native/report.json)。集計値と描画値は[summary.json](../../artifacts/realtime-vla/inference-profile-report/summary.json)と[plot-data.json](../../artifacts/realtime-vla/inference-profile-report/plot-data.json)にsource SHAとともに保存する。

## 共通入力とnative条件

LIBERO Spatial task0の公式初期状態1〜5、seed `20261102`、20Hz、settle 10 stepを使用する。物理sceneをresetし、snapshotを別の同期camera sceneへ復元してRGBと自己状態を得る。これは既存CameraWorkerと同じ `observe_snapshot` の経路。直接resetから返るcached観測とは撮像位相が異なり得るため、以前のbootstrapと一致したかはcapture manifestのセンサhash比較で記録する。

今回のcaptureは120.74秒、rendererは`D3D12 (NVIDIA GeForce RTX 5080)`。5初期状態の全5センサhashが以前の正式SmolVLA realtime bootstrapと完全一致した。sceneを閉じてcaptureを終了した後に推論測定を開始した。出典：[capture manifest](../../artifacts/realtime-vla/inference-profile-v1/observations/manifest.json)。

保存NPZには未回転の外部・手首RGB各`256×256×3 uint8`、手先位置3値、XYZW quaternion 4値、gripper 2値だけを格納する。自己状態の元の浮動小数dtypeを保存し、各モデルの公式前処理が変換する。物体の正解位置、成功フラグ、物理snapshotはモデル入力に含めない。言語指示はmanifestに分けて保存する。各NPZと各センサのSHA256を全consumerで照合し、captureプロセスを終了してからモデルをロードする。

| native条件 | SmolVLA | π0.5 | π0-FAST |
| --- | --- | --- | --- |
| 生成する行動列 | H50 | H50 | H10 |
| 生成方式 | flow 10 step | flow 10 step | greedy自己回帰、KV cache、上限256 token |
| 方策内の画像解像度 | 512×512 | 224×224 | 224×224 |
| 空camera slot | 0 | 1 | 1 |

H50/H50/H10はnativeの出力全体を表す。総時間を行動数で割って同じ仕事量へ正規化しない。SmolVLAの `n_action_steps=1` は、ここで呼ぶ `predict_action_chunk` がH1を生成することを意味しない。実際に返った`action_shape`を各要求に記録する。

nativeの重み・processor・精度・内部解像度・生成step・行動horizon・依存ソースを保持する。追加学習、量子化、`torch.compile`、モデルのソース編集を行わない。RTX5080 16GBで他の実験とGPU使用時間を調整し、一度に1モデルをロードする。torch intraop 4、interop target 1、OpenBLAS 1 threadを事前条件とし、実際に反映された値をJSONへ残す。

## 対照と要求順序

各モデルで次の3blockを同じ観測順・乱数scheduleで実行する。

1. `baseline_before`: wrapperなしのnative推論。
2. `instrumented`: native境界へ一時的なwrapperとCUDA eventを設置。
3. `baseline_after`: wrapperを復元したnative推論。

各blockの直前に10回warmupし、その後5観測×5 repeatsの25要求を測る。合計75測定要求/モデル。代表的なactivation精度を調べる別のprobeはwarmup・測定回数から除外する。乱数reset、入力arrayのcopy、行動hash計算、precision probe、重み監査は要求の計時外で行う。

追加のCUDA同期は要求の外側境界だけで行い、stageごとの同期を追加しない。元のnative runtime内の同期やCPU出力変換は保持する。各要求の`sample_id`、repeat、seed、センサhashを3blockで照合し、返る行動を完全一致と `rtol=1e-5, atol=1e-6` の両方で確認する。

同じ要求の前後baselineの算術平均をbracket baselineとし、instrumentedとの差と比率を記録する。前後baselineの差は温度・clock・cache等を含む時間変動の手がかりであり、wrapperのoverheadと同一視しない。stage分解はinstrumented実行の値なので、両baselineから離れていないかを併記する。

## 時間の定義と集計

| 指標 | 意味 | 集計上の注意 |
| --- | --- | --- |
| request wall elapsed | native `predict` の呼出しと終了側CUDA同期を含む実経過時間、ms | before/instrumented/afterを別集計 |
| host inclusive | stage開始から終了までのhost経過時間、ms | CPU計算、kernel launch、暗黙の待ちを含む。親子は重なる |
| host exclusive | host inclusiveから直下の子intervalだけを差し引いた残り、ms | 純粋なCPU計算量ではない |
| CUDA event inclusive | 同じCUDA stream上のevent間経過時間、ms | hostのlaunch間隔やidle gapを含み得る。kernel busy timeではない |
| CUDA event exclusive | CUDA event inclusiveから直下の子event区間だけを差し引いた残り、ms | 別streamの並列処理を扱う指標ではない |
| invocation count | 1要求中の同じstageの実呼出し数 | flow step、視覚batch、FAST decode回数を区別 |

同一stage名が1要求中に複数回現れる場合、先にその要求内で時間を合計し、25要求の分布を要約する。表の「stage平均」は個々のdenoise stepやtoken forward 1回の平均時間とは異なる。stageが実行されない要求ではそのstageの合計と呼出し数を0とする。

統計は`count, mean, median, p95, min, max`。p95は昇順データの位置`0.95×(n−1)`における線形補間値。warmupを含めず、主解析はblock別・モデル別に集計する。参考の`baseline_pooled_reference`だけはbefore/afterの50要求を連結した記述値として明示する。反復は同じ5初期観測から得ており、25独立sceneの母集団推定や信頼区間として扱わない。

inclusive stage同士は加算しない。各intervalのexclusiveを合計した値がroot inclusiveに一致するかをhostとCUDAで別々に確認する。builderもstage集計からこのpartitionを再計算する。hostとCUDAの時間も互いに足し合わせない。

保存postprocessorにはCPUへのdevice移動と先行GPU処理の待ちが入る場合がある。rootのexclusive残差にはraw変換、同期、最終cast/NumPy view、検証、wrapper overheadが残る。これらの時間をそのまま「D2H転送時間」と命名しない。

## 実装・精度の証拠

source JSONへnative checkpoint/config、processorのhash、実行した依存ファイルのSHA256、library版、module構造、登録parameter数、保存parameterのdtype分布・bytesを残す。登録parameter数には推論で使わないheadや共有表現の影響が含まれ得るため、推論演算量の代用にしない。

別のprecision probeで、代表的なvision/language/expertの`q_proj`入出力dtypeと、その時点のCUDA autocast設定を調べる。保存FP32とactivation BF16が併存し得る。代表点の出力dtypeだけから、全kernelの積和精度を断言しない。stage内のtensor shape、実呼出し数、FASTの実生成token数も合わせて読む。

重みdigestの前後一致、勾配を持つparameterがないこと、eval状態、出力照合、nested accountingを実行ごとに監査する。GPU全体とhostのresource traceはrootが連続記録する。推論前のmodel復元を含むprocess peak RSS/CUDA peakと、測定blockの常駐量は同じ指標ではない。

## 任意のπ0.5 camera対照

nativeのcamera一括embeddingとcameraごとのembeddingを比較する場合は、同じπ0.5の重み、精度、入力、processor、H50、flow10、依存ソースを保持した別条件として扱う。`pi05-per-camera/report.json`または明示した `--ablation-report` のみを別表へ追加し、3モデルnativeの図へ混ぜない。

この対照は今回実行していない。nativeのvision時間が約18〜19msだったことは確認したが、camera batchingを変更した対照の時間差・出力差は得ていない。camera batchingを総時間差の主因とはしない。タスク成功への効果もこの実験では測らない。

## 成果物と再集計

入力は `artifacts/realtime-vla/inference-profile-v1/` の次のJSONに限定する。

- `smolvla-native/report.json`
- `pi05-native/report.json`
- `pi0fast-native/report.json`

集計はCPUだけで次を実行する。出力先は新規directoryが必要。

```bash
.venv-realtime/bin/python scripts/build_inference_profile_report.py \
  --input-dir artifacts/realtime-vla/inference-profile-v1 \
  --output artifacts/realtime-vla/inference-profile-report
```

builderはモデル・観測NPZをロードしない。`summary.json`に再計算した統計と元の実装・precision metadata、`totals.csv`と`stages.csv`に単位付き表、`plot-data.json`に描画した数値とsource SHAを保存する。`native-fullchunk.png/svg`は適格モデルの3blockの平均・中央値・p95、`stage-host.png/svg`と`stage-cuda-stream.png/svg`は適格モデル別の包含/排他的時間を別panelで示す。今回のprimary図からはFASTを除外する。stage図の軸範囲はpanelごとに異なるため、棒の見かけの長さだけでpanel間比較しない。

モデル間のmanifest/sample provenance一致とsource内のvalidity flagsを別途表示する。今回のbundleはFAST除外を伴う`partial_validated`で、CSVにも`paired_comparison_eligible=false`と除外理由を明示する。source JSONのvalidation値は変更しない。結果の解釈では実測の時間割合、wrapper overhead、baseline drift、実精度、画像/token shape、native実装をまとめて確認する。この構成・GPU・依存版での観測を、恒常的なモデル固有の速度順位へ一般化しない。

旧35試行・追加70試行の集計、レビュー用PDF、README、およびreadability variantは変更しない。

## 配布版のパス置換

配布するログ・集計JSONでは、参加者のホームディレクトリを置換している。数値・出力照合結果・重み監査結果は変えない。ログに保存されたsource SHAは置換前の原本を指すため、配布ファイル自身のSHAとは異なる場合がある。詳細証拠ZIPの`INFERENCE-PROFILE-MANIFEST.json`、Gitの`inference-profile-distribution.json`に原本と配布版のSHAを併記する。観測・行動NPZとresource traceは別配布の証拠ZIPに含み、モデル重みは含まない。
