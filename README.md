# PAI最終課題_44055：SmolVLAとπ0.5による実時間アーム操作

Pandaアームが黒いボウルをつかみ、指定された皿へ置くLIBEROタスクを実行する。公開済みのSmolVLAとπ0.5を用い、画像・言語指示からの行動生成、ROS 2通信、シミュレータ内の制御を接続した。追加学習は行わず、推論中もシミュレーションを進める非同期実行を評価する。

比較の出発点は、小型のSmolVLAの低遅延が実時間操作に有利か、という問いだった。同期・非同期のタスク評価に加え、推論の処理区間と行動列の接続方法を調べた。

## 提出資料と実験範囲

- [説明資料PDF](artifacts/realtime-vla/submission-smol-pi05/説明資料_PAI最終課題_44055.pdf)：目的、構成、工夫、主比較、推論時間、RTC等の結果。
- [主比較20試行の再計算・出典](artifacts/realtime-vla/submission-smol-pi05/evidence-primary.json)：SmolVLAとπ0.5、同期・非同期、各5配置。
- [接続方法・推論時間の出典](artifacts/realtime-vla/submission-smol-pi05/evidence-continuity.json)：RTCの正式30試行、VLASH・A2C2の参考検証、固定観測による時間計測。
- [タスク動画の構成・時刻対応](artifacts/realtime-vla/submission-smol-pi05/video/video-manifest.json)：91.6秒のSmolVLA・π0.5比較映像。MP4は別途配布。
- 使用コード：このリポジトリの`experiments/realtime_vla/`、環境構築・取得用の`scripts/`、固定依存ファイル。

主比較、RTC、固定観測による時間計測は条件が異なるため、試行数や成功率を合算しない。初期のπ0-FASTを含む35試行は[従来の実験記録](docs/experiments/realtime-vla.md)と[結果JSON](artifacts/realtime-vla/report/results.json)に保持している。旧版の動画・初版資料はその実験範囲の記録として残している。新しい91.6秒動画は同期・非同期・RTCの10試行例であり、全試行を収録したものではない。

## タスクとシステム構成

| 項目 | 内容 |
| --- | --- |
| タスク・環境 | LIBERO Spatial task0、Panda、robosuite／MuJoCo |
| 入力 | 外部・手首RGB各256×256、手先位置・姿勢・グリッパ状態、言語指示 |
| 方策 | LIBERO向け学習済みSmolVLA／π0.5。重みを固定して推論 |
| 出力 | 相対位置3・姿勢3・グリッパ1の7次元行動列 |
| 低位制御 | ROSで受信した行動 → 既存OSC → 関節トルク → シミュレータの接触物理 |
| 通信 | Sensor/Broker・PolicyTransportの2ノードでセンサと行動列を送受信 |
| 評価 | LIBERO公式成功判定。物体の正解位置・成功フラグはモデル入力に含めない |

### 非同期化と資源管理の工夫

- **制御・撮像・推論を別プロセスへ分離した。** 同期2カメラの平均描画時間85.64msが、20Hz制御の1周期50msを超えたため、撮像を制御ループから外した。
- 画像と自己状態を同じ取得時点へそろえた。時刻付きのシミュレータ状態を別のカメラシーンへ復元し、元の取得時刻を保持する。撮像用の待ち行列は長さ1とし、古い状態をためない。
- ROSで受信した値をモデルと制御器へ渡した。センサ・行動の送信値と両ノードの受信値をhashで照合し、通信後のデータを使用したことを確認した。
- 行動が届かない周期もシミュレータを更新した。行動待ち行列が空なら位置・姿勢の差分をゼロにし、グリッパ指令を維持する。最終の相対移動指令を反復して停止の代わりにすることは避けた。
- RTX 5080 16GBへモデルを1種類ずつロードした。推論側CPU4スレッド、撮像側BLAS1スレッド、同時推論1要求とし、追加のRTC系列ではGPU使用時間の調整と0.5秒間隔の資源監視を行った。

## 主比較：同期・非同期の20試行

両モデルとも50行動をflow反復10回で生成し、先頭10行動を実行する。行動列を使い切ってから次の推論を要求し、先読みは行わない（E10/P0）。各方式で同じ配置1〜5、seed、ROS通信を使用した。

| モデル | 実行方式 | 公式成功 | 方策生成・出力処理 | 生成＋ROS | 待機割合 | 実測制御頻度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolVLA | 同期（lockstep） | 5/5 | 578ms | 615ms | — | 7.13Hz |
| π0.5 | 同期（lockstep） | 5/5 | 335ms | 372ms | — | 7.81Hz |
| SmolVLA | 非同期（realtime） | 3/5 | 508ms | 604ms | 53.2% | 19.92Hz |
| π0.5 | 非同期（realtime） | 5/5 | 332ms | 412ms | 44.5% | 19.98Hz |

同期実行では推論・描画中にシミュレーションを止め、最大280制御ステップで評価する。これは14シミュレーション秒相当であり、14実秒以内の完了を意味しない。非同期実行では推論を待たず、最大14実秒または280ステップで評価する。同期の推論待ちは制御tickに現れないため、待機割合を「—」とした。

表の処理時間は初回行動列を除く各試行平均の5試行平均で、終了後に完成した要求も含む。方策生成・出力処理は、入力構築と保存preprocessorの後から、方策生成・保存postprocessor・CUDA同期までの実経過時間であり、GPU演算だけの時間ではない。生成＋ROSはセンサROS、モデル、行動ROSを含み、親プロセスとのIPC・poll待ちは含まない。待機割合は全5試行の待機tick合計を全tick合計で割った値である。

同期は両者5/5で、非同期の成功数はSmolVLAが3/5、π0.5が5/5だった。同じ方式内のモデル間では初期センサが一致したが、同期と非同期では撮像の更新位相も異なる。したがって、成功数の差をモデル遅延だけの効果とは解釈しない。対象は既知の1タスク・固定5配置である。

## 固定観測による推論時間の分解

タスク実行時の並列負荷を外すため、保存済みの同じ5観測から推論した。各モデルで「区間計測なし→区間計測あり→区間計測なし」の順に実行し、各ブロックの準備運転10回を除いた25要求を測定した。重みは主比較と同一で、計測有無による出力も完全一致した。

| 入力前処理込み・計測処理なしの呼出し全体 | SmolVLA | π0.5 |
| --- | ---: | ---: |
| 前後50要求の平均 | 603.6ms | 367.9ms |

| 内部区間（計測処理あり、25要求平均） | SmolVLA | π0.5 |
| --- | ---: | ---: |
| 画像エンコーダ | 18.2ms | 18.9ms |
| 観測理解（VLM prefix） | 58.6ms | 55.4ms |
| 行動生成（flow全体） | 497.6ms | 231.0ms |

内部区間はCUDA stream上の経過時間で、CPUからの実行指示の間隔や待ちを含み得る。GPUが演算していた時間だけを示す値ではない。上の2表は別の計測条件である。呼出し全体には入力構築・保存preprocessorも含み、主比較の方策生成・出力処理とは計測範囲が異なる。タスク実行中の並列負荷も異なるため、それぞれを分けて扱う。

大きな差はflow行動生成に現れた。画像解像度は512×512対224×224だが、画像エンコーダの時間は近い値だった。行動expertは32層対18層で、幅やattention方式も異なる。層数・実装差の個別の寄与を測る対照は行っておらず、パラメータ数や画像解像度のどちらか一つを原因と断定しない。

前後の平均時間はSmolVLAで6.3%、π0.5で4.2%変動した。π0.5では計測ありの時間が対照より短く、差をそのまま計測処理の負荷とはみなせない。測定手順、実行精度、元ログ、集計定義は[推論時間の実験記録](docs/experiments/realtime-inference-profiling.md)と[集計JSON](artifacts/realtime-vla/inference-profile-report/summary.json)に保存した。π0-FASTは対照同士でも出力が変わったため、確定した区間比較から除外している。

## RTC：行動列の接続方法を比べる30試行

行動列の切替時に古い動作と新しい動作が食い違う可能性があるため、生成中の行動列を既知の実行予定へ近づけるRTCを試した。50行動を保持し、残り30行動で次を要求し、到着時には経過した部分を除いて接続する（H50/R30）。同じスケジューラ内で、補正なし、公開LeRobot RTC、入力微分を有効にした完全VJP版を各5配置で比較した。RTCで微分する対象は行動変数であり、モデルの重みは更新しない。

| モデル | 接続方法 | 公式成功 | 生成＋ROS平均 | 待機割合 |
| --- | --- | ---: | ---: | ---: |
| SmolVLA | 補正なし | 1/5 | 615ms | 0.25% |
| SmolVLA | 公開LeRobot RTC | 1/5 | 620ms | 0.00% |
| SmolVLA | 完全VJP RTC | 1/5 | 1356ms | 22.35% |
| π0.5 | 補正なし | 5/5 | 367ms | 0.00% |
| π0.5 | 公開LeRobot RTC | 4/5 | 455ms | 0.00% |
| π0.5 | 完全VJP RTC | 4/5 | 890ms | 0.00% |

π0.5はH50/R30の補正なしで待機0%・5/5成功した。一方、RTCによる成功数の改善は確認できず、完全VJPでは生成時間が増えた。SmolVLAの公開RTCでは切替時の旧予定指令と新指令の差が小さくなったが、成功は1/5のままだった。接続指令の一致だけでは操作完了を判断できない。

主比較のE10/P0とH50/R30では実行長、先読み、時刻整列が異なるため、両系列の差をRTCだけの効果とは扱わない。最初に行ったGPU共有状態の系列は参考記録に残し、この表には使用時間の調整・監視を加えた`continuity-clean`系列だけを用いた。

### VLASH・A2C2の参考検証

| 検証 | 同じ基盤モデル内の条件 | 公式成功 |
| --- | --- | ---: |
| VLASH専用π0.5 | 画像と同時点／現在の実測／未来状態予測 | 3/5／1/5／2/5 |
| A2C2作者450M SmolVLA | headなし／残差解釈／直接行動解釈 | 4/5／1/5／0/5 |

VLASHには専用の学習済み重みを、A2C2には主比較と異なる450M基盤モデルを用いた。A2C2の公開headには学習コードと出力意味の対応が未確定な部分がある。これらを通常π0.5への同一条件の追加や、手法全体の一般的な順位とは扱わない。詳細は[接続方法の実験記録](docs/experiments/realtime-continuity.md)に記載した。

## 環境構築

検証環境はUbuntu 24.04／WSL2、CPython 3.12、RTX 5080 16GB。Linux x86_64、CUDA 12.8対応ドライバ、Docker、初回取得用ネットワークを使用する。

~~~bash
bash scripts/setup_realtime_vla.sh
.venv-realtime/bin/python experiments/realtime_vla/download_smolvla.py
.venv-realtime/bin/python experiments/realtime_vla/download_pi05.py
docker build -t physical-ai-ros2:jazzy-libero-vla experiments/realtime_vla
~~~

setupは単独の`.venv-realtime`を作成し、固定したPython依存とLIBEROアセットを導入する。他の実験のvirtualenvや`.pth`による代替参照は使用しない。アセットmanifestだけが存在する場合も、実ファイルを確認して不足分を取得する。重みはSmolVLAが約1.22GB、π0.5が約7.47GB、LIBEROアセットは約422MBである。

`requirements-realtime.lock`には検証済みruntimeの固定版を列挙している。setupは`--no-deps`でこの集合を導入し、LeRobotとLIBERO等のimportを確認する。モデル重みは別の取得スクリプトが固定revision・容量・digestを検証する。日本語の動画ラベルを新しく生成する場合は、Noto Sans CJK等の日本語フォントも使用する。

## 主比較の再実行

別端末でROSのSensor/Broker・PolicyTransportを起動する。

~~~bash
export REALTIME_ROS_IMAGE=physical-ai-ros2:jazzy-libero-vla
bash experiments/realtime_vla/run_ros_bridge.sh artifacts/realtime-vla/reproduction/ros
~~~

評価側では[再現ガイド](docs/guides/realtime-vla.md)の描画設定を適用する。今回のWSL2/NVIDIA環境では次の設定を使用した。native LinuxではEGL driverに合わせて変更する。

~~~bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"

.venv-realtime/bin/python experiments/realtime_vla/evaluate.py \
  --model smolvla --initial-states 1 2 3 4 5 --conditions lockstep realtime \
  --control-hz 20 --wall-seconds 14 --execute-steps 10 --prefetch-steps 0 \
  --merge fifo --seed 20261102 --camera-hz 10 --max-sensor-age .5 \
  --ros-transport --video --output artifacts/realtime-vla/reproduction/smolvla-e10

.venv-realtime/bin/python experiments/realtime_vla/evaluate.py \
  --model pi05 --initial-states 1 2 3 4 5 --conditions lockstep realtime \
  --control-hz 20 --wall-seconds 14 --execute-steps 10 --prefetch-steps 0 \
  --merge fifo --seed 20261102 --camera-hz 10 --max-sensor-age .5 \
  --ros-transport --video --output artifacts/realtime-vla/reproduction/pi05-e10
~~~

2つの評価は順に実行し、同じGPUへ別モデルを同時にロードしない。初期化・準備運転2回・初回行動列の生成はタスク計時前に行い、時間を別に記録する。出力先には新しい名前を指定し、保存済みの証拠を上書きしない。

### RTCの再実行

ROSブリッジを起動した状態で、次を順に実行する。

~~~bash
bash experiments/realtime_vla/run_continuity_clean_series.sh \
  smolvla artifacts/realtime-vla/reproduction-rtc
bash experiments/realtime_vla/run_continuity_clean_series.sh \
  pi05 artifacts/realtime-vla/reproduction-rtc
~~~

各モデル・各方式は別プロセスで配置0〜5を実行する。配置0は動作確認用で、正式集計には1〜5だけを含める。モデル・方式・GPU利用を調整した[事前条件](artifacts/realtime-vla/continuity-clean-protocol.json)と、詳細な[実行・集計手順](docs/experiments/realtime-continuity.md)を参照できる。

### 固定観測による時間計測の再実行

ROSブリッジやタスク評価を終了し、GPUを他の推論と共有しない状態で保存観測を取得する。その後、モデルを1種類ずつ計測する。

~~~bash
.venv-realtime/bin/python -m experiments.realtime_vla.capture_profile_inputs \
  --output artifacts/realtime-vla/reproduction-profile/observations \
  --initial-states 1 2 3 4 5 --seed 20261102 --resolution 256

for model in smolvla pi05; do
  .venv-realtime/bin/python -m experiments.realtime_vla.profile_inference \
    --model "$model" \
    --observations artifacts/realtime-vla/reproduction-profile/observations \
    --output "artifacts/realtime-vla/reproduction-profile/$model-native" \
    --warmup 10 --repeats 5 --seed 20261102 --threads 4 --interop-threads 1
done
~~~

各出力先の`report.json`に、要求ごとの時間、出力照合、重み監査、処理区間の集計を保存する。この2モデルの個別計測と、FASTも含む3モデル一括スクリプトは区別する。

~~~bash
.venv-realtime/bin/python -m pytest -q tests experiments/realtime_vla
.venv-realtime/bin/python experiments/realtime_vla/summarize.py --help
.venv-realtime/bin/python experiments/realtime_vla/make_demo.py --help
~~~

## コード・出典・再現性の記録

| 場所 | 内容 |
| --- | --- |
| `experiments/realtime_vla/` | 凍結方策、シミュレータ・カメラ・推論、ROS、待ち行列、評価・録画 |
| `scripts/setup_realtime_vla.sh` | 単独runtime構築、固定アセット取得、import確認 |
| `requirements-realtime.in`／`.lock` | 検証済み依存の固定版 |
| [新提出資料と主比較の出典](artifacts/realtime-vla/submission-smol-pi05/) | PDF、20試行と追加検証の値・出典field・SHA |
| [システム構成](docs/architecture/realtime-vla.md) | センサからアクチュエータまでの構成 |
| [推論時間の記録](docs/experiments/realtime-inference-profiling.md) | 計時範囲・精度・区間分解・変動・除外条件 |
| [接続方法の記録](docs/experiments/realtime-continuity.md) | RTC・VLASH・A2C2の条件と試行錯誤 |
| [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json) | 提出コード各ファイルのSHA256 |
| [配布時のパス置換記録](artifacts/realtime-vla/inference-profile-distribution.json) | 置換前後の原本・配布版SHA256 |

モデル・シミュレータ・タスクには既存資産を利用した。追加した範囲は、非同期実行、観測と行動の時刻管理、ROSで受信したデータの利用、比較と監査である。別環境で同じコードを実行しても、OS・GPU・描画・推論完了時刻の違いにより、同一の遅延や非同期軌跡になる保証はない。

## 公開資産の利用条件

コード、モデル重み、tokenizer、シミュレータの形状データは利用条件を分けて確認する。π0.5の選択重みにはGemmaの条件があり、モデル名や実装コードのライセンスからMIT／Apacheと読み替えない。VLASHで選択した重みの明示ライセンスは確認できていない。

外部重み・tokenizer本体・mesh・個人の認証情報はリポジトリに含めず、固定revisionから取得する。第三者実装のnoticeは[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)と`LICENSES/`に保持した。[主な公開資産](docs/reference/realtime-vla-assets.md)、[π0.5の復元と利用条件](docs/reference/pi05-libero-assets.md)、[接続方式の資産](docs/reference/realtime-continuity-methods.md)に出典と確認範囲を記録している。
