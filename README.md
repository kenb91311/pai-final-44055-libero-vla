# PAI最終課題_44055：実時間VLAによるボウル移載

Pandaアームが、皿とラメキンの間にある黒いボウルをつかみ、皿へ置く。
LIBERO向け学習済みSmolVLAとπ0-FASTを使い、画像認識から行動列生成、ROS 2通信、
物理制御までを接続した。追加学習は行わず、推論中にも物理が進む条件で評価する。

## タスクとAIの役割

| 項目 | 内容 |
|---|---|
| 環境 | LIBERO Spatial task0、Panda、robosuite／MuJoCo |
| 観測 | 外部・手首RGB256×256、手先位置・姿勢・グリッパ状態、言語指示 |
| 方策 | 公開されたLIBERO適応済みSmolVLA／π0-FAST |
| 出力 | 相対位置3・姿勢3・グリッパ1の7次元行動列 |
| 制御 | ROSで受信した行動 → 既存OSC → 関節トルク → 接触物理 |
| 判定 | LIBERO公式成功判定。物体の真値位置は評価器だけが使う |

## 設計と工夫

- **物理・撮像・推論を別プロセスへ分離。** 同期2カメラの平均85.64msが、
  20Hz制御の予算50msを超えたため、撮像を物理ループから外した。
- **画像と自己状態を同じ時点へ揃える。** 時刻付きsnapshotを同じシーンへ復元し、
  取得時刻を保ったまま推論する。カメラqueueは長さ1で最新状態のみ保持する。
- **ROSを通ったデータで動かす。** 2ノード間で画像と行動列を送受信し、
  受信した値をモデルと制御器へ渡す。送受信hashを照合する。
- **待機も物理へ反映する。** 行動queueが空なら姿勢差分0・グリッパ指令維持。
  推論待ちの間も物理を進め、待機割合と観測から適用までの時間を記録する。
- **資源を制限して比較する。** RTX5080 16GBにモデルを1種類ずつ常駐させる。
  モデルCPU threads4、カメラBLAS1、同時推論1要求で測定する。

## 結果

共通条件は各5初期配置、20Hz制御、実行10行動、先読みなし、ROS経由。
実時間側は14壁時計秒、lockstepは物理を止めて推論し最大280行動で評価する。

| モデル／方式 | 公式成功 | ROS込みworker時間 | 待機割合 | 終端XY中央値 |
|---|---:|---:|---:|---:|
| SmolVLA / lockstep | 5/5 | 615ms | 0% | 24.1mm |
| SmolVLA / realtime | 3/5 | 604ms | 53.2% | 29.9mm |
| π0-FAST / lockstep | 4/5 | 1000ms | 0% | 12.0mm |
| π0-FAST / realtime | 1/5 | 1018ms | 66.7% | 125.3mm |

主比較20試行に加え、残量5行動で次の推論を始める先読み条件を両モデルで各5試行、
Smolの生成50行動を全実行する条件を5試行実施した。先読みは両モデル0/5へ悪化した。
50行動実行は3/5のままで、成功する配置が2件ずつ入れ替わった。
待機短縮だけでは成功数が増えず、行動列の整合と観測更新頻度が重要だった。

モデルの生成長はSmol50・π10で異なり、共通比較は両者の先頭10行動を実行する。
固定5配置・1課題の小標本であり、モデル全般の優劣や実機性能は推定しない。
全35試行の条件別集計と個別結果は
[results.json](artifacts/realtime-vla/report/results.json) に保存した。

## 環境構築

検証環境はUbuntu24.04／WSL2、CPython3.12、RTX5080 16GB。
Linux x86_64、CUDA12.8対応ドライバ、Docker、初回ダウンロード用ネットワークを使う。
日本語動画ラベルの生成にはシステムの日本語フォントが必要で、
Noto Sans CJKまたは明示したfontファイルを使用する。

~~~bash
bash scripts/setup_realtime_vla.sh
.venv-realtime/bin/python experiments/realtime_vla/download_smolvla.py
.venv-realtime/bin/python experiments/realtime_vla/download_pi0fast.py
docker build -t physical-ai-ros2:jazzy-libero-vla experiments/realtime_vla
~~~

setupは単独の .venv-realtime を作成し、固定したPython依存とLIBEROアセットを導入する。
他の実験のvirtualenvや .pth fallbackは使用しない。外部重み・mesh・tokenizerは
固定revisionから取得し、配布元の利用条件とdigestを記録する。
アセットmanifestだけが存在しても、実ファイルを確認して不足分を取得する。
主要モデルはSmol約1.22GB、π約7.73GB、アセットは約422MB。

requirements-realtime.lockは実測版に合わせたruntime pin集合を含む。
opencv-pythonがcv2、hf-egl-probeがegl_probeを提供する。
使用しないプラットフォーム別・notebook依存を追加しないため、setupは --no-deps で
この固定集合を導入し、両VLAとLIBEROの実importを検証する。

## 実行

別端末でROSのSensor/Broker・PolicyTransportを起動する。

~~~bash
export REALTIME_ROS_IMAGE=physical-ai-ros2:jazzy-libero-vla
bash experiments/realtime_vla/run_ros_bridge.sh artifacts/realtime-vla/reproduction/ros
~~~

評価側では[再現ガイド](docs/guides/realtime-vla.md)の描画設定を適用する。
今回のWSL2/NVIDIA描画経路では次の設定を使用した。

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
  --ros-transport --video --output artifacts/realtime-vla/reproduction/smolvla
~~~

モデル終了後、--model pi0fastと別の新規outputへ変更して同じ条件を実行する。
初期化・warmup2回・初回chunk生成はタスク計時前に行い、時間を別記録する。
出力先は新しい名前を指定し、既存の証跡を上書きしない。
native Linuxの描画設定は使用するEGL driverに合わせる。

~~~bash
.venv-realtime/bin/python -m pytest -q experiments/realtime_vla
.venv-realtime/bin/python experiments/realtime_vla/summarize.py --help
.venv-realtime/bin/python experiments/realtime_vla/make_demo.py --help
~~~

## コードと証跡

| 場所 | 内容 |
|---|---|
| experiments/realtime_vla/ | 凍結方策、物理・カメラworker、ROS、queue、評価・集計・録画 |
| scripts/setup_realtime_vla.sh | 単独runtime構築、固定アセット取得、import確認 |
| requirements-realtime.in／.lock | 主な依存と完全なruntime pin集合 |
| artifacts/realtime-vla/report/ | 条件別集計、図、監査記録 |
| artifacts/realtime-vla/*-protocol.json | 主比較・探索条件の事前定義 |
| [構成](docs/architecture/realtime-vla.md) | センサからアクチュエータまでの構成 |
| [実験記録](docs/experiments/realtime-vla.md) | 切り分け・失敗・得られた知見 |
| [資産一覧](docs/reference/realtime-vla-assets.md) | 固定revision、容量、利用条件 |

モデル・シミュレータ・タスクは既存資産を利用し、追加した範囲は
非同期の実行系、観測と行動の時刻管理、ROS実経路、比較・監査である。
コードのライセンスを外部の重み・tokenizer・meshへ一括適用しない。
