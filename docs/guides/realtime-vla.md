# 実時間VLA比較の再実行

構成は[architecture](../architecture/realtime-vla.md)、結果は[実験記録](../experiments/realtime-vla.md)、モデルと利用条件は[資産一覧](../reference/realtime-vla-assets.md)を参照。

## 前提

検証ホストはWSL2/Ubuntu24.04、Python3.12.3、RTX5080 16GB。公開コードの `scripts/setup_realtime_vla.sh` は、固定した依存を独立した `.venv-realtime` へ導入する。他の実験の環境・ソースコードは不要。PyTorch2.7.1+cu128/torchvision0.22.1、LeRobotの固定commitなど、評価に用いた主要な版を再現する。

計測時には共通CUDA環境と専用overlayを使用した。配布用の構築手順は、同じ主要版を1つの環境へ用意する構成に整理している。実測時の版・環境情報はartifactに保存している。

CUDA利用・Docker・描画権限が必要。モデルはSmol約1.22GB、FAST約7.73GB、LIBEROアセット約422MBをダウンロードする。学習データは追加取得しない。外部重み・tokenizerはリポジトリに含めず、固定revisionの取得スクリプトを使う。

## 環境・公開モデル

workspaceのルートで実行する。

```bash
bash scripts/setup_realtime_vla.sh
.venv-realtime/bin/python experiments/realtime_vla/download_smolvla.py
.venv-realtime/bin/python experiments/realtime_vla/download_pi0fast.py
# pytestがない環境では、テスト用依存を追加する。
UV_CACHE_DIR="$PWD/.cache/uv" .tools/uv pip install --python .venv-realtime/bin/python pytest==8.4.2
.venv-realtime/bin/python -m pytest experiments/realtime_vla/test_chunk_queue.py experiments/realtime_vla/test_libero_env.py experiments/realtime_vla/test_camera_worker.py experiments/realtime_vla/test_ros_bridge.py -q
```

FASTは初回runtimeロード時に、公開Google SentencePieceと固定revisionのFAST tokenizerを準備・検証する。公開tokenizerの利用でも、モデルと依存資産の利用条件は別に確認する。厳密な復元に失敗すると評価を止め、初期化された未学習policyで代用しない。

再構築用の依存は `requirements-realtime.in/.lock`、実測時の主要版は `artifacts/realtime-vla/environment-versions.json`。`installed-freeze.txt` と `installed-freeze-combined.txt` は元のoverlay／共通環境を含む実測時の版一覧であり、再構築用lockとは区別する。ローカルwheelの取得先は、同一ファイルの公開URLとSHAへ置換して記録した。PyTorch2.7.1のCUDA12.8 buildでBlackwell GPU動作を確認済み。

## ROS 2を起動

ROS2 JazzyとCycloneDDSを含むタスク専用イメージを作成する。

```bash
docker build -t physical-ai-ros2:jazzy-libero-vla -f experiments/realtime_vla/Dockerfile .
REALTIME_ROS_IMAGE=physical-ai-ros2:jazzy-libero-vla bash experiments/realtime_vla/run_ros_bridge.sh artifacts/realtime-vla/ros-reproduction
```

この起動端末を保持し、別端末で評価する。domain45、host loopback8780、Sensor/BrokerとPolicyTransportの2ノード。終了は起動端末でCtrl+C。ノード・コンテナの停止処理を含む。元の実測イメージのIDとROSパッケージ版は `artifacts/realtime-vla/ros-system/` に保存している。

## 正式評価

下記2コマンドは**順番に**実行する。両モデルの同時常駐で速度を比較しない。出力先は新しいディレクトリ名を使い、既存結果を上書きしない。

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
# 次の3つは今回のWSL/NVIDIA描画経路。native LinuxのEGLでは不要。
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

.venv-realtime/bin/python experiments/realtime_vla/evaluate.py --model smolvla --output artifacts/realtime-vla/repeat-smolvla-e10 --conditions lockstep realtime --initial-states 1 2 3 4 5 --control-hz 20 --wall-seconds 14 --execute-steps 10 --prefetch-steps 0 --video --ros-transport

.venv-realtime/bin/python experiments/realtime_vla/evaluate.py --model pi0fast --output artifacts/realtime-vla/repeat-pi0fast-e10 --conditions lockstep realtime --initial-states 1 2 3 4 5 --control-hz 20 --wall-seconds 14 --execute-steps 10 --prefetch-steps 0 --video --ros-transport
```

設定は事前の `formal-protocol.json` と対応する。起動と2回warmup後、初期状態・画像を取り直して最初のchunkを生成してからタスク時計を開始する。14秒に起動を含めない。時刻付きsnapshot、非同期poll、一度に1推論、空queueの差分ゼロを用いる。`--sync-camera` は描画律速を調べる診断用で、主比較の条件ではない。

## 集計・動画

```bash
.venv-realtime/bin/python experiments/realtime_vla/summarize.py --runs artifacts/realtime-vla/repeat-smolvla-e10 artifacts/realtime-vla/repeat-pi0fast-e10 --output artifacts/realtime-vla/repeat-report
.venv-realtime/bin/python experiments/realtime_vla/make_demo.py --help
```

主比較の20episodeだけなら上記で集計する。追加条件を再実行する場合は、評価コマンドの条件を `--conditions realtime --prefetch-steps 5 --execute-steps 10` に変更し、両モデルの5状態を別の新規出力先へ保存する。Smolの生成50を全実行する条件は `--conditions realtime --prefetch-steps 0 --execute-steps 50` とする。control20Hz・camera10Hz・14秒・ROSは維持。`formal-protocol.json` の20試行を上書きしない。

今回の35試行を図・動画へまとめたコマンドは次の通り。

```bash
.venv-realtime/bin/python experiments/realtime_vla/summarize.py --ablation-runs artifacts/realtime-vla/ablation-smolvla-prefetch5 artifacts/realtime-vla/ablation-pi0fast-prefetch5 --native-smol-run artifacts/realtime-vla/native-smolvla-e50 --output artifacts/realtime-vla/report
.venv-realtime/bin/python experiments/realtime_vla/make_demo.py --report artifacts/realtime-vla/report/results.json --output artifacts/realtime-vla/report
```

元データは各episodeの `initial.json / ticks.jsonl / chunks.json / result.json` とrun全体の `config.json / model-audit.json / summary.json`。モデル真値を渡していないことはworker入力のallowlist、ROS使用はchunk metadataのhash、重み固定はmodel-auditで確認する。

raw rollout MP4は制御tickのfpsで保存しており、lockstepの推論停止時間は映らない。比較映像では `ticks.jsonl` に記録した経過時間に合わせてフレームを配置する。raw lockstepを実時間動画として示さない。

全GPU評価を終えてから `sensor_phase_audit.py` を実行すると、通常stepのcacheとsnapshot強制観測更新の差を同じ初期物理状態で計測できる。再集計前に実行するとauditを結果JSONへ添付する。モデル評価と同時にGPU診断を動かして速度を比較しない。

## 失敗の読み方

- `status:error` はタスク失敗と分けて読む。モデルの復号失敗をゼロactionへ置換しない。
- 成功は公式predicate。XY距離が近いだけでは、把持・配置の成功にならない。
- 観測ageはbootstrap、通常action、最新cameraの3種類を区別する。
- 20Hz設定だけで判断せず、実際のtick間隔とRTFを確認する。
- wall budget・制御Hz・E・prefetch・native生成長を変えた追加試行は、主比較を置き換えず別条件として保存する。
