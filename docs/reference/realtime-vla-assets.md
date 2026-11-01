# リアルタイムVLA比較で使用する資産・版・利用条件

確認日: **2026-11-02**。対象は、同一のLIBEROアーム操作を、公開されたタスク向けSmolVLAとπ0-FASTで**追加学習なし**に実行する比較。設計は[リアルタイムVLAの構成](../architecture/realtime-vla.md)を参照する。本資料は実際に取得・使用した資産の参照表であり、タスク成功率の報告とは分ける。

## 1. 選択したモデル

| 項目 | SmolVLA | π0-FAST |
|---|---|---|
| 公開checkpoint | [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102) | [lerobot/pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero/tree/840f4b503f4c09110421c33c810a85b6684fd658) |
| 固定revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | `840f4b503f4c09110421c33c810a85b6684fd658` |
| 行動の生成方式 | flow matching、10 flow steps | FAST tokenの自己回帰生成、KV cache有効、greedy |
| ネイティブな**生成**chunk | **50×7** | **10×7** |
| 保存configの `n_action_steps` | 1 | 10 |
| 共通比較の実行prefix | 最初の10行動（E10） | 最初の10行動（E10） |
| 精度条件 | BF16 autocast、公開checkpointの混合dtypeを維持 | BF16/F32混在、公開checkpointの混合dtypeを維持 |
| 重みの利用条件 | モデルカードにApache-2.0を宣言 | 選択revisionにモデルカード・明示的license欄がない。PaliGemma/Gemma由来の条件を保持し、MIT/Apache扱いにしない |
| ローカル学習・量子化 | 実施なし | 実施なし |

`smolvla_base`や`pi0fast-base`を無調整で接続する実験ではない。π0、π0.5は別のモデルであり、今回のπ0-FASTを置き換えていない。公式LeRobotの前処理、保存された正規化統計、後処理を使用し、モデル内部の単一行動queueを使わず `predict_action_chunk()` の生成時間を測る。

SmolVLAの生成長50とπ0-FASTの生成長10は一致しない。共通E10比較でも**計算する生成長そのものは各モデル固有**である。SmolVLAの50行動をすべて再生する条件は、E10比較とは別条件になる。保存configの消費長1を、生成長50と読み替えない。

## 2. 容量・パラメータ数・単発GPU検証

容量のGBは10進、GiBは `bytes / 2^30`。GPUピークはPyTorchの `max_memory_allocated()` であり、デスクトップやドライバを含むGPU全体の使用量ではない。

| 数量 | SmolVLA | π0-FAST |
|---|---:|---:|
| ユニークParameter数 | **604,934,176** | **3,449,982,704** |
| checkpointに保存されたテンソル数 | 788 | 604 |
| checkpoint保存要素数 | 604,934,176 | 3,449,982,704 |
| `model.safetensors` 実ファイルbytes | 1,218,047,032 | 7,729,830,688 |
| 同ファイルの10進GB | 1.218047032 | 7.729830688 |
| 保存dtype別要素数 | BF16: 600,902,304 / F32: 4,031,872 | BF16: 3,035,103,232 / F32: 414,879,472 |
| 単発smokeのCUDAピークbytes | 1,574,203,392 | 7,876,128,256 |
| 同CUDAピークGiB | **1.466091** | **7.335216** |
| 初回chunk生成、入力からCPU出力まで | 1,621.5 ms | 1,284.9 ms |
| warm後の2回 | 508.8 / 463.5 ms | 673.4 / 706.1 ms |

SmolVLAのParameter数は、実際にロードした `sum(p.numel() for p in runtime.policy.parameters())` の保存結果。π0-FASTは同じ公式constructorを**meta device**で構築し、同じ `.parameters()` 集計を追加確認した。全Parameterのdeviceは `meta`、重み用ストレージの確保・GPU起動・推論は行っていない。π0-FASTは `lm_head.weight` と入力embeddingを共有せず、aliasを含めた集計、`state_dict()`、checkpoint保存要素数も3,449,982,704で一致した。凍結によるtrainable Parameter数0と、モデルの全Parameter数を区別する。

単発smokeは**実シミュレータの固定観測に対する3回の推論**である。ROS往復・連続物理制御・タスク成功率を含めず、2回のwarm値からp95等を推定しない。SmolVLAは[実測JSON](../../artifacts/realtime-vla/smolvla-api-smoke-b/result.json)、π0-FASTは[実測JSON](../../artifacts/realtime-vla/models/pi0fast/gpu-smoke.json)に保存した。両モデルとも実行前後の全登録重みSHA256が一致した。RTX5080 16GBには**1モデルずつ**ロードして検証した。

### 重みファイルのSHA256

| モデル | HF LFS宣言と照合したSHA256 |
|---|---|
| SmolVLA | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| π0-FAST | `f1f8236d44016a533daba7b8249756e84df047f46e20ca17d3fc34d8f2f691ba` |

取得した全ファイルの容量・digestは、[SmolVLA provenance](../../artifacts/realtime-vla/models/smolvla/provenance.json)と[π0-FAST provenance](../../artifacts/realtime-vla/models/pi0fast/provenance.json)を参照。

## 3. シミュレータ・ロボット・タスク

| 項目 | 固定した内容 |
|---|---|
| 環境 | [HF版LIBERO](https://github.com/huggingface/LIBERO)、配布package `hf-libero==0.1.4` |
| タスク | `libero_spatial` / task0、黒いボウルをplateとramekinの間から取り、plateに置く |
| ロボットと制御 | LIBERO既存Pandaアーム・グリッパ、robosuiteの相対姿勢OSC制御→関節トルク→MuJoCo物理 |
| 入力センサ | raw RGB256×256×3を2視点、EEF位置3・quaternion4・gripper位置2、自然言語指示 |
| モデル側state | 公式 `LiberoProcessorStep` がquaternionをaxis-angleへ変換し、位置3＋姿勢3＋gripper2の8次元を作る |
| 画像の向き | ROSではrawのまま。公式model adapterで180度回転を**1回**適用 |
| 出力 | 7次元の相対XYZ・姿勢・gripper行動列。モデル出力を関節角やトルクそのものとは扱わない |
| 物理制御周期 | canonical 20Hz、物理積分dt 0.002秒、reset後10ステップ整定 |
| HFデータmetadataのfps | 10。物理制御周期20Hzとは別の値として保存 |
| 共通アセット | [lerobot/libero-assets](https://huggingface.co/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6)、revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| 初期状態 | タスク同梱50状態。評価で使う `init_state_id` をrunごとに保存 |
| 初期状態ファイルSHA256 | `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a` |

物体の真値位置、成功フラグ、評価用距離はmodel入力へ渡さない。入力に使うのは上表のカメラ・proprioception・指示だけ。環境の[metadata](../../artifacts/realtime-vla/environment-smoke/environment-metadata.json)と[アセットmanifest](../../artifacts/realtime-vla/libero-assets-manifest.json)に固定値・取得digestを保存した。

20Hzで10行動を再生すると供給予算は0.5秒になる。10Hzなら1秒になるが、制御周期の変更であり同じ実験条件ではない。撮像を別processに分ける実時間条件では、同一物理snapshotから両画像とproprioceptionを再構成し、capture時刻・tick・episodeを保持する。単なる推論時間の短縮と、観測の新鮮さ・物理ループの追従を区別する。依存と撮像に関する切り分けは[environment-build-notes](../../artifacts/realtime-vla/environment-build-notes.json)を参照。

## 4. 実行ライブラリとROSの固定版

計測には共通CUDA環境と専用 `.venv-realtime` overlayを使用し、モデル用のNumPy・Transformers等を分離した。[環境version記録](../../artifacts/realtime-vla/environment-versions.json)が実際のimport結果である。再構築用スクリプトは、他の実験への依存を避けるため同じ主要版を独立した `.venv-realtime` へ導入する。

| 層 | 実行した版 / revision | コードの利用条件・一次資料 |
|---|---|---|
| LeRobot | 0.6.2 / `3f797de5952432284a7c77ad2760218cb74e2f26` | [Apache-2.0](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/LICENSE) |
| Python | 3.12.3 | [PSF License](https://docs.python.org/3.12/license.html) |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128、CUDA build12.8 | [PyTorch BSD-3-Clause等の同梱notice](https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE)、[torchvision BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.22.1/LICENSE) |
| Transformers / HF Hub | 5.4.0 / 1.6.0 | [Transformers Apache-2.0](https://github.com/huggingface/transformers/blob/v5.4.0/LICENSE)、[HF Hub Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/v1.6.0/LICENSE) |
| NumPy | 2.2.6 | [BSD-3-Clauseと同梱ライブラリのnotice](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt) |
| HF LIBERO | 0.1.4 | [MIT](https://github.com/huggingface/LIBERO/blob/main/LICENSE) |
| robosuite / robomimic | 1.4.0 / 0.2.0 | [robosuite MIT](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/LICENSE)、[robomimic MIT](https://github.com/ARISE-Initiative/robomimic/blob/v0.2.0/LICENSE) |
| MuJoCo | 3.3.7 | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/3.3.7/LICENSE) |
| SentencePiece | 0.2.1 | [Apache-2.0](https://github.com/google/sentencepiece/blob/v0.2.1/LICENSE) |
| ROS2 | Jazzy、rclpy7.1.12、sensor_msgs/std_msgs5.3.8 | [rclpy Apache-2.0](https://github.com/ros2/rclpy/blob/jazzy/LICENSE)、[common_interfaces Apache-2.0](https://github.com/ros2/common_interfaces/blob/jazzy/LICENSE) |
| DDS adapter | rmw_cyclonedds_cpp2.2.4 | [adapter Apache-2.0](https://github.com/ros2/rmw_cyclonedds/blob/jazzy/LICENSE)、[CycloneDDS EPL-2.0 / EDL-1.0](https://github.com/eclipse-cyclonedds/cyclonedds/blob/master/LICENSE) |

補助packageはPyAV15.1.0、pandas2.3.3、pyarrow26.0.0、matplotlib3.11.2、regex2026.9.29。上表は主要な構成資産の整理であり、CUDA・OS・全間接依存を単一のMIT/Apacheライセンスにまとめたものではない。

ROS用Docker imageは `physical-ai-ros2:jazzy-puzzle-cyclone`、実体ID `sha256:69da362aa080cb4380ab81c1a51b7bae7b1bf13d6868ef4566ed3a7215bd3303`。domain45、loopback port8780。独立したSensor/Broker nodeとPolicyTransport nodeの間で実際のImageと行動chunkを送る。モデル推論はhostのGPU child processが担当し、ROS内で実行したと表現しない。[実ROS合成入力テスト](../../artifacts/realtime-vla/ros-smoke/transport-checks.json)は通信の証拠であり、学習モデルの成功実績に含めない。

## 5. tokenizerとロード時の互換対応

| 対応 | 実際に行ったこと | 検証・境界 |
|---|---|---|
| SmolVLAの視覚言語モデル | [SmolVLM2-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)のconfig/tokenizerをrevision固定。追加の基盤重みは取得せず、policy checkpointから全層を復元 | VLM metadataのrevision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`、Apache-2.0宣言。保存788tensorの値とロード後の値を照合 |
| π0-FASTのtext tokenizer取得 | gated HF URLからの取得は401。公式Google公開配布の[PaliGemma SentencePiece](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)を使用 | 無関係なtokenizerで代用していない。秘密token・gated model weightsは使用していない |
| tokenizer形式変換 | 公式Transformers `GemmaConverter` を使用。5.4.0のextractorが欠く `self.sp` を変換時だけ補うlocal adapterを適用 | installed packageと学習重みは変更なし。全257,152語彙piece ID、PAD/EOS/BOS/UNK=0/1/2/3、6入力のencode一致を検証。公式converterのtab表記aliasのみ許容 |
| FAST action tokenizer | [jadechoghari/tokenizer-lib-mean](https://huggingface.co/jadechoghari/tokenizer-lib-mean/tree/79ae83e3cbd8786dcb84b628569f8d076ca8151e)を固定し、DCT/BPEのlocal processor codeを確認してロード | revision `79ae83e3cbd8786dcb84b628569f8d076ca8151e`、horizon10、action7、vocab1024、scale10。保存されたMEAN_STD統計を維持 |
| π0-FASTの重み復元 | 上流 `from_pretrained()` が復元例外を捕捉して未復元モデルを返す経路を避け、公式constructorと公式key remap後に `load_state_dict(strict=True)` を実行 | 604tensorを厳密ロード。失敗は試行エラー。ランダム重みを学習済モデルとして扱わない |
| FAST復号エラー | 上流が復号失敗でzero chunkを返すwarningを捕捉し、試行エラーにする | 正常な停止指令と復号失敗を混同しない。単発smokeでfallback0件 |

Google tokenizerのSHA256は `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。検証明細は[tokenizer-validation.json](../../artifacts/realtime-vla/models/pi0fast/tokenizer/tokenizer-validation.json)、action tokenizerの取得digestは[provenance](../../artifacts/realtime-vla/models/pi0fast/tokenizer/action/provenance.json)に保存した。

これらは固定した学習済モデルを正しく復元するための接続・検査であり、追加学習や別モデルへの置換ではない。π0-FASTでは `compile_model=False` のeager条件を採用した。現実装のcompileフラグはKV cacheを使う生成関数を直接compileしないため、フラグだけで実時間性能が向上したとは扱わない。[公式モデル実装](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py)参照。

## 6. データ・重み・meshの条件を分ける

| 対象 | 確認できた条件 | 今回の扱い |
|---|---|---|
| SmolVLA checkpointとSmolVLM metadata | 各モデルカードのApache-2.0宣言 | 宣言・revision・digestを保存 |
| π0-FAST checkpoint | 選択main revisionのcard/license欄は欠落。[base](https://huggingface.co/lerobot/pi0fast-base)と[公式v044 sibling](https://huggingface.co/lerobot/pi0fast-libero-v044)はGemmaを宣言 | PaliGemma/Gemma由来の利用条件を保持。今回使用したmain重みをApache/MITと断定しない。v044の重みは本実験に混ぜない |
| 公開text tokenizer | Google公式公開URLから取得でき、元SPとID一致を検証 | 公開ダウンロード可能であることを、条件なしの許諾とは読み替えない |
| FAST action tokenizer | 選択repoのcardはlicense未記入 | 固定source・digestを記録。Apache/MITと推定せず、他の自作tokenizerへ置換しない |
| LIBERO環境コード | upstream/HF版ともMIT | 既存タスク・robot/controllerを再利用 |
| `lerobot/libero-assets` のmesh/texture群 | 選択snapshotのAPIにcard/license宣言・README/LICENSEがない。コードのMITを個々の外部assetへ一括適用できない | 元のasset・noticeを保持し、manifestで出所を追跡。個別mesh/textureの権利棚卸は未完了 |
| [lerobot/libero実演データ](https://huggingface.co/datasets/lerobot/libero) | 公開dataset cardはApache-2.0宣言 | 学習データ本体を今回のローカル学習に使用していない。実行に必要なtask/初期状態/assetと公開モデルを取得 |

Gemma TermsはPaliGemmaを対象に含み、取得元が異なっても適用対象を変えない。配布・hosted serviceには利用制限の引継ぎ、利用規約の提供、変更表示等の条件があり、ファイル配布ではNoticeも求められる。禁止用途ポリシーと法令遵守が利用条件に含まれる。今回の配布物はコード・動画・評価証跡を中心とし、巨大なモデル重みをWebページへ再配布しない。[GoogleのGemma Terms §1・§3](https://ai.google.dev/gemma/terms)、[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)、[ローカル記録](../../artifacts/realtime-vla/models/pi0fast/license-notes.json)参照。

### 上流学習履歴について確認できないこと

選択SmolVLA cardの `datasets` は `unknown`。選択π0-FASTの `train_config.json` はdatasetがlocal path、revision未記入、steps20,000となっている。したがって、上流で使われた実演データの完全な版・episode集合・学習回数を、この取得記録だけから確定しない。[一般のLeRobot π0-FAST文書](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/docs/source/pi0fast.mdx)にある学習設定・公表成功率と、選択重み・本環境の実測値を自動的に同一視しない。

今回確認できるのは、**固定した公開checkpointを復元し、追加学習せず、同一タスク仕様へ接続して評価していること**である。独自のtask学習、ROS通信の導入、wallclockの周期追従、タスク成功はそれぞれ別の証拠で示す。
