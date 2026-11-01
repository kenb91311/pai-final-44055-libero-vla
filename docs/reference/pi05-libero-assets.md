# π0.5 LIBERO追加検証の固定資産

確認日: **2026-11-02**。先行する[SmolVLA / π0-FAST比較](realtime-vla-assets.md)とは追加検証を分離する。既存35試行の母数・成果物・公開Gitには未検証のπ0.5結果を混ぜない。

## 選定したcheckpoint

| 項目 | 固定した内容 |
|---|---|
| repository | [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned/tree/7152f13e34c6c115d808aea0e5d08aca74eac14b) |
| revision | `7152f13e34c6c115d808aea0e5d08aca74eac14b` |
| 公式alias | `lerobot/pi05_libero_finetuned_v044`も同一revision・同一重みを返す |
| taskへの適合 | cardにLIBEROで学習・評価済みと記載。言語・2画像・state8からaction7を生成 |
| 採用理由 | 追加学習を要求するgeneric `pi05_base`ではなく、実行対象に対応した公開学習済み重み |
| action head | 連続値のflow matching。FAST action tokenizerは使用しない |
| native chunk / 保存消費長 | 50×7 / `n_action_steps=50`。公式benchmarkの評価例は消費長10を明示 |
| flow推論 | 10 steps、今回の基準はeager、RTC・visual/proprio memoryは無効 |
| 正規化 | 保存された**MEAN_STD**。モデル一般のQUANTILES既定値へ変更しない |
| 重みbytes | **7,473,096,344**（7.473 GB） |
| LFS SHA256 | `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae` |
| HF宣言 | 公開・gateなし、モデルカードのlicenseは**Gemma** |

取得スクリプトは [download_pi05.py](../../experiments/realtime_vla/download_pi05.py)。revision、容量、LFS digestの不一致では停止する。14.47GBのF32 generic/base重みは取得していない。

## 公表benchmarkと今回の測定

[公式LeRobot LIBERO文書](https://huggingface.co/docs/lerobot/libero)は、π0.5のSpatial97.0%、Object99.0%、Goal98.0%、LIBERO-10 96.0%、4suite平均97.5%を報告する。[π0.5文書](https://huggingface.co/docs/lerobot/pi05)はMEAN_STD設定と、このfinetuned_v044重みの関係を明示している。

これは公開側の複数タスク評価であり、今回の黒いボウル1タスク・固定初期状態・wallclock動作の成績ではない。先行SmolVLA / π0-FASTの手元成績と、公表97.5%を同列の実測結果として比較しない。今回も同じタスク・初期状態・制御周期・ROS・chunk消費条件を揃えた追加評価が必要になる。

公開cardは、このLeRobot版に含まれるのはflow action headであり、論文のsubtask prediction・action tokenization・RL等の未公開部分を含まないと説明している。モデル名からそれらの機能を実装済みと扱わない。[固定card](https://huggingface.co/lerobot/pi05_libero_finetuned/blob/7152f13e34c6c115d808aea0e5d08aca74eac14b/README.md)

## 既存環境に接続する方法

LeRobot0.6.2 / commit `3f797de5952432284a7c77ad2760218cb74e2f26`、Transformers5.4.0、torch2.7.1+cu128の既存 `.venv-realtime` を使用する。旧モデルruntime・package versionは変更していない。

[pi05_runtime.py](../../experiments/realtime_vla/pi05_runtime.py) の `FrozenPI05` は、`predict(raw_observation,prompt) -> (actions, metadata)`、`reset(seed)`、`audit()`、`close()` を提供する。入力はraw RGB256の2視点とEEF位置・quaternion・gripperだけ。公式 `LiberoProcessorStep` が180度回転を1回行い、axis-angleを含むstate8へ変換する。真値object poseや成功フラグは渡さない。

text tokenizerはGoogle公式の[公開PaliGemma SentencePiece](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)、SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` を使用する。Transformersの公式GemmaConverterを用い、5.4.0のextractorが欠くSP handleを変換時だけ補う。全257,152語彙ID、特殊ID、6入力のencode一致を検査する。無関係なvocabulary・FAST action token・gated model weightsを代用しない。

## strict restorationとパラメータ数

上流 `PI05Policy.from_pretrained()` には、復元失敗を捕捉して未復元モデルを返す経路がある。この追加runtimeは公式constructorと公式key conversionを使用し、例外を捕捉しない `load_state_dict(strict=True)` の後、**全mapped tensorを元checkpointの公式dtype cast後の値と `torch.equal` で許容誤差なしに比較**する。

| 集計 | 値 |
|---|---:|
| 保存tensor | 812 |
| 保存要素 | 3,616,757,520 |
| 公式conversion後のruntime tensor | 813 |
| ユニークruntime Parameter数 | 4,143,404,816 |
| runtime Parameterのdtype | BF16: 3,609,853,952要素 / F32: 533,550,864要素 |
| Parameterストレージの静的集計 | 9,353,911,360 bytes。activation等を含むGPUピークとは別 |
| 追加された要素の由来 | 526,647,296要素の入力embeddingを、保存された `lm_head.weight` から公式処理で複製 |

既存環境のmeta-device構築で、strict key/shape検査のmissing/unexpectedは0、全Parameterのdeviceはmetaだった。[schema確認JSON](../../artifacts/realtime-vla/models/pi05/schema-smoke.json)に保存した。追加embeddingは新しい学習・ランダム重みではなく、公式復元規則によるコピーである。モデル名の概数やHF保存要素数を、そのまま実runtimeのユニークParameter数とは扱わない。

## 検証状況と利用条件

CPU確認に加え、RTX5080で実保存LIBERO init0のraw観測からnative50×7行動を生成した。warmup2回後に3回測定し、全813mapped tensorの復元一致、全登録重みの実行前後SHA256一致、`requires_grad` Parameter数0を確認した。

| API単発確認 | 実測 |
|---|---:|
| model初期化・全値検査等のstartup | 102.11秒 |
| warmup2回 | 1273.93 / 346.78 ms |
| warm後3回のraw入力→CPU行動出力 | 370.62 / 353.71 / 340.45 ms |
| 同3回の平均 | 354.93 ms |
| CUDAピーク | 9,522,044,928 bytes = **8.8681 GiB** |
| 学習・量子化・compile・RTC | すべて実施なし |

[API実測JSON](../../artifacts/realtime-vla/models/pi05/api-smoke.json)、[全tensor復元検査](../../artifacts/realtime-vla/models/pi05/restoration.json)、[processor CPU確認](../../artifacts/realtime-vla/models/pi05/processor-smoke.json)に保存した。この確認は固定観測を用いる単発推論であり、ROS往復・物理進行・タスク成功の測定を含まない。20HzのE10再生予算0.5秒に対してモデル単体は約0.35秒だったが、実システムの遅れや成功を保証する値ではない。

実タスクの追加検証は[π0.5専用protocol](../../artifacts/realtime-vla/pi05-protocol.json)を先行登録した。init0をdevelopment、init1..5×lockstep/realtimeをformalとして分離する。公開baseline `31e788163a1776a95f0739c90f5eeeaeaee7efaa` からの共有コード変更はmodel dispatchとCLI選択肢、および対応docstringのみ。元の35試行を上書きしない。

### 追加formal結果

同じE10/prefetch0・20Hz・14秒上限・実ROS条件で、**lockstep5/5、realtime5/5がnative成功**した。init0のdevelopment2件とAPI smokeはこの分母に含めない。対象は既知の1タスク・固定5状態であり、5/5のWilson95%区間は約56.6–100%。公表4suite benchmarkの再現値とは呼ばない。

| 計測 | lockstep N=5 | realtime N=5 |
|---|---:|---:|
| native成功 | 5/5 | 5/5 |
| pooled待機tick率 | 0% | **44.49%** |
| 終端goalXY中央値 | 9.90 mm | 17.65 mm |
| RTF中央値 | 0.3933 | **1.0039** |
| 実際の制御周期Hzのepisode平均を平均 | 7.81 | **19.98** |
| 定常worker往復時間のepisode平均を平均 | 372.02 ms | 411.70 ms |
| 定常model処理時間のepisode平均を平均 | 335.11 ms | 332.15 ms |
| 定常action元観測ageのepisode平均を平均 | 785.97 ms | 811.82 ms |

成功しても、prefetch0ではchunk終了後の推論待ちが残る。20Hzで物理・低位制御が進むことと、学習policyの新しい指令が途切れず供給されることを分けて報告する。host最大RAMはこの完了runでは計測しておらず、GPU/重みサイズから推定していない。

[コンパクト比較JSON](../../artifacts/realtime-vla/pi05-comparison-summary.json)、[10試行の詳細集計](../../artifacts/realtime-vla/pi05-report/results.json)、[監査](../../artifacts/realtime-vla/pi05-report/validation.json)に保存した。87生成chunk・174ROS往復を独立したpolicy/brokerノードの受信ログと照合し、813tensor復元・前後重み不変、既存35試行の190個JSON/JSONLの不変、登録source SHAの不変を確認した。既存35と追加10の計45記録は条件別に扱い、まとめた単一成功率を出さない。

LeRobot実装のコードはApache-2.0だが、この選択checkpointのcardはGemmaを宣言する。Googleの[Gemma Terms](https://ai.google.dev/gemma/terms)と[禁止用途ポリシー](https://ai.google.dev/gemma/prohibited_use_policy)の利用・再配布条件を保持する。公開ダウンロード可能であることを、MIT/Apache条件だけで使えることと読み替えない。重み・tokenizer本文を既存35試行のコード提出物へ追加配布しない。
