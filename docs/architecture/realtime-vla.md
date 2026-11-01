# リアルタイムを想定するVLA action chunk実行

2026-11-02。LIBEROのボウル移載タスクを、凍結したSmolVLAとπ0-FASTで実行する。仮説は「小型モデルの低遅延が、より大きいモデルの静的な方策性能差を、実行時には逆転させる場合がある」。モデル名から精度・速度の優劣を決めない。

## 再利用する資産

同じLIBEROのタスク・カメラ・Panda・制御器を使い、タスク適応済みSmolVLAとπ0-FASTの公開重みを追加学習なしで実行する。π0やπ0.5をπ0-FASTの代替として比較しない。HF版の公式LeRobotライブラリで前処理・正規化・行動復号を行う。モデルと実装のrevisionを固定し、同じ初期状態で比較する。

## 比較する実行方式

lockstep基準では推論やROSを待つ間に物理時刻を止める。実時間実行ではシミュレータの制御ループと推論を分離し、chunk生成中も周期的な制御と物理更新を継続する。wallclockの制御周期を基準とし、物理計算自体の遅れも記録する。

## 比較する量

- タスク成功・到達時間・物体と目標の終端誤差。
- chunk生成時間（平均・p95・最大）、観測取得から実行までのage。
- action queue残量・枯渇、待機action比率、expired action数。
- 実際の制御周期・期限超過・simulation/wall時間比。
- 同じモデルのlockstep基準との性能低下。
- GPUメモリ。モデルは1種類ずつ実行し、同時推論による競合を避ける。

制御指令の変化量は滑らかさの指標であり、位置精度そのものとは呼ばない。モデルが出す差分目標と関節追従誤差、タスクの物体目標誤差は区別する。chunkが長いだけで観測が新鮮になるとは扱わない。

## queueが空のとき

相対姿勢指令を繰り返すと、停止ではなく移動を継続してしまう。空queue時は位置・姿勢の差分をゼロにして、グリッパ状態の指令のみ保持する。学習モデルのactionを勝手に「最後の値保持」で再実行しない。これもtaskを成功させる補助policyではなく、入力が届かない間の待機規則として全モデルへ同じ条件で適用する。

## 公平性

SmolVLAのflow action expertと、π0-FASTの自己回帰token生成ではchunk構造と学習条件が異なる。まず共通のqueue処理・動作周期・消費action数で比較する。SmolVLAだけのRTCを試す場合は、モデル差とscheduler差を混ぜず別条件にする。最大の公表精度を今回のtask成績の代用にしない。

## 実装の境界

モデル生成長、実際に実行するaction数、次の推論を要求する残量、遅れて届いたchunkの採否を設定として保存する。固定した初期状態と乱数を用い、追加学習せず比較する。実時間周期に追従できない場合は、リアルタイム達成とは報告せず、その不足も評価結果へ含める。

## 実際に構築した構成

```mermaid
flowchart LR
  P[LIBERO / MuJoCo 物理20Hz] -->|時刻付きsnapshot| C[別プロセスの2カメラ10Hz]
  C -->|RGB256×256×2 + 同時点の自己状態| S[ROS Sensor/Brokerノード]
  S -->|sensor_msgs/Image + メタデータ| R[ROS PolicyTransportノード]
  R -->|受信済み観測| V[GPUワーカー / 凍結VLA]
  V -->|chunk生成| R
  R -->|action_chunkトピック| S
  S -->|受信済みchunk| Q[共通FIFO / 先頭10action]
  Q -->|7D相対手先指令20Hz| O[既存OSC 位置・姿勢制御]
  O -->|関節トルク| P
```

実装は `experiments/realtime_vla/`。ROS Jazzy/CycloneDDSを別コンテナに置き、ホストのGPUライブラリとはTCP adapterで接続する。画像・自己状態・生成actionは実際にROSトピックを往復し、受信した値をモデル入力および制御に使用する。metadataだけをROSに送り、モデルは別の元画像を使う構成ではない。送受信のハッシュを各chunkに保存する。

Sensor/BrokerとPolicyTransportは別ROSプロセス。物理、カメラ、モデルも別プロセスに分離する。GPUワーカー内でROSを待っても、物理の周期ループは待たない。queueは一度に1推論、カメラは最新snapshotのみ採用。制御途中の `wait()` はlockstep参照だけで許し、実時間ループでは `poll()` / `put_nowait()` を使う。

### カメラが周期を壊さないようにする

同期2カメラはWSLのNVIDIA描画経路を選んでも20Hz制御予算50msを超えた。物理は画像なしで進め、複製した同一シーンへ時刻付き物理状態を適用して別プロセスで描画する。元の同期環境とsnapshotを一致させ、画像・手先・グリッパを全バイト比較して一致を確認した。画像と自己状態は同じsnapshotから作る。描画終了時刻を取得時刻へ書き換えない。reset境界はepisode_idで古いフレームを除外する。

この一致検証は、同じsnapshotを両sceneで `regenerate_obs_from_state()` した場合の一致。通常の同期 `env.step()` が返す観測cacheとの一致ではない。正式試行の初期監査では、同じ方式のモデル間はセンサhashが一致し、方式間は物体位置とactuator ctrlが一致する一方でセンサhashは異なる。robosuiteの周期サンプリングcacheと、snapshot復元後の強制観測更新の位相差があるためで、lockstep対realtimeを純粋な推論遅延だけの比較として解釈しない。

### 凍結・実時間・成否を別に確認する

公開重みを厳密に復元し、実行前後の重みdigestが同じことを監査する。追加学習、optimizer、物体真値を入力する補正policyはない。controllerは学習時と同じrobosuite OSCであり、今回独自のIKを追加したとは扱わない。公式成功判定・物体の目標XY誤差は評価器だけで計算する。

初期化・2回のwarmup・最初のchunk生成は初期物理状態を保持して計時外で実施する。warmup後に観測を取り直し、その取得時刻から初回actionまでのageは保持する。起動時間を実時間達成へ含めたとは報告しない。queueが空なら差分姿勢ゼロ＋グリッパ指令維持。FIFOは遅延actionの時刻を現在へ書き換えずに実行し、ageを計測する。この単純な待機方式はモデルへ遅延を条件づけるRTCとは異なる。

正式評価の事前条件は `artifacts/realtime-vla/formal-protocol.json`、モデル・依存・ライセンスは[資産一覧](../reference/realtime-vla-assets.md)、実測は[実験記録](../experiments/realtime-vla.md)を参照する。
