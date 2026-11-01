"""Create a structured, self-contained five-page Japanese submission report.

Args: CLI --report and --output pointing to completed experimental evidence.
Returns: Print-ready HTML with embedded figures; a browser prints the final PDF.
Raises: ValueError for incomplete trials, failed audits or a different protocol.
Design:
    Context: An evaluator needs to assess the robot task and the engineering contribution.
    Concern: Production history can obscure the purpose, implementation and measured outcome.
    Decision: Explain task, architecture/resources, measured results, trials and applications.
    Alternatives: A chronological development diary.
    Quality Attributes: Readability, provenance. Trade-offs: Printing needs a browser with Japanese fonts.
Evidence:
    Internal: Final35-episode JSON, validation, sensor diagnostics and immutable source videos.
    External: Official LIBERO/LeRobot model references.
    Findings: Lower latency and fewer holds do not alone guarantee task completion.
    Metrics: Native success, goalXY, source age, resource budget and model audit.
    Confidence: Five fixed initial states of one known task.
Constraints: Evidence-based claims, evaluator-facing content and no personal identifiers.
Reconsider When: Task, evaluation protocol or submission format changes.
References: docs/experiments/realtime-vla.md.
ADR: None.
"""
import argparse
import base64
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_image(path):
    """Embed a measured figure without network or filesystem dependencies in the PDF.

    Args: path: existing PNG. Returns: Data URL. Raises: IO errors.
    Design: Context: Browser printing runs on the Windows host.
        Concern: WSL-relative assets may be unavailable.
        Decision: Embed PNG bytes. Alternatives: Remote image URLs.
        Quality Attributes: Portability. Trade-offs: Larger HTML.
    Evidence: Internal: Final measured figures. External: RFC2397.
        Findings: No model data is fetched at print time. Metrics: None. Confidence: Byte identity.
    Constraints: Local PNG only. Reconsider When: Figures exceed browser memory.
    References: main. ADR: None.
    """
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def page(number, label, content):
    """Wrap one A4 page with a stable reading order and explicit page identity.

    Args: number/label/content: final report body. Returns: HTML section.
    Raises: None.
    Design: Context: Readers scan a short submission PDF.
        Concern: Automatic pagination can separate findings from their evidence.
        Decision: One topic per fixed page. Alternatives: Unbounded print flow.
        Quality Attributes: Clarity. Trade-offs: Overflow must be checked before printing.
    Evidence: Internal: Browser geometry checks. External: CSS paged media.
        Findings: Header identifies the task; footer carries page numbers.
        Metrics: Five pages. Confidence: QA dependent.
    Constraints: No hidden overflow. Reconsider When: Content grows.
    References: build_realtime_report.py. ADR: None.
    """
    return f'''<section class="page"><header><b>PAI最終課題_44055 / LIBERO VLA</b><span>{escape(label)}</span></header>
    {content}<footer><span>{number} / 5</span></footer></section>'''


def main():
    """Read audited results and build an evaluator-facing explanation of the robot task.

    Args: CLI paths. Returns: None. Raises: Missing or incomplete evidence.
    Design: Context: Final figures and counts must agree across website/PDF/repository.
        Concern: Hand-edited values drift from raw trials.
        Decision: Build all outcome tables from final JSON.
        Alternatives: Re-entering successful examples only.
        Quality Attributes: Consistency. Trade-offs: Fixed layout for one task.
    Evidence: Internal: 35 measured episodes and five frozen-model audits.
        External: Primary model/library sources linked on page5.
        Findings: Primary and exploratory conditions remain separate.
        Metrics: Success, delay, source age, holds and real control frequency.
        Confidence: Restricted sample and sensor-schedule caveats included.
    Constraints: Never interpret task distance as native task success.
    Reconsider When: New formal trials are added.
    References: summarize.py. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report', type=Path, default=ROOT/'artifacts/realtime-vla/report')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder = args.report
    result = json.loads((folder/'results.json').read_text())
    validation = json.loads((folder/'validation.json').read_text())
    if result['status'] != 'complete' or validation['episodes_verified'] != 35:
        raise ValueError('Final35-episode results required')
    if not validation['all_episode_ros_sensor_and_chunk_hashes_verified']:
        raise ValueError('ROS audit failed')
    groups = {(g['model'], g['mode']): g for g in result['groups']}
    smol, fast = groups['smolvla', 'realtime'], groups['pi0fast', 'realtime']
    names = {'smolvla': 'SmolVLA', 'pi0fast': 'π0-FAST'}
    mode_names = {'lockstep': '基準実行', 'realtime': 'リアルタイム実行'}
    rows = ''.join(f'''<tr><th>{names[g["model"]]}<small>{mode_names[g["mode"]]}</small></th>
        <td class="score">{g["successes"]}/{g["n"]}</td>
        <td>{g["full_chunk_worker_roundtrip_ms_steady_episode_mean"]["mean"]:.0f} ms</td>
        <td>{g["pooled_hold_fraction"]*100:.1f}%</td>
        <td>{g["final_goal_xy_mm"]["p50"]:.1f} mm</td>
        <td>{g["realtime_factor"]["p50"]:.3f}</td></tr>''' for g in result['groups'])
    exploratory = result['scheduler_ablation']['groups'] + result['native_horizon_smol']['groups']
    exp_rows = ''.join(f'''<tr><th>{names[g["model"]]}</th><td>{"50行動／先読みなし" if i==2 else "10行動／先読みあり"}</td>
        <td class="score">{g["successes"]}/{g["n"]}</td><td>{g["pooled_hold_fraction"]*100:.1f}%</td>
        <td>{g["final_goal_xy_mm"]["p50"]:.1f} mm</td></tr>''' for i, g in enumerate(exploratory))
    variants = [
        ('Smol：10行動・先読みなし', result['episodes'], 'smolvla'),
        ('Smol：10行動・先読みあり', result['scheduler_ablation']['episodes'], 'smolvla'),
        ('Smol：50行動・先読みなし', result['native_horizon_smol']['episodes'], 'smolvla'),
        ('π0-FAST：10行動・先読みなし', result['episodes'], 'pi0fast'),
        ('π0-FAST：10行動・先読みあり', result['scheduler_ablation']['episodes'], 'pi0fast')]
    state_rows = ''
    for title, episodes, model in variants:
        by_state = {r['init_state_index']: r for r in episodes if r['model']==model and r['mode']=='realtime'}
        state_rows += '<tr><th>'+title+'</th>'+''.join(
            f'<td class="{"ok" if by_state[i]["success"] else "no"}">{"成功" if by_state[i]["success"] else "未達"}</td>'
            for i in range(1,6))+'</tr>'
    # This is an existing recorded sensor frame, not a generated scene illustration.
    import imageio.v2 as imageio
    chosen=next(r for r in result['episodes'] if r['model']=='smolvla' and r['mode']=='realtime' and r['init_state_index']==1)
    task_image = folder/'task-frame.png'
    recorded_video=ROOT/chosen['source_directory']/'rollout.mp4'
    if recorded_video.is_file():
        reader = imageio.get_reader(recorded_video)
        try:
            frame = reader.get_data(125)
        finally:
            reader.close()
        imageio.imwrite(task_image, frame)
    elif not task_image.is_file():
        raise FileNotFoundError('A recorded task frame or its source rollout is required')
    task = data_image(task_image)
    outcomes = data_image(folder/'outcomes.png')
    pages = []
    pages.append(page(1, '課題の概要・目的', f'''
    <div class="eyebrow">CAMERA → VLA → ROS 2 → ACTUATOR</div>
    <h1>画像を見て、<br>実時間でボウルを運ぶ。</h1>
    <p class="lead">学習済みAIでPandaアームを操作する。<br>画像認識・動作列の生成・ROS 2通信・物理制御を接続し、<br>推論中もシミュレーションを進めて操作を評価した。</p>
    <figure><img class="task" src="{task}"><figcaption>実際のSmolVLA実行映像から。左：外部カメラ ／ 右：手首カメラ</figcaption></figure>
    <div class="columns"><article><h2>タスク</h2><p>皿と小鉢の間にある黒いボウルを、Pandaでつかんで皿へ運ぶ。ボウルが皿上にあるかを、LIBEROの公式判定で評価した。</p></article>
    <article><h2>目的</h2><p>小型モデルの推論の速さが操作の成功につながるかを調べる。同じタスクと制御周期で2モデルを比較し、実行時の遅れと失敗の関係を検討する。</p></article></div>
    <div class="callout"><b>公開モデルを、画像から物理操作まで接続する。</b><p>VLAは画像・言語・ロボット自身の状態から動作を推論するAI。SmolVLAとπ0-FASTを追加学習なしで使用した。まとめて生成する制御指令を「動作列」と呼び、その受け渡しと、物理・撮像・推論を分離する構成を実装した。</p></div>
    <div class="tags"><span>RTX 5080 / 16GB</span><span>実時間制御 約20Hz</span><span>2視点RGB</span><span>全35試行</span></div>'''))
    pages.append(page(2, '構成と資源・プロセス管理', '''
    <div class="eyebrow">DESIGN / REALTIME SYSTEM</div><h1>制御周期を保つため、<br>描画と推論を分離する。</h1>
    <div class="lanes">
      <article><span>PROCESS A</span><h2>物理・制御</h2><strong>20 Hz</strong><p>位置3・姿勢3を既存OSC（手先制御器）へ渡し、誤差と速度から関節トルクを計算。残る1指令でグリッパを開閉する。</p></article>
      <article><span>PROCESS B</span><h2>2視点カメラ</h2><strong>10 Hz 設定</strong><p>コピーした物理状態を別シーンに再現。256×256の2画像と、同時点の手先・グリッパ状態を取得する。</p></article>
      <article><span>PROCESS C</span><h2>GPU推論</h2><strong>非同期</strong><p>ROS 2で受信した画像などから動作列を生成。生成結果もROS 2で送り、受信した列を制御へ戻す。</p></article>
    </div>
    <div class="path">物理状態 → 撮像 → 観測・制御通信ノード ⇄ 方策通信ノード → VLA<br>VLAの動作列 → ROS 2受信 → 待ち行列 → OSC → 関節トルク → 物理計算</div>
    <h2>同期描画では、50msの制御周期を守れなかった</h2>
    <table><thead><tr><th>比較した構成</th><th>平均処理時間</th><th>判断・対応</th></tr></thead><tbody>
    <tr><td>NVIDIA描画・同期2カメラ</td><td>85.64 ms</td><td>20Hzの予算50msを超過</td></tr>
    <tr><td>画像なしの物理計算</td><td>8.98 ms</td><td>描画を別プロセスへ分離</td></tr>
    <tr><td>カメラ分離・待機指令のみの5秒試験</td><td>10.74 ms</td><td>95%点12.59ms、時間比約1.01</td></tr></tbody></table>
    <p class="small">CPU描画からNVIDIA D3D12へ切り替えたうえで比較した。時間比(RTF)＝シミュレーション内の時間÷実際の経過時間。推論も動かしたときの実測周期は次頁に示す。</p>
    <div class="columns compact"><article><h2>GPUメモリ</h2><p>モデルは1種類ずつGPUに載せる。PyTorchで測った最大メモリはSmol約1.47GiB、π約7.34GiB。描画とドライバの使用量は含まない。</p></article>
    <article><h2>待ち行列とCPU</h2><p>撮像は最新の状態1件のみ保持。推論は同時1件まで。モデル側CPU4スレッド、撮像側の数値計算1スレッドとし、完了待ちで制御を止めない。</p></article></div>
    <div class="callout"><b>観測時刻を保ち、入力と制御をそろえる</b><p>画像と手先状態を同じ時点で取得し、時刻を保存する。推論後も時刻を書き換えず、実行時の遅れを測る。動作列が途切れた間は手先の差分指令を0とし、グリッパ指令を保って物理計算を続ける。</p></div>'''))
    pages.append(page(3, '共通条件での比較結果', f'''
    <div class="eyebrow">PRIMARY / 20 EPISODES</div><h1>速さと成功を、<br>同じ条件で比べる。</h1>
    <p>各5配置（配置1〜5）、同じ乱数設定。制御20Hz・ROS経由で、両モデルとも動作列の先頭10指令を実行した。撮像10Hzはリアルタイム側の設定。次の推論は動作列を使い切ってから開始する。</p>
    <div class="callout thin"><b>実行方法と上限：</b>リアルタイム実行は実際の経過時間で14秒。基準実行は推論完了までシミュレーション更新を一時停止し、最大280行動（シミュレーション内の時間14秒）。起動・準備用推論2回・最初の動作列生成は計時外とした。</div>
    <table class="results"><thead><tr><th>モデル／方式</th><th>成功</th><th>生成＋通信</th><th>待機割合</th><th>最終距離</th><th>時間比</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="small">生成＋通信は、各試行で初回の動作列を除いた生成・ROS往復時間の平均を求め、その5試行平均を示した。待機割合は動作列が空だった制御更新の割合。最終距離は終了時のボウルと皿の中心間の水平距離（中央値）で、公式成功判定とは別。</p>
    <figure><img class="chart" src="{outcomes}"><figcaption>各方式5配置の結果。縦線はWilson法の95%信頼区間。固定配置の少数試行による不確かさが大きい。図中lockstepは基準実行、realtimeはリアルタイム実行。</figcaption></figure>
    <div class="columns compact"><article><h2>SmolVLA（リアルタイム）</h2><p>実測の制御頻度：<b>{smol["effective_control_hz"]["mean"]:.2f}Hz</b>／モデル推論：約{smol["model_inference_wall_ms_steady_episode_mean"]["mean"]:.0f}ms<br>観測取得から指令実行まで：約{smol["action_source_age_ms_steady_episode_mean"]["mean"]:.0f}ms</p></article>
    <article><h2>π0-FAST（リアルタイム）</h2><p>実測の制御頻度：<b>{fast["effective_control_hz"]["mean"]:.2f}Hz</b>／モデル推論：約{fast["model_inference_wall_ms_steady_episode_mean"]["mean"]:.0f}ms<br>観測取得から指令実行まで：約{fast["action_source_age_ms_steady_episode_mean"]["mean"]:.0f}ms</p></article></div>
    <p>リアルタイム実行ではSmolが成功数・最終距離とも良好だった。基準実行ではπの距離が小さい一方、成功数はSmolが多かった。位置誤差と成功数では順位が異なった。</p>'''))
    pages.append(page(4, '試行錯誤・得られた知見', f'''
    <div class="eyebrow">EXPLORATORY / 15 EPISODES</div><h1>待機を減らしても、<br>成功は増えなかった。</h1>
    <table><thead><tr><th>モデル</th><th>追加条件</th><th>成功</th><th>待機割合</th><th>最終距離の中央値</th></tr></thead><tbody>{exp_rows}</tbody></table>
    <p class="small">先読みありは残り5指令で次の動作列を要求。50行動条件はSmolの出力50指令を全実行した。同じ5配置を再利用した追加検証で、主比較とは分けて集計した。</p>
    <table class="states"><thead><tr><th>リアルタイムの実行条件</th>{''.join(f'<th>配置{i}</th>' for i in range(1,6))}</tr></thead><tbody>{state_rows}</tbody></table>
    <div class="callout"><b>動作列のつながりと、観測を取り直す頻度</b><p>先読みで待機は減ったが、両モデルとも0/5へ悪化した。動作途中の観測から作った次の列が、実行中の列とつながらない可能性がある。この単純な先読みには、計算遅延や既存動作を考慮して列をつなぐ処理（RTC）を用いていない。</p><p>50行動条件では待機が約16%まで減っても、成功は3/5のまま。新たに成功した配置と失敗した配置が2件ずつあった。長い動作列は待機を減らす一方、画像から修正する機会も減らした。</p></div>
    <h2>比較結果を支える整合性の確認</h2>
    <table class="audit"><tbody>
    <tr><th>重みの復元</th><td>学習済み重みを復元し、Smol788個・π604個のテンソルを照合。5回の評価プログラム実行の前後で、重みの不変を確認した。</td></tr>
    <tr><th>ROSの通信経路</th><td>受信した画像・動作列を実際に使用。全35試行で送受信値の照合結果が一致し、物体の正解位置は評価にのみ使用した。</td></tr>
    <tr><th>観測タイミング</th><td>同じ方式では、モデル間の初期画像・自己状態が一致。方式間では物理状態は一致したが、通常取得と描画し直した画像は異なった。</td></tr></tbody></table>
    <p class="small">診断時の手先位置差は1.33µm、グリッパ位置差は9.02µm。値が変わった画素は外部0.130%・手首4.791%で、物理状態と制御値は不変だった。撮像方法も異なるため、方式間の差を推論時間だけの効果とは断定できない。</p>'''))
    pages.append(page(5, '応用可能性・結論と再現性', '''
    <div class="eyebrow">APPLICATION / CONCLUSION</div><h1>移載作業への応用と、<br>残された課題。</h1>
    <div class="applications"><article><b>卓上の移載・仕分け</b><p>部品や容器を指定トレーへ運ぶ作業が応用候補。対象物・配置を広げるには、対応する方策と評価が必要になる。</p></article>
    <article><b>搬送設備との接続</b><p>周期の異なるカメラ・推論・制御を分離する設計を、搬送系への接続にも再利用する想定。移動対象の追従は未検証。</p></article>
    <article><b>16GB GPUでの試作</b><p>学習済みの公開モデルと既存シミュレータで試作した。追加学習なしで、メモリと処理時間を測って実行周期を設計した。</p></article></div>
    <h2>再利用した主要資産</h2>
    <table><thead><tr><th>資産</th><th>役割と固定版</th><th>利用条件の記録</th></tr></thead><tbody>
    <tr><th>SmolVLA LIBERO</th><td>604,934,176パラメータ<br>出力50時点×7次元／重み1.218GB<br>フローマッチング（10ステップ）</td><td>モデルカード：Apache-2.0</td></tr>
    <tr><th>π0-FAST LIBERO</th><td>3,449,982,704パラメータ<br>出力10時点×7次元／重み7.730GB<br>FASTによる自己回帰生成</td><td>使用版のカードにライセンス明記なし。Gemma由来の条件などを個別記録</td></tr>
    <tr><th>LeRobot / LIBERO</th><td>LeRobot0.6.2／hf-libero0.1.4<br>実装の版を固定</td><td>コードApache-2.0／MIT。外部形状データの条件は個別に確認</td></tr>
    <tr><th>実行基盤</th><td>robosuite1.4.0／MuJoCo3.3.7<br>ROS 2 Jazzy・CycloneDDS／torch2.7.1+cu128</td><td>詳細な利用条件は、コード内の資産一覧に記録</td></tr></tbody></table>
    <div class="columns compact"><article><h2>得られた結論</h2><p>10行動ずつの共通条件で、Smolのリアルタイム成功は3/5、πは1/5だった。先読みや長い動作列で待機を減らしても成功数は増えず、計算速度に加えて、列のつながりと観測更新頻度が重要だった。</p></article>
    <article><h2>限界と次の検証</h2><p>評価は1課題・固定5配置に限る。一般的なOS上で動かすため実行周期には揺らぎがある。実機・未知物体・移動対象は未検証。次はRTCを使い、動作列のつながりを改善できるか調べる。</p></article></div>
    <div class="callout thin"><b>再現性と検証可能性</b><p>モデル・依存版・初期状態・乱数・実行条件を固定した。35試行の指令と時刻、重みの不変性、ROS送受信値の一致を記録。18単体テストで待ち行列・観測・通信の動作を確認し、モデルと実行方法の影響を検討できる構成とした。</p></div>
    <div class="sources"><b>参照元と固定した版</b><br>
    <a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102">SmolVLA 学習済み重み 6721902…</a> ／
    <a href="https://huggingface.co/lerobot/pi0fast-libero/tree/840f4b503f4c09110421c33c810a85b6684fd658">π0-FAST 学習済み重み 840f4b5…</a><br>
    <a href="https://github.com/huggingface/lerobot/tree/3f797de5952432284a7c77ad2760218cb74e2f26">LeRobot 実装 3f797de…</a> ／
    <a href="https://huggingface.co/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6">LIBERO 形状データ 0b3ea86…</a>
    </div>'''))
    css = '''
    @page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;color:#203630;background:#e4e8e3;
    font-family:"Meiryo","Yu Gothic",sans-serif;font-size:14px;line-height:1.65}
    .page{width:210mm;height:297mm;padding:13mm 15mm 11mm;background:#f8f7f1;
      break-after:page;display:flex;flex-direction:column;gap:2.4mm;page-break-inside:avoid}
    .page:last-child{break-after:auto}header,footer{display:flex;justify-content:space-between;font-size:10px;color:#47695a}
    header{border-bottom:1px solid #9fb2a3;padding-bottom:3mm;letter-spacing:.5px}footer{border-top:1px solid #b7c5b7;padding-top:2mm;margin-top:auto;justify-content:flex-end}
    h1{font-size:32px;line-height:1.35;margin:0 0 1mm;letter-spacing:-.7px}h2{font-size:17px;margin:0 0 2mm}
    p{margin:0 0 1mm}.eyebrow{color:#146b5c;font-size:10px;letter-spacing:1.6px;margin-top:2mm}
    .lead{font-size:15px;line-height:1.8}.columns{display:grid;grid-template-columns:1fr 1fr;gap:6mm}
    .columns article{border-top:2px solid #a3b9a5;padding-top:3mm}.compact{gap:5mm}.compact p{font-size:13px}
    .callout{padding:3.5mm 4mm;background:#e8eee2;border-left:3px solid #146b5c;font-size:13px}
    .callout p{margin:1.5mm 0 0}.thin{padding:2.5mm 3mm;font-size:12px}.tags{display:flex;flex-wrap:wrap;gap:2mm}
    .tags span{border:1px solid #9eb49f;padding:1mm 2mm;font-size:11px}figure{margin:0}
    img{max-width:100%;display:block}img.task{width:100%;height:77mm;object-fit:contain;background:#e4e8df}
    img.chart{width:100%;height:55mm;object-fit:contain;background:white}
    figcaption{font-size:10px;color:#526557;margin-top:1mm}
    .lanes{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3mm}.lanes article{border:1px solid #789c85;padding:3mm;background:#eef2e9}
    .lanes span{font-size:9px;color:#47695a}.lanes h2{font-size:16px;margin-top:1mm}.lanes strong{font-size:22px;color:#146b5c}
    .lanes p{font-size:12px;margin-top:2mm}.path{text-align:center;border:1px dashed #789c85;font-size:12px;padding:3mm}
    table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.55}th,td{padding:1.7mm 2mm;border-bottom:1px solid #c4cfc0;text-align:left;vertical-align:top}
    thead th{background:#214e42;color:#fff;font-size:11px}tbody th{font-weight:600}td.score{font-size:19px;font-weight:700;color:#146b5c}
    th small{display:block;font-size:10px;font-weight:400}.small{font-size:10.5px;line-height:1.65;color:#526557}
    .states td{text-align:center;font-size:10px}.states th{font-size:10.5px}.ok{background:#e0eddd;color:#1b583d}.no{color:#83422d}
    .audit{font-size:11px}.audit th{width:20%}.applications{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3mm}
    .applications article{padding:3mm;background:#e8eee2}.applications b{color:#146b5c}.applications p{font-size:11px;margin-top:2mm}
    ol{margin:0;padding-left:4mm;font-size:12px}.sources{font-size:9.5px;line-height:1.8;color:#526557}a{color:#146b5c}
    @media screen{.page{margin:8mm auto;box-shadow:0 2px 18px #0002}}
    '''
    output = args.output or folder/'report.html'
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>PAI最終課題_44055 実時間VLAによるボウル移載</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>'
    output.write_text(html)
    print(json.dumps({'html':str(output),'pages':5,'source_report':'results.json','evidence_episodes':35},ensure_ascii=False))


if __name__ == '__main__':
    main()
