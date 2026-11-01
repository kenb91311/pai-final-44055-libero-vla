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
        Findings: Footer identifies the task/date. Metrics: Five pages. Confidence: QA dependent.
    Constraints: No hidden overflow. Reconsider When: Content grows.
    References: build_realtime_report.py. ADR: None.
    """
    return f'''<section class="page"><header><b>PAI最終課題_44055 / LIBERO VLA</b><span>{escape(label)}</span></header>
    {content}<footer><span>2026.11.02 · RTX 5080 · 追加学習なし</span><span>{number} / 5</span></footer></section>'''


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
    rows = ''.join(f'''<tr><th>{names[g["model"]]}<small>{g["mode"]}</small></th>
        <td class="score">{g["successes"]}/{g["n"]}</td>
        <td>{g["full_chunk_worker_roundtrip_ms_steady_episode_mean"]["mean"]:.0f} ms</td>
        <td>{g["pooled_hold_fraction"]*100:.1f}%</td>
        <td>{g["final_goal_xy_mm"]["p50"]:.1f} mm</td>
        <td>{g["realtime_factor"]["p50"]:.3f}</td></tr>''' for g in result['groups'])
    exploratory = result['scheduler_ablation']['groups'] + result['native_horizon_smol']['groups']
    exp_rows = ''.join(f'''<tr><th>{names[g["model"]]}</th><td>{"E50 / 先読み0" if i==2 else "E10 / 先読み5"}</td>
        <td class="score">{g["successes"]}/{g["n"]}</td><td>{g["pooled_hold_fraction"]*100:.1f}%</td>
        <td>{g["final_goal_xy_mm"]["p50"]:.1f} mm</td></tr>''' for i, g in enumerate(exploratory))
    variants = [
        ('Smol / E10・先読み0', result['episodes'], 'smolvla'),
        ('Smol / E10・先読み5', result['scheduler_ablation']['episodes'], 'smolvla'),
        ('Smol / E50・先読み0', result['native_horizon_smol']['episodes'], 'smolvla'),
        ('π0-FAST / E10・先読み0', result['episodes'], 'pi0fast'),
        ('π0-FAST / E10・先読み5', result['scheduler_ablation']['episodes'], 'pi0fast')]
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
    <p class="lead">LIBERO向け学習済みVLAでPandaを操作。<br>画像認識・行動列の推論・ROS 2通信・物理制御を接続し、<br>推論中も進むロボットの操作を評価する。</p>
    <figure><img class="task" src="{task}"><figcaption>実際のSmolVLA実行映像から。左：外部カメラ ／ 右：手首カメラ</figcaption></figure>
    <div class="columns"><article><h2>タスク</h2><p>皿とラメキンの間にある黒いボウルを、Pandaでつかみ、皿へ置く。LIBERO Spatial・task0の既存アセットと公式成功判定を使う。</p></article>
    <article><h2>確かめたいこと</h2><p>小さいモデルの低遅延は、実時間での成功や目標誤差に有利か。モデルの能力と、実行システムの時間制約を分けて確認する。</p></article></div>
    <div class="callout"><b>公開方策を、実時間の実行系へつなぐ。</b><p>VLAは画像・言語・自己状態から複数時点の行動列を推論する。公開重み・公式前処理・既存制御器を再利用し、物理・撮像・推論の分離と時刻付き観測、ROS上の行動経路を実装した。追加学習なしで検証する。</p></div>
    <div class="tags"><span>RTX 5080 / 16GB</span><span>実時間制御 約20Hz</span><span>2視点RGB</span><span>全35試行</span></div>'''))
    pages.append(page(2, '構成と資源・プロセス管理', '''
    <div class="eyebrow">DESIGN / REALTIME SYSTEM</div><h1>遅い処理を、<br>制御の時計から分ける。</h1>
    <div class="lanes">
      <article><span>PROCESS A</span><h2>物理・制御</h2><strong>20 Hz</strong><p>画像なしで物理step。7D相対手先指令を既存OSCへ渡し、位置・速度のfeedbackから関節トルクを出す。</p></article>
      <article><span>PROCESS B</span><h2>2視点カメラ</h2><strong>10 Hz 設定</strong><p>同一シーンに時刻付き物理snapshotを復元。RGB256×256を2枚と、同じ時点の自己状態を生成。</p></article>
      <article><span>PROCESS C</span><h2>GPU推論</h2><strong>非同期</strong><p>ROSで受信した画像・状態・言語指示からchunkを生成。ROSで受信したchunkを行動queueへ戻す。</p></article>
    </div>
    <div class="path">物理snapshot → カメラ → ROS Sensor/Broker ⇄ PolicyTransport → VLA<br>VLA chunk → ROS受信 → action queue → OSC → 関節トルク → 物理</div>
    <h2>描画が、50msの制御予算を超えていた</h2>
    <table><thead><tr><th>切り分けた構成</th><th>物理step平均</th><th>判断・対応</th></tr></thead><tbody>
    <tr><td>NVIDIA描画・同期2カメラ</td><td>85.64 ms</td><td>20Hzの予算50msを超過</td></tr>
    <tr><td>画像なしの物理</td><td>8.98 ms</td><td>描画を別processへ分離</td></tr>
    <tr><td>カメラ分離・holdの5秒試験</td><td>10.74 ms</td><td>p95 12.59ms、RTF約1.01</td></tr></tbody></table>
    <p class="small">上表は構成切り分け用の測定。モデル同時実行の実周期は次ページ。WSLのCPU描画llvmpipeを検出し、NVIDIA D3D12経路へ変更したうえで分離した。</p>
    <div class="columns compact"><article><h2>GPU・メモリ</h2><p>モデルは1種類ずつ常駐。PyTorchピークはSmol約1.47GiB、π約7.34GiB。描画・ドライバ分はこの値に含まない。</p></article>
    <article><h2>待ち行列・CPU</h2><p>カメラはqueue長1で最新snapshotのみ保持。推論は同時に1件。モデルCPU threads4・カメラBLAS1。物理は非同期pollで待たない。</p></article></div>
    <div class="callout"><b>リアルタイム・観測への追従を重視</b><p>画像と自己状態を同じsnapshotに揃え、取得時刻を保持。推論完了を新しい観測時刻にしない。空queueでは姿勢差分0＋グリッパ指令維持とし、物理更新を続ける。</p></div>'''))
    pages.append(page(3, '共通条件での比較結果', f'''
    <div class="eyebrow">PRIMARY / 20 EPISODES</div><h1>速さと成功を、<br>同じ条件で比べる。</h1>
    <p>各5初期配置（状態1〜5）、seed固定。物理20Hz、実時間側カメラ10Hz、ROS経由。両者とも生成列の先頭10行動を実行し、使い切ってから次を要求。</p>
    <div class="callout thin"><b>時間予算：</b>実時間は14壁時計秒。lockstepは最大280行動＝物理14秒で、推論・描画中は物理停止。起動・warmup2回・最初のchunk生成は計時外。</div>
    <table class="results"><thead><tr><th>モデル／方式</th><th>成功</th><th>worker時間</th><th>待機割合</th><th>終端XY</th><th>RTF</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="small">worker時間：各episodeの定常chunk平均を5回で平均。ROS往復・モデル処理を含み、親IPC/pollは除く。終端XYは中央値であり公式成功判定とは別。RTF＝物理時間／壁時計時間。</p>
    <figure><img class="chart" src="{outcomes}"><figcaption>全5状態の個別結果と幅の大きいWilson95%区間。単一課題の小標本として読む。</figcaption></figure>
    <div class="columns compact"><article><h2>SmolVLA / realtime</h2><p><b>{smol["effective_control_hz"]["mean"]:.2f}Hz</b>、モデル処理約{smol["model_inference_wall_ms_steady_episode_mean"]["mean"]:.0f}ms。指令元の観測ageは約{smol["action_source_age_ms_steady_episode_mean"]["mean"]:.0f}ms。</p></article>
    <article><h2>π0-FAST / realtime</h2><p><b>{fast["effective_control_hz"]["mean"]:.2f}Hz</b>、モデル処理約{fast["model_inference_wall_ms_steady_episode_mean"]["mean"]:.0f}ms。指令元の観測ageは約{fast["action_source_age_ms_steady_episode_mean"]["mean"]:.0f}ms。</p></article></div>
    <p>この条件ではSmolが実時間の成功数・終端誤差で良好だった。πはlockstepのXY中央値が小さいが、成功数はSmolより少ない。性能を一つの「精度」でまとめない。</p>'''))
    pages.append(page(4, '試行錯誤・得られた知見', f'''
    <div class="eyebrow">EXPLORATORY / 15 EPISODES</div><h1>待機を減らしても、<br>成功は増えなかった。</h1>
    <table><thead><tr><th>モデル</th><th>追加する条件</th><th>成功</th><th>待機割合</th><th>終端XY中央値</th></tr></thead><tbody>{exp_rows}</tbody></table>
    <p class="small">E＝実行する行動数。先読み5は残量5で次を要求。E50はSmolが生成した50行動を全実行。どちらも主比較と同じ5状態を再使用した探索条件。共通E10比較へ混ぜない。</p>
    <table class="states"><thead><tr><th>実時間の条件</th>{''.join(f'<th>状態{i}</th>' for i in range(1,6))}</tr></thead><tbody>{state_rows}</tbody></table>
    <div class="callout"><b>行動列の整合と、観測を取り直す頻度</b><p>先読みは待機を減らしたが両モデル0/5へ悪化。動作途中の観測から作った次の列が、実行中の列とうまくつながらない可能性がある。単純FIFO先読みは遅延を条件にしたRTCではない。</p><p>E50はSmolの待機を約16%へ減らしても3/5のまま。成功へ2配置、失敗へ2配置が入れ替わった。長い予測を使う効果と、視覚修正が減る不利益を分ける必要がある。</p></div>
    <h2>境界を検証してから、モデル差を読む</h2>
    <table class="audit"><tbody>
    <tr><th>重み復元</th><td>固定checkpointを厳密に復元し、Smol788・π604テンソルを照合。全5実行で重みの前後不変を確認。</td></tr>
    <tr><th>ROSの実経路</th><td>受信した画像・chunkを実際に消費。全35試行の送受信hashが一致。モデルへ物体真値を渡さない。</td></tr>
    <tr><th>観測の位相</th><td>同方式のモデル間は初期sensor完全一致。方式間は物理状態一致だが、step cacheと強制更新の画像は異なる。</td></tr></tbody></table>
    <p class="small">位相診断：同一物理状態でEEF位置差1.33µm、gripper差9.02µm。変更pixelは外部0.130%・手首4.791%。物理状態・ctrl不変。lockstep対実時間はシステム比較であり、純粋なモデル遅延の因果効果とは断定しない。</p>'''))
    pages.append(page(5, '応用可能性・結論と再現性', '''
    <div class="eyebrow">APPLICATION / CONCLUSION</div><h1>行動の速さを、<br>タスクの確かさへつなぐ。</h1>
    <div class="applications"><article><b>卓上の移載・仕分け</b><p>部品や容器を指定トレーへ運ぶ作業が応用候補。対象物・配置を広げるには、対応する方策と評価が必要になる。</p></article>
    <article><b>搬送設備との接続</b><p>周期の異なるカメラ・推論・制御を分離する設計は、搬送系との接続にも適用できる。移動対象の追従は未検証。</p></article>
    <article><b>16GB GPUでの試作</b><p>公開方策と既存シミュレータで、学習計算を追加せず試作。メモリと処理時間を測り、実行周期を設計できる。</p></article></div>
    <h2>再利用した主要資産</h2>
    <table><thead><tr><th>資産</th><th>使用した内容・版</th><th>利用条件の記録</th></tr></thead><tbody>
    <tr><th>SmolVLA LIBERO</th><td>604,934,176 parameters／生成50×7<br>重み1.218GB、flow10steps</td><td>モデルカード：Apache-2.0</td></tr>
    <tr><th>π0-FAST LIBERO</th><td>3,449,982,704 parameters／生成10×7<br>重み7.730GB、FAST自己回帰</td><td>選択cardに明示欄なし。Gemma由来の条件・tokenizerを個別記録</td></tr>
    <tr><th>LeRobot / LIBERO</th><td>0.6.2・commit固定／hf-libero0.1.4</td><td>コードApache-2.0／MIT。外部meshの宣言は個別に確認</td></tr>
    <tr><th>実行基盤</th><td>robosuite1.4.0／MuJoCo3.3.7<br>ROS2 Jazzy・CycloneDDS／torch2.7.1+cu128</td><td>コード・モデル・アセットを一括のlicenseにしない</td></tr></tbody></table>
    <div class="columns compact"><article><h2>得られた結論</h2><p>共通E10条件ではSmolの実時間成功3/5、πは1/5。先読みや長い実行列で待機を減らしても成功数は増えず、計算速度に加えて、行動列の整合と観測更新頻度が重要だった。</p></article>
    <article><h2>限界と次の検証</h2><p>結果は1課題・固定5配置のsoft realtime評価。実機・未知物体・移動対象の操作は未検証。次は、計算遅延と既存行動を条件にするRTCで、行動列のつながりを評価する。</p></article></div>
    <div class="callout thin"><b>再現性と検証可能性</b><p>モデル・依存版・初期状態・乱数と実行条件を固定。35試行の行動・時刻、重み不変、ROS送受信hashを記録し、18単体テストでqueueや観測・通信の契約を確認した。公開モデルの能力と実行系の影響を分けて検討できる構成とした。</p></div>
    <div class="sources"><b>一次資料・固定revision</b><br>
    <a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102">SmolVLA checkpoint 6721902…</a> ／
    <a href="https://huggingface.co/lerobot/pi0fast-libero/tree/840f4b503f4c09110421c33c810a85b6684fd658">π0-FAST checkpoint 840f4b5…</a><br>
    <a href="https://github.com/huggingface/lerobot/tree/3f797de5952432284a7c77ad2760218cb74e2f26">LeRobot commit 3f797de…</a> ／
    <a href="https://huggingface.co/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6">LIBERO assets 0b3ea86…</a>
    </div>'''))
    css = '''
    @page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;color:#203630;background:#e4e8e3;
    font-family:"Meiryo","Yu Gothic",sans-serif;font-size:14px;line-height:1.65}
    .page{width:210mm;height:297mm;padding:13mm 15mm 11mm;background:#f8f7f1;
      break-after:page;display:flex;flex-direction:column;gap:3.2mm;page-break-inside:avoid}
    .page:last-child{break-after:auto}header,footer{display:flex;justify-content:space-between;font-size:10px;color:#47695a}
    header{border-bottom:1px solid #9fb2a3;padding-bottom:3mm;letter-spacing:.5px}footer{border-top:1px solid #b7c5b7;padding-top:2mm;margin-top:auto}
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
