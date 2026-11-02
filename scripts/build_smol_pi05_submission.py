"""Build the evaluator-facing SmolVLA/pi05 report from independently audited measurements.

Args: --output: fresh report directory, default the audited submission directory.
Returns: Self-contained six-page HTML, a measured plot and a numeric/source manifest.
Raises: ValueError for failed provenance/output checks; IO errors for missing evidence.
Design: Context: The final task report needs system design and measured comparisons.
    Concern: Task timings, isolated profiling and continuity ablations have different scopes.
    Decision: Separate those scopes by page and render tables directly from audited JSON.
    Alternatives: Edit the historical FAST report or copy web screenshots.
    Quality Attributes: Readability, traceability. Trade-offs: Browser PDF rendering is separate.
Evidence: Internal: Independently recomputed twenty task episodes and clean continuity series.
    External: Pinned japanese-tech-writing skill. Findings: Pi05 is faster in these implementations.
Constraints: No model/GPU execution, original report mutation or personal identifiers in report.
Reconsider When: Model weights, task conditions or accepted evidence change.
References: docs/experiments/realtime-inference-profiling.md. ADR: None.
"""
import argparse
import base64
import hashlib
from html import escape
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/realtime-vla/submission-smol-pi05'
NAMES = {'smolvla': 'SmolVLA', 'pi05': 'π0.5'}
REPO = 'https://github.com/kenb91311/pai-final-44055-libero-vla'
STYLE = '''
@page{size:A4;margin:0}*{box-sizing:border-box}html,body{margin:0;background:#e7e9e1;color:#203630;font-family:Meiryo,"Yu Gothic",sans-serif;font-synthesis:none}body{font-size:9.4pt;line-height:1.8}.page{width:210mm;height:297mm;padding:13mm 16mm 14mm;background:#fffefa;position:relative;break-after:page}.page:last-child{break-after:auto}header{height:10mm;border-bottom:1px solid #cdd4c8;display:flex;justify-content:space-between;align-items:start;font-size:7.1pt;margin-bottom:7mm;color:#58675d}header b{font-weight:600;color:#203630}footer{position:absolute;left:16mm;right:16mm;bottom:8mm;border-top:1px solid #cdd4c8;padding-top:2mm;font-size:7pt;color:#58675d;display:flex;justify-content:flex-end}h1{font-size:22pt;line-height:1.5;letter-spacing:-.02em;font-weight:600;margin:1mm 0 4mm}h2{font-size:12pt;line-height:1.6;margin:5mm 0 2mm;font-weight:600}h3{font-size:10pt;line-height:1.6;margin:0 0 1.5mm}p,figure{margin:0}p+p{margin-top:2.5mm}.eyebrow{font:7pt Consolas,monospace;letter-spacing:.1em;color:#146b5c;margin-bottom:2mm}.subhead{font-size:11pt;color:#58675d;margin-bottom:4mm}.small{font-size:7.7pt;line-height:1.8;color:#58675d}.tiny{font-size:6.8pt;line-height:1.8;color:#58675d}strong{font-weight:700}a{color:#146b5c;text-decoration:none}figure img{display:block;width:100%;height:auto}figcaption{font-size:7.3pt;line-height:1.7;color:#58675d;margin-top:2mm}.columns{display:grid;grid-template-columns:1fr 1fr;gap:5mm}.scene-pair{display:grid;grid-template-columns:1fr 1fr;gap:3mm;margin:4mm 0 2mm}.scene-pair img{width:100%;height:auto;border:1px solid #cdd4c8}.scene-pair b{display:block;font-size:8pt;margin-bottom:1mm}.callout{background:#e4ede3;border-left:3px solid #146b5c;padding:3mm 4mm;margin-top:4mm}.callout b{display:block;font-size:10pt;margin-bottom:1mm}.callout p{font-size:8.4pt;line-height:1.8}.warm{background:#f2e7db;border-color:#9b4d2b}.cards{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3mm;margin-top:4mm}.card{border-top:2px solid #91ad96;background:#f0f3eb;padding:3mm}.card .label{font-size:7.5pt;color:#58675d}.card .value{font-size:20pt;line-height:1.5;color:#146b5c}.card .value small{font-size:9pt}.card p{font-size:7.6pt;line-height:1.7}.spec{display:grid;grid-template-columns:1fr 1fr;gap:0 5mm;margin-top:3mm}.spec>div{border-bottom:1px solid #dce1d7;padding:2mm 0;font-size:8pt}.spec b{display:inline-block;min-width:20mm;font-weight:600}table{border-collapse:collapse;width:100%;font-size:8.1pt;line-height:1.75;margin-top:3mm;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #cdd4c8;padding:2mm 2mm;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}thead{background:#edf1e8;color:#58675d;font-size:7.2pt}thead th{font-weight:500}tbody th{font-weight:500}td.score{font-weight:700;color:#146b5c;font-size:10pt}tr.highlight{background:#eaf1e5}.states td{text-align:center}.states th{width:60mm}.ok{color:#146b5c}.no{color:#9b4d2b}.caption{font-size:7.4pt;color:#58675d;margin-top:2mm;line-height:1.8}.diagram{margin-top:3mm;border:1px solid #cdd4c8;background:#f7f8f2;padding:3mm}.diagram svg{display:block;width:100%;height:auto}.engineering{display:grid;grid-template-columns:1fr 1fr;gap:4mm;margin-top:4mm}.engineering article{padding:3mm;border:1px solid #cdd4c8}.engineering p{font-size:8.1pt;line-height:1.8}.timeline{display:grid;grid-template-columns:31mm 1fr;gap:3mm;align-items:center;margin-top:3mm}.timeline>b{font-size:8pt}.track{display:flex;height:10mm;align-items:stretch;font-size:7.2pt}.track>span{display:flex;align-items:center;justify-content:center;padding:1mm}.run{background:#dce9d9;color:#146b5c}.wait{background:#eee4da;color:#8b4a2d}.flow{display:flex;justify-content:space-between;gap:3mm;margin-top:3mm}.flow>div{flex:1;padding:2.5mm;background:#edf1e8;font-size:8.2pt;text-align:center}.flow>span{align-self:center;color:#146b5c}.section-note{font-size:8.2pt;border-bottom:1px solid #cdd4c8;padding-bottom:3mm}.scope-label{display:inline-block;font-size:7.3pt;padding:1mm 2mm;background:#e4ede3;color:#146b5c;margin-bottom:1.5mm}.profile-metrics{display:flex;gap:4mm;margin:2mm 0}.profile-metrics>div{flex:1;padding:3mm;background:#203630;color:#fffefa;font-size:8pt}.profile-metrics strong{font-size:23pt;font-weight:500;display:block;line-height:1.6}.profile-metrics small{font-size:8pt}.profile-plot{margin-top:2mm}.profile-plot img{width:100%;max-height:43mm;object-fit:contain}.trial-table th{width:29mm}.trial-table td{text-align:left;font-size:8pt}.refs{font-size:6.8pt;line-height:1.85;color:#58675d;margin-top:3mm}.refs a{overflow-wrap:anywhere}.refs div{margin-top:1mm}.tight h2{margin-top:4mm}.asset-table{font-size:7.6pt}.asset-table td{text-align:left}.summary-line{font-size:9.2pt;line-height:1.9}.divider{border-top:1px solid #cdd4c8;margin:4mm 0}.no-break{white-space:nowrap}@media screen{.page{margin:15px auto;box-shadow:0 2px 12px #0002}}
'''


def read_json(path):
    """Read a committed-schema measurement without importing model libraries.

    Args: path: existing JSON. Returns: Parsed JSON. Raises: IO/JSON errors.
    Design: Context: Rendering must be CPU only. Concern: Runtime imports have side effects.
        Decision: Standard-library read. Alternatives: Model loaders. Quality Attributes: Isolation.
        Trade-offs: Relies on prior numeric audits.
    Evidence: Internal: evidence-primary/continuity.json. External: None. Findings: Audit status checked by main.
    Constraints: Read only. Reconsider When: JSON schema changes. References: main. ADR: None.
    """
    return json.loads(Path(path).read_text())


def embedded_png(path):
    """Embed an existing measured PNG in portable report HTML.

    Args: path: PNG file. Returns: Data URI. Raises: IO errors.
    Design: Context: Windows renders a WSL-authored report. Concern: Local URLs can leak paths.
        Decision: Embed exact bytes. Alternatives: file URLs. Quality Attributes: Portability/privacy.
        Trade-offs: Larger HTML.
    Evidence: Internal: frame-provenance.json. External: RFC2397. Findings: No image synthesis.
    Constraints: Do not alter original media. Reconsider When: Image sizes grow. References: build_pages. ADR: None.
    """
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def plot_profile(rows, output):
    """Render disjoint measured CUDA-stream intervals as a standalone comparison figure.

    Args: rows: two audited profile records; output: directory for PNG/SVG.
    Returns: Exact plotted data. Raises: ValueError on negative residual or unverified outputs.
    Design: Context: Inclusive nested stages overlap. Concern: Double counting and GPU-busy mislabeling.
        Decision: Stack vision, prefill, full flow and root residual; expert stays a child in prose.
        Alternatives: Stack every inclusive stage. Quality Attributes: Numeric integrity.
        Trade-offs: Stream elapsed includes host launch gaps.
    Evidence: Internal: Independently audited primary profiling reports. External: Matplotlib.
        Findings: Flow accounts for the large interval difference.
    Constraints: No kernel-busy or causal-ablation claim. Reconsider When: Concurrent streams supported.
    References: profiling_hooks.py. ADR: None.
    """
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Vision encoder', 'VLM prefix', 'Flow action generation', 'Other']
    colors = ['#9a642f', '#6b8777', '#146b5c', '#c5cdbf']
    data=[]
    for row in rows:
        stages=row['instrumented_cuda_stream_elapsed_ms_mean']
        values=[stages['vision_tower'],stages['vlm_prefix_prefill'],stages['flow_denoise_step']]
        values.append(stages['request_inclusive']-sum(values))
        if not row['all_paired_outputs_exact'] or min(values)<0:
            raise ValueError('Profile output/partition verification failed')
        data.append({'model':row['model'],'stages_ms':dict(zip(labels,values))})
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#cdd4c8','text.color':'#203630','axes.labelcolor':'#58675d','xtick.color':'#58675d','ytick.color':'#203630','svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(9.5,2.5),layout='constrained');fig.set_facecolor('#fffefa');ax.set_facecolor('#fffefa')
        for y,model in enumerate(data):
            left=0
            for label,color in zip(labels,colors):
                value=model['stages_ms'][label];ax.barh(y,value,left=left,height=.42,color=color,label=label if y==0 else None)
                if label=='Flow action generation':ax.text(left+value/2,y,f'{value:.1f} ms',ha='center',va='center',color='white',fontsize=12)
                left+=value
            ax.text(left+9,y,f'{left:.1f}',va='center',fontsize=10)
        ax.set_yticks([0,1],[NAMES[r['model']] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,650);ax.set_ylim(1.6,-.65);ax.set_xlabel('CUDA-stream elapsed per request (ms)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncols=4,frameon=False,fontsize=9)
        fig.savefig(output/'inference-stages.png',dpi=180);fig.savefig(output/'inference-stages.svg');plt.close(fig)
    return data


def page(number, label, body):
    """Wrap one fixed A4 topic with page-number-only footer.

    Args: number/label/body: page content. Returns: HTML section. Raises: None.
    Design: Context: Short evaluator report. Concern: Broken diagrams across page boundaries.
        Decision: Six explicit pages, checked at actual PDF render. Alternatives: Unbounded flow.
        Quality Attributes: Scanability. Trade-offs: Requires overflow inspection.
    Evidence: Internal: Browser geometry checks. External: CSS paged media. Findings: Pending render.
    Constraints: No user-instruction or editing notes. Reconsider When: More scope added.
    References: build_pages. ADR: None.
    """
    return f'<section class="page"><header><b>PAI最終課題_44055 / LIBERO VLA</b><span>{escape(label)}</span></header>{body}<footer><span>{number} / 6</span></footer></section>'


def build_pages(primary, continuity, output):
    """Compose task, system, synchronous comparison, profiling, continuity and lessons pages.

    Args: primary/continuity: audited records; output: figure directory.
    Returns: Six HTML sections. Raises: KeyError for incomplete audited records.
    Design: Context: Reader evaluates creative task/system construction, not conversation compliance.
        Concern: Unsupported claims of general model superiority or RTC success.
        Decision: State scope and definitions adjacent to each table; preserve failed outcomes.
        Alternatives: Chronological diary or benchmark-rank assertions.
        Quality Attributes: Readability, honesty. Trade-offs: Some technical detail stays in linked repository.
    Evidence: Internal: Twenty task episodes, thirty clean RTC episodes and isolated profiling.
        External: Fixed model/library references. Findings: Same lockstep success, different realtime counts.
    Constraints: Different backbones/measurement clocks remain separate. Reconsider When: New trials added.
    References: evidence-primary.json; evidence-continuity.json. ADR: None.
    """
    groups={(r['model'],r['mode']):r for r in primary['groups']}
    sm,pi=groups['smolvla','realtime'],groups['pi05','realtime']
    profile={r['model']:r for r in continuity['inference_profile']}
    body1=f'''
    <div class="eyebrow">AI × ROBOTICS / TASK & OBJECTIVE</div>
    <h1>VLA推論とアーム制御の<br>非同期化</h1>
    <p class="subhead">SmolVLAとπ0.5による、ボウル移載タスクの比較</p>
    <p>ロボットが画像と言語指示から行動列を生成し、黒いボウルをつかんで皿へ置く。LIBERO向けに調整済みの公開モデルを用い、本制作では追加学習せず、センサ・ROS 2・方策・シミュレータ内の制御を接続した。</p>
    <div class="scene-pair"><figure><b>① 開始時の配置</b><img src="{embedded_png(output/'task-initial.png')}" alt="ボウル、ラメキン、皿とPandaの初期配置"></figure><figure><b>② ボウルを皿へ移載した状態</b><img src="{embedded_png(output/'task-complete.png')}" alt="π0.5非同期実行でボウルを皿へ置いた状態"></figure></div>
    <p class="tiny">π0.5・非同期・配置1の実行記録。各画像の左は外部カメラ、右は手首カメラ。成否はLIBEROの公式判定を使用。</p>
    <h2>目的と検証した仮説</h2>
    <p>推論中も制御周期を維持するシステムを構築し、モデルの処理時間と操作成功の関係を調べた。計画時は、小型SmolVLAの低遅延が非同期操作に有利と予想した。そこで、同じ課題・初期配置・行動実行数でπ0.5と比較した。</p>
    <div class="cards"><article class="card"><div class="label">同期実行の成功</div><div class="value">5/5 <small>：5/5</small></div><p>SmolVLA：π0.5<br>両者とも全配置で成功</p></article><article class="card"><div class="label">非同期実行の成功</div><div class="value">3/5 <small>：5/5</small></div><p>SmolVLA：π0.5<br>実経過14秒以内の評価</p></article><article class="card"><div class="label">非同期中の方策生成・出力処理</div><div class="value">{sm['mean_of_episode_mean_model_call_ms']:.0f}<small> / {pi['mean_of_episode_mean_model_call_ms']:.0f} ms</small></div><p>SmolVLA / π0.5<br>入力前処理を除く試行平均</p></article></div>
    <div class="callout"><b>制作の中心：モデルを実時間システムへ接続する設計</b><p>既存のPanda・接触物理・学習済み方策を再利用し、撮像、推論、制御を別プロセスへ分離した。取得時刻の管理、行動の待ち行列、ROS受信値の照合を実装し、処理時間と操作結果を対応づけて検証した。</p></div>
    <div class="spec"><div><b>環境</b>LIBERO Spatial task0 / MuJoCo</div><div><b>機体</b>Pandaアーム＋グリッパ</div><div><b>計算機</b>RTX 5080・VRAM 16GB / WSL2</div><div><b>評価範囲</b>既知の1課題・固定5配置</div></div>
    '''
    system='''<div class="diagram"><svg viewBox="0 0 690 232" role="img" aria-label="シミュレータ、撮像、ROS、VLA、待ち行列、OSCの接続図"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 6 3 0 6" fill="#146b5c"/></marker></defs><g font-family="Meiryo,sans-serif" fill="#203630"><rect x="9" y="12" width="200" height="91" rx="3" fill="#e4ede3" stroke="#a6bca7"/><text x="23" y="35" font-size="13">シミュレータ制御 / 20Hz</text><text x="23" y="58" font-size="11">待ち行列 → 7次元差分指令</text><text x="23" y="80" font-size="11">既存OSC → 関節トルク → MuJoCo</text><rect x="249" y="12" width="187" height="91" rx="3" fill="#f5efe5" stroke="#cbb397"/><text x="263" y="35" font-size="13">別プロセスの撮像 / 上限10Hz</text><text x="263" y="58" font-size="11">同じ状態を別シーンへ復元</text><text x="263" y="80" font-size="11">2画像＋同時点の自己状態</text><rect x="476" y="12" width="202" height="91" rx="3" fill="#eaf0e5" stroke="#a6bca7"/><text x="490" y="35" font-size="13">ROS 2：Sensor/Broker</text><text x="490" y="58" font-size="11">画像・自己状態を送信</text><text x="490" y="80" font-size="11">生成行動列を受信・照合</text><rect x="475" y="150" width="203" height="65" rx="3" fill="#eaf0e5" stroke="#a6bca7"/><text x="490" y="175" font-size="13">ROS 2：PolicyTransport</text><text x="490" y="197" font-size="11">受信した観測をモデルへ渡す</text><rect x="249" y="150" width="187" height="65" rx="3" fill="#146b5c"/><text x="264" y="175" font-size="13" fill="#fffefa">GPU：SmolVLA / π0.5</text><text x="264" y="197" font-size="11" fill="#e4ede3">言語指示＋観測 → 50行動</text><path d="M210 56H242M438 56H468M575 104V143M468 181H443" fill="none" stroke="#146b5c" stroke-width="2" marker-end="url(#arrow)"/><path d="M436 198H467 M663 150V111 M475 90H451V120H109V110" fill="none" stroke="#146b5c" stroke-dasharray="5 3" stroke-width="1.5" marker-end="url(#arrow)"/><text x="9" y="172" font-size="11">戻りの行動列もROSを通し、</text><text x="9" y="191" font-size="11">受信値を制御へ使用</text><text x="254" y="129" font-size="10" fill="#58675d">実線：観測 ／ 破線：ROS往復後の行動</text></g></svg></div>'''
    # The feedback label explicitly indicates ROS transport before queue consumption.
    body2=f'''<div class="eyebrow">SYSTEM / SENSOR → POLICY → ACTUATOR</div><h1>センサ・ROS・制御の接続と<br>計算資源の管理</h1>
    <p>外部・手首画像、ロボット自身の状態、言語指示をVLAへ入力する。モデルは相対位置3・姿勢3・グリッパ1の行動列を出力し、既存の操作空間制御器（OSC、位置・姿勢のPD制御）が関節トルクへ変換する。</p>{system}
    <p class="caption">行動列は複数時点の指令をまとめたもの（action chunk）。画像・自己状態と行動列は、実際にROSの別ノード間で送受信し、受信した値を使用する。物体の正解位置は評価器だけが使用する。</p>
    <div class="engineering"><article><h3>撮像を制御周期から分離</h3><p>2カメラの同期描画は平均85.64msで、20Hzの制御予算50msを超えた。画像なしのシミュレータ更新は8.98msだったため、状態を複製するカメラプロセスへ描画を分離した。</p></article><article><h3>取得時刻と待ち行列の管理</h3><p>画像と自己状態を同じsnapshotから作り、取得時刻を保持する。カメラの待ち行列は長さ1、推論要求は同時に1件とし、古い観測の蓄積を抑えた。</p></article></div>
    <h2>ハードウェア・プロセス管理の実装</h2>
    <table class="asset-table"><tbody><tr><th>GPU</th><td>モデルを1種類ずつ実行。RTC追加系列と推論内訳計測では、別実験とGPU利用を調整し、全体使用量も監視した。</td></tr><tr><th>CPU・メモリ</th><td>推論側4スレッド、撮像側BLAS 1スレッド。ROSコンテナはCPU 2コア相当・メモリ2GiBへ制限した。</td></tr><tr><th>主比較のモデル最大割当</th><td>SmolVLA {sm['model_process_peak_cuda_allocated_GiB']:.2f}GiB / π0.5 {pi['model_process_peak_cuda_allocated_GiB']:.2f}GiB。PyTorchのモデルプロセス内の最大値で、描画・デスクトップ込みのGPU全体とは範囲が異なる。</td></tr><tr><th>指令が未着のとき</th><td>位置・姿勢の差分をゼロとし、グリッパ指令のみ維持。相対移動指令を繰り返して移動を続けない。</td></tr></tbody></table>
    <div class="callout"><b>低位制御は既存実装を再利用</b><p>LIBERO / robosuiteのOSCとグリッパ制御を用い、MuJoCoの接触物理へ反映した。独自IKの追加ではなく、認識から既存制御器までの接続と、非同期時の時刻・資源管理を工夫した。</p></div>'''
    table_rows=[]
    for model,mode in [('smolvla','lockstep'),('pi05','lockstep'),('smolvla','realtime'),('pi05','realtime')]:
        r=groups[model,mode];hold='—' if mode=='lockstep' else f"{r['pooled_hold_fraction']*100:.1f}%"
        table_rows.append(f'<tr class="{"highlight" if mode=="realtime" else ""}"><th>{NAMES[model]} / {"同期" if mode=="lockstep" else "非同期"}</th><td class="score">{r["successes"]}/5</td><td>{r["mean_of_episode_mean_model_call_ms"]:.0f}ms</td><td>{r["mean_of_episode_mean_worker_with_ros_ms"]:.0f}ms</td><td>{hold}</td><td>{r["mean_effective_control_hz"]:.2f}Hz</td></tr>')
    state_rows=[]
    for model in ('smolvla','pi05'):
        r=groups[model,'realtime'];cells=''.join(f'<td class="{"ok" if r["successes_by_initial_state"][str(i)] else "no"}">{"成功" if r["successes_by_initial_state"][str(i)] else "未達"}</td>' for i in range(1,6));state_rows.append(f'<tr><th>{NAMES[model]} / 非同期</th>{cells}</tr>')
    body3=f'''<div class="eyebrow">MAIN COMPARISON / 20 TASK EPISODES</div><h1>同期・非同期実行の比較</h1>
    <p><strong>同期（lockstep）</strong>は推論中にシミュレーション時間を止める。<strong>非同期（realtime）</strong>は推論中も実経過時間に合わせ、50msごとに制御と物理演算を進める。</p>
    <div class="timeline"><b>同期</b><div class="track"><span class="run" style="width:30%">10行動を実行</span><span class="wait" style="width:40%">推論中は物理進行を停止</span><span class="run" style="width:30%">次の10行動</span></div></div>
    <div class="timeline"><b>非同期・主比較</b><div class="track"><span class="run" style="width:30%">10行動を実行</span><span class="wait" style="width:40%">推論中も待機指令で物理を更新</span><span class="run" style="width:30%">到着後に実行</span></div></div>
    <p class="tiny">概念図。帯の長さは実測時間を表さない。主比較では先読みせず、行動列を使い切ってから次を要求する。</p>
    <h2>共通条件と評価時間</h2><p class="small">両モデルとも生成50行動・flow反復10回。先頭10行動を実行し、同じ配置1〜5・乱数を使用した。同期は最大280制御step（14シミュレーション秒）、非同期は最大14実秒または280step。重み読込み・準備運転2回・初回行動列の生成は計時外とした。</p>
    <table><thead><tr><th>モデル / 方式</th><th>成功</th><th>方策生成・出力</th><th>生成＋ROS</th><th>待機</th><th>実測制御頻度</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
    <p class="caption">時間は初回を除く各試行平均の5試行平均。方策生成・保存済み出力処理・CUDA同期までの実経過時間で、入力前処理を除く。生成＋ROSは入力前処理と通信を含み、親プロセスの待ちを除く。同期の推論待ちは停止中なので待機tickとして数えない。</p>
    <table class="states"><thead><tr><th>非同期の個別結果</th><th>配置1</th><th>配置2</th><th>配置3</th><th>配置4</th><th>配置5</th></tr></thead><tbody>{''.join(state_rows)}</tbody></table>
    <div class="columns" style="margin-top:4mm"><article><h3>観測から実行までの遅れ</h3><p class="small">通常指令の観測取得→実行は、SmolVLA約{sm['mean_of_episode_mean_action_source_age_ms']:.0f}ms、π0.5約{pi['mean_of_episode_mean_action_source_age_ms']:.0f}ms。π0.5も待機44.5%を含み、20Hzの制御維持と新指令の連続供給は別の指標である。</p></article><article><h3>比較の範囲</h3><p class="small">同期は両者5/5、成功数の差は非同期で確認された。方式間は撮像の更新位相も異なるため、差を推論遅延だけの効果と断定しない。各5配置であり、一般的な性能順位とは扱わない。</p></article></div>
    <div class="callout"><b>計画時の速度予想と実測の違い</b><p>この課題・実装では、π0.5は非同期の成功数が多く、方策生成・出力処理も短かった。モデル規模だけでは説明できないため、同じ保存観測を用いた追加計測で処理区間を分けた。</p></div>'''
    sp,pp=profile['smolvla'],profile['pi05'];stages=['vision_tower','vlm_prefix_prefill','flow_denoise_step'];stage_names=['画像エンコーダ','観測理解：VLM prefix','行動生成：flow全体']
    stage_rows=''.join(f'<tr class="{"highlight" if key=="flow_denoise_step" else ""}"><th>{label}</th><td>{sp["instrumented_cuda_stream_elapsed_ms_mean"][key]:.1f}ms</td><td>{pp["instrumented_cuda_stream_elapsed_ms_mean"][key]:.1f}ms</td></tr>' for key,label in zip(stages,stage_names))
    body4=f'''<div class="eyebrow">ISOLATED INFERENCE / SAME SAVED OBSERVATIONS</div><h1>推論時間の内訳と速度差</h1>
    <p>同じ5観測を各5回使用し、撮像・ROS・制御を終了して1モデルずつ測定した。両者とも50行動・flow反復10回を維持し、各計測前に10回の準備運転を行った。</p>
    <span class="scope-label" style="margin-top:3mm">A　区間計測なし：入力前処理を含む呼出し全体</span>
    <div class="profile-metrics"><div>SmolVLA<strong>{sp['pooled_uninstrumented_50_mean_wall_ms']:.1f}<small> ms</small></strong></div><div>π0.5<strong>{pp['pooled_uninstrumented_50_mean_wall_ms']:.1f}<small> ms</small></strong></div></div>
    <p class="caption">区間計測の前後で各25要求、計50要求の平均。π0.5は約39%短かった。前→後の対照平均はSmolVLA 585.1→622.1ms、π0.5 360.3→375.5msで、時間変動も記録した。</p>
    <span class="scope-label" style="margin-top:3mm">B　区間計測あり：CUDA stream上の経過時間</span>
    <figure class="profile-plot"><img src="{embedded_png(output/'inference-stages.png')}" alt="画像、VLM prefix、flowの経過時間。行動生成の区間が大きく異なる"></figure>
    <table style="margin-top:1mm"><thead><tr><th>処理区間 / 各25要求平均</th><th>SmolVLA</th><th>π0.5</th></tr></thead><tbody>{stage_rows}</tbody></table>
    <p class="caption">各25要求の平均で、flowは10反復の合計。CUDA stream経過時間は実行指示の間隔や待ちも含み、GPU演算時間だけではない。区間計測ありの呼出し全体は637.7 / 341.7msだった。</p>
    <h2>画像処理とflow生成の比較</h2>
    <p class="small">内部画像はSmolVLA 512×512、π0.5 224×224だが、画像エンコーダは約18〜19msだった。一方、flow生成は約498 / 231msであり、大きな差はこの区間に確認できた。行動expertは32層 / 18層で、幅やattention方式も異なる。繰り返す部分の構造は速度を考える手掛かりになるが、個別の寄与を分ける対照実験は未実施である。</p>
    <div class="callout"><b>精度と測定範囲を確認して解釈</b><p>代表的な行動expertのq_proj出力は両者BF16だった。SmolVLAを一律FP32とする説明は採用しない。区間計測の有無で両者の出力は完全一致した。前ページの508 / 332msは入力前処理を除く。計測範囲に加え並行負荷・準備運転も異なるため、この表を前ページの内訳としては扱わない。</p></div>'''
    rtc_names={'off':'接続補正なし','rtc-stock':'LeRobot公開RTC','rtc-vjp':'入力微分を追加したRTC'}
    rtc_rows=''.join(f'<tr><th>{NAMES[r["model"]]} / {rtc_names[r["variant"]]}</th><td class="score">{r["successes"]}/5</td><td>{r["worker_ms_episode_mean"]:.0f}ms</td><td>{100*r["pooled_hold_fraction"]:.2f}%</td></tr>' for r in continuity['rtc'])
    body5=f'''<div class="eyebrow">CONTINUITY / RTC AND ADDITIONAL METHODS</div><h1>行動列の接続を改善する検証</h1>
    <p>非同期生成では、新しい行動列が実行中の列と異なる動作を選ぶ場合がある。RTC（Real-Time Chunking）は、既存の予定指令と整合するよう生成中の行動変数を補正し、列の切り替えを滑らかにするための手法である。</p>
    <div class="flow"><div>既存の実行履歴・予定指令</div><span>→</span><div>遅れに合わせて生成中の列を補正</div><span>→</span><div>経過した指令を除いて切替</div></div>
    <p class="small" style="margin-top:3mm">接続手法は別条件で比較した。生成50行動を保持し、残り30行動で次の推論を開始する（H50/R30）。同じモデル内で初期センサと最初の出力を照合し、補正なし・LeRobot公開RTC・生成中の行動変数への微分を追加した版を各5配置で評価した。重みは固定し、学習は行わない。</p>
    <table><thead><tr><th>同じ基盤モデル内の条件</th><th>成功</th><th>生成＋ROS</th><th>待機割合</th></tr></thead><tbody>{rtc_rows}</tbody></table>
    <p class="caption">GPU使用を調整した再評価系列、計30試行。時間は初回を除く各試行平均の平均、待機は全tick合算。主比較の10行動・先読みなしとは待ち行列の条件が異なる。</p>
    <div class="callout"><b>切替時の指令差と、タスク成功を分けて評価</b><p>SmolVLAの公開RTCでは、同じ時点の旧予定と新採用の位置指令差が平均0.951→0.00869へ減少した。ただし成功は1/5のままだった。入力微分追加版は計算時間が増え、SmolVLAでは待機も22.35%へ増加した。π0.5は補正なし5/5に対し、RTCは両版4/5であり、成功数の改善は確認できなかった。</p></div>
    <p class="tiny">指令差はOSCの正規化位置指令のL2差（57 / 58切替）。メートル単位の位置誤差や、隣接する制御周期間の動作変化ではない。</p>
    <h2>異なる専用重みを使った追加検証</h2>
    <table class="asset-table"><thead><tr><th>手法・基盤</th><th>比較した条件</th><th>公式成功</th></tr></thead><tbody><tr><th>VLASH / 専用π0.5</th><td>画像と同時点 / 現在の自己状態 / 予測した自己状態</td><td>3/5・1/5・2/5</td></tr><tr><th>A2C2 / 作者450M SmolVLA</th><td>headなし / 残差として加算 / 直接行動として復号</td><td>4/5・1/5・0/5</td></tr></tbody></table>
    <p class="caption">通常のπ0.5・約605M SmolVLAとは基盤が異なり、主比較と合算しない。VLASHは状態時刻と行動の原点を対応づけた。A2C2は公開headと学習コードの厳密な対応が未確定であり、復号解釈の検証として扱う。</p>'''
    body6=f'''<div class="eyebrow">ENGINEERING CONTRIBUTION / LESSONS & REPRODUCIBILITY</div><h1>工夫した点・得られた知見と<br>応用可能性</h1>
    <table class="trial-table"><thead><tr><th>技術課題</th><th>実装・切り分け</th><th>得られた知見</th></tr></thead><tbody>
    <tr><th>描画で周期を超過</th><td>シミュレータ制御とカメラを別プロセスに分離。取得時刻つき状態から撮像。</td><td>描画を待たず約20Hzで更新できた。撮像完了時刻を取得時刻へ代用しない。</td></tr>
    <tr><th>モデルの速度予想</th><td>総時間に加え、同一観測で視覚・VLM prefix・flowを分けて計測。</td><td>小型モデルが常に速いとは限らず、今回の差は行動生成区間に集中した。</td></tr>
    <tr><th>行動列の不連続</th><td>先読みの条件を固定した対照を置き、RTCの指令差と成否を別に評価。</td><td>指令の滑らかさ・待機短縮だけでは、把持・移載の成功は増えなかった。</td></tr>
    <tr><th>通信のまれな待ち</th><td>A2C2追加検証では最終指令のROS往復を別プロセス化。期限と指令IDを検査。</td><td>通常の通信速度と遅い応答を区別。未着時は受信済み基盤指令へ戻した。</td></tr>
    </tbody></table>
    <div class="columns" style="margin-top:4mm"><article><h2>実用への接続</h2><p class="small">容器・部品の卓上移載が応用候補となる。既存の学習済み方策と制御器を用い、周期の異なる撮像・推論・制御を分離する設計は、限られたGPUでの試作に再利用できる。</p></article><article><h2>評価の限界と次の検証</h2><p class="small">1課題・固定5配置のため、実機・未知物体・別タスクへの汎化は未検証である。撮像位相やOSの実行揺らぎも残る。今後は接触場面の切替誤差と、指令の観測時刻をそろえて検証する。</p></article></div>
    <h2>再利用した資産と検証可能性</h2>
    <table class="asset-table"><thead><tr><th>資産</th><th>使用した役割・固定版</th><th>利用条件の記録</th></tr></thead><tbody>
    <tr><th>SmolVLA</th><td>SmolVLMを視覚言語基盤とする方策。LIBERO調整済み重み 6721902…</td><td>カード：Apache-2.0</td></tr>
    <tr><th>π0.5</th><td>LIBERO調整済み公開重み 7152f13…、50行動・flow10反復</td><td>選択重み：Gemma条件</td></tr>
    <tr><th>LeRobot / LIBERO</th><td>LeRobot 0.6.2、hf-libero 0.1.4、robosuite 1.4.0、MuJoCo 3.3.7</td><td>コード・重み・形状データの条件を個別記録</td></tr>
    <tr><th>実行・通信</th><td>ROS 2 Jazzy / CycloneDDS、PyTorch 2.7.1+cu128、Ubuntu 24.04 / WSL2</td><td>依存版と取得元を固定</td></tr></tbody></table>
    <p class="small" style="margin-top:3mm">主比較20試行のROS受信内容364件を照合し、学習済み重みの復元と実行前後の不変を確認した。主比較、RTC、異なる専用重みの追加検証、推論内訳を別に保存し、条件と失敗結果を追えるようにした。</p>
    <div class="refs"><b>コードと測定記録</b><div><a href="{REPO}">{REPO}</a></div><div>主比較：artifacts/realtime-vla/submission-smol-pi05/evidence-primary.json<br>RTC等・推論内訳：同フォルダのevidence-continuity.json。実行手順・取得条件はREADMEとdocs/に収録。</div><div><b>一次資料</b>　<a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero">SmolVLA重み</a> ／ <a href="https://huggingface.co/lerobot/pi05_libero_finetuned">π0.5重み</a> ／ <a href="https://github.com/huggingface/lerobot">LeRobot・RTC</a> ／ <a href="https://github.com/Lifelong-Robot-Learning/LIBERO">LIBERO</a><br>モデル重み・tokenizer・外部形状データは配布せず、固定revisionと取得手順を記録した。</div></div>'''
    return ''.join(page(i,label,body) for i,(label,body) in enumerate([
        ('課題の概要・目的',body1),('システム構成と資源管理',body2),('同期・非同期の主比較',body3),
        ('モデル処理時間の追加計測',body4),('RTCなどの追加検証',body5),('工夫・知見・再現性',body6)],1))


def main():
    """Verify audited source identities and emit report assets without modifying prior artifacts.

    Args: CLI output. Returns: None. Raises: ValueError for failed source checks.
    Design: Context: New report supersedes the historical comparison without erasing it.
        Concern: Generated numbers drift from raw trial logs.
        Decision: Consume independently audited records and verify every listed source hash.
        Alternatives: Manual numeric transcription. Quality Attributes: Traceability.
        Trade-offs: Source measurements must remain available.
    Evidence: Internal: All source_files/sources records, frame provenance and pinned skill digest.
        External: None. Findings: Checks run at each build.
    Constraints: No experiment reruns. Reconsider When: Input schema changes. References: build_pages. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--output',type=Path,default=FOLDER);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    primary=read_json(FOLDER/'evidence-primary.json');continuity=read_json(FOLDER/'evidence-continuity.json')
    if primary['status']!='cpu_recomputed_and_crosschecked' or not continuity['original_files_unchanged']:
        raise ValueError('Independent evidence audits required')
    sources={}
    for record in primary['sources']+continuity['source_files']:
        path=record['path'];digest=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        if digest!=record['sha256']:raise ValueError('Original source changed: '+path)
        sources[path]=digest
    if any(r['n']!=5 for r in primary['groups']+continuity['rtc']):raise ValueError('Unexpected trial count')
    plot_data=plot_profile(continuity['inference_profile'],out)
    html='<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>説明資料_PAI最終課題_44055</title><style>'+STYLE+'</style></head><body>'+build_pages(primary,continuity,out)+'</body></html>'
    (out/'report.html').write_text(html)
    manifest={'scope':'Evaluator-facing SmolVLA/pi05 report; main20, separate RTC30 and separately scoped additional methods; no new runs','pages':6,'sources':sources,'figure_data':plot_data,'skill':{'name':'japanese-tech-writing','revision':'8f2d57610a73efc97d743c9b0b0ecb1002e09fa4','sha256':'ad350e0e6f5630fd94d8ab98a6aee194053b0096fdc944c23e271d6dfa09ee68','url':'https://gist.github.com/k16shikano/fd287c3133457c4fd8f5601d34aa817d'},'report_html_sha256':hashlib.sha256(html.encode()).hexdigest(),'evidence_inputs':{name:hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() for name in ('evidence-primary.json','evidence-continuity.json')},'primary_group_counts':[{'model':r['model'],'mode':r['mode'],'n':r['n'],'successes':r['successes']} for r in primary['groups']]}
    (out/'report-provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'html':str(out/'report.html'),'pages':6,'verified_source_files':len(sources)},ensure_ascii=False))


if __name__=='__main__':
    main()
