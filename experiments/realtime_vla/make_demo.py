"""Compose1–3minutes of recorded VLA task execution with brief context labels.

Args: Complete report JSON, output directory and an installed Japanese font.
Returns: Task-only MP4, poster and source/timeline provenance JSON.
Raises: ValueError for incomplete evidence, bad frame counts or unsupported timing.
Design:
    Context: Source rollout videos are recorded at simulation tick rate.
    Concern: Paused lockstep playback can appear faster than actual operation.
    Decision: Reconstruct real-time footage from recorded wall timestamps; visibly label
        lockstep as a paused reference with inference waits omitted.
    Alternatives: Concatenating raw videos would conceal timing differences.
    Quality Attributes: Honesty, legibility, bounded memory. Trade-offs: Annotated comparison
        examples illustrate outcomes; complete counts remain in the aggregate report.
Evidence:
    Internal: summarize.py validates paired runs; ticks.jsonl carries wall time and step cost.
    External: imageio-ffmpeg streaming RGB interfaces and Pillow installed font rendering.
    Findings: Native success and queue starvation are separate labels.
    Metrics: Source hashes, decoded frame count, output duration and timing mapping.
    Confidence: Full decode plus manual poster/frame review required.
Constraints: No source trajectory edits, invented successes, unlabelled speed changes or font copying.
Reconsider When: Recorder layout, action cadence or media schema changes.
References: summarize.py; docs/architecture/realtime-vla.md.
ADR: None.
"""
from __future__ import annotations
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
from video_text import default_font,draw_text

SIZE=(1080,720);FPS=20;PAPER='#f6f5ef';INK='#18343a';MUTED='#56666d';TEAL='#087f8c';ORANGE='#d46928'
NAMES={'smolvla':'SmolVLA','pi0fast':'π0-FAST'}


def digest(path):
    """Identify source evidence with a streamed SHA256 digest.

    Args: path: source file. Returns: Hex digest. Raises: IO errors.
    Design: Context: Formal directories must remain immutable.
        Concern: Names do not prove media identity. Decision: Hash bytes.
        Alternatives: Only storing relative paths. Quality Attributes: Traceability.
        Trade-offs: One extra sequential read.
    Evidence: Internal: Video manifest. External: SHA256.
        Findings: Sources are not modified. Metrics: Digest. Confidence: Deterministic.
    Constraints: Read only. Reconsider When: Content-addressed storage is available.
    References: main. ADR: None.
    """
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def label(image,text,xy,font,size=26,color=INK,width=960):
    """Fit one Japanese caption using existing measured font utilities.

    Args: image/text/xy/font/size/color/width: Pillow text contract.
    Returns: None. Raises: ValueError when a caption cannot fit.
    Design: Context: Video captions must be readable on the final canvas.
        Concern: Silent clipping hides units or qualifications.
        Decision: Reuse measured text fitting. Alternatives: Fixed glyph counts.
        Quality Attributes: Legibility. Trade-offs: Long labels shrink.
    Evidence: Internal: video_text.py draw_text.
        External: Pillow metrics. Findings: Font remains installed, not redistributed.
        Metrics: Pixel width. Confidence: Visual review.
    Constraints: No truncation. Reconsider When: Canvas changes.
    References: card. ADR: None.
    """
    draw_text(ImageDraw.Draw(image),text,xy,font,size,width,color)


def select_episodes(report):
    """Pick a paired state plus existing outcome examples with transparent deterministic rules.

    Args: report: complete formal evidence. Returns: Selected episode records/reasons.
    Raises: ValueError for missing required real-time/baseline clips.
    Design: Context: A short movie cannot show all twenty episodes in full.
        Concern: Selecting only successful motion would conceal failures.
        Decision: Earliest cross-model disagreement, otherwise earliest paired state;
            include existing outcomes, same-state baselines, first pi baseline success
            and declared first-state Smol exploratory conditions.
        Alternatives: Picking aesthetically pleasing trajectories.
        Quality Attributes: Integrity. Trade-offs: Examples are not a statistical sample.
    Evidence: Internal: Complete counts remain in report JSON.
        External: None. Findings: No outcome is fabricated if all trials share one result.
        Metrics: Selection reason in manifest. Confidence: Deterministic selection.
    Constraints: Source initial states and trajectories remain unchanged.
    Reconsider When: All episodes fit the submission duration.
    References: load_clip. ADR: None.
    """
    rows=report['episodes'];rt=[r for r in rows if r['mode']=='realtime'];states=sorted({r['init_state_index'] for r in rt})
    differing=[]
    for state in states:
        pair=[r for r in rt if r['init_state_index']==state]
        if len(pair)==2 and len({r['success'] for r in pair})>1:differing.append(state)
    state=differing[0] if differing else states[0];selected=[]
    for model in ('smolvla','pi0fast'):
        matches=[r for r in rt if r['init_state_index']==state and r['model']==model]
        if len(matches)!=1:raise ValueError('missing paired real-time clip')
        selected.append((matches[0],'earliest paired disagreement' if differing else'earliest paired state'))
    for outcome in (True,False):
        if any(bool(r['success'])==outcome for r,_ in selected):continue
        candidates=sorted([r for r in rt if bool(r['success'])==outcome],key=lambda r:(r['init_state_index'],r['model']))
        if not candidates:candidates=sorted([r for r in rows if bool(r['success'])==outcome],key=lambda r:(r['mode'],r['init_state_index'],r['model']))
        if candidates:selected.append((candidates[0],'first recorded success' if outcome else'first recorded failure'))
    for model in ('smolvla','pi0fast'):
        matches=[r for r in rows if r['mode']=='lockstep' and r['init_state_index']==state and r['model']==model]
        if len(matches)!=1:raise ValueError('missing paired lockstep clip')
        if not any(r['source_directory']==matches[0]['source_directory'] for r,_ in selected):selected.append((matches[0],'same-state paused reference'))
    pi_success=sorted([r for r in rows if r['model']=='pi0fast' and r['mode']=='lockstep' and r['success']],key=lambda r:r['init_state_index'])
    if pi_success and not any(r['model']=='pi0fast' and r['mode']=='lockstep' and r['success'] for r,_ in selected):selected.append((pi_success[0],'first pi paused-reference success to show pretrained competence'))
    ablation=report.get('scheduler_ablation',{})
    variants=sorted([r for r in ablation.get('episodes',[]) if r['model']=='smolvla'],key=lambda r:r['init_state_index'])
    if variants:selected.append((variants[0],'first-state exploratory Smol E10/P5; outcome retained'))
    native=report.get('native_horizon_smol',{})
    variants=sorted(native.get('episodes',[]),key=lambda r:r['init_state_index'])
    if variants:selected.append((variants[0],'first-state exploratory Smol E50/P0; outcome retained'))
    return selected


def load_clip(row,reason):
    """Validate immutable source media against per-step evidence before composition.

    Args: row: formal episode; reason: deterministic selection description.
    Returns: Media descriptor. Raises: ValueError for frame/time inconsistencies.
    Design: Context: Every source frame corresponds to one completed physics tick.
        Concern: Partial videos or wrong FPS can misrepresent task timing.
        Decision: Verify metadata and all tick IDs; decode count is checked during streaming.
        Alternatives: Trusting the filename. Quality Attributes: Traceability.
        Trade-offs: Final decode is required.
    Evidence: Internal: result ticks and wall timestamps.
        External: imageio-ffmpeg metadata. Findings: Real-time frame times use step completion.
        Metrics: Expected frame count/fps/duration. Confidence: Structural validation.
    Constraints: No source editing. Reconsider When: Recorder adds an initial frame.
    References: stream_clip. ADR: None.
    """
    directory=ROOT/row['source_directory'];video=directory/'rollout.mp4';ticks=[json.loads(line) for line in (directory/'ticks.jsonl').read_text().splitlines() if line]
    if len(ticks)!=row['ticks'] or not ticks:raise ValueError('missing source ticks')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    try:metadata=next(reader)
    finally:reader.close()
    if tuple(metadata['size'])!=(512,256) or abs(metadata['fps']-row['control_hz'])>.01:raise ValueError(f'unexpected source video properties: {video}')
    if abs(metadata['duration']-len(ticks)/row['control_hz'])>.08:raise ValueError('source video duration disagrees with ticks')
    wall_times=[float(t['wall_time_since_start'])+float(t['step_wall_s']) for t in ticks]
    if any(b<a for a,b in zip(wall_times,wall_times[1:])):raise ValueError('nonmonotonic source wall timestamps')
    duration=float(row['wall_seconds']) if row['mode']=='realtime' else len(ticks)/row['control_hz']
    return {'row':row,'reason':reason,'directory':directory,'video':video,'ticks':ticks,'wall_times':wall_times,'duration':duration,'source_metadata':metadata}


def annotate(raw,clip,index,playback_time,font):
    """Keep recorded task views clear with brief condition, clock and terminal labels.

    Args: raw: RGB bytes; clip/index/time: source position; font: installed typeface.
    Returns: RGB task frame. Raises: ValueError for malformed frame bytes.
    Design:
        Context: The user requires task-execution footage only.
        Concern: Explanatory slides or dense metrics would interrupt the robot motion.
        Decision: Maximize both recorded camera views; show only model/E/P/state,
            timing mode, clock and explicitly terminal outcome.
        Alternatives: Interpretation cards belong in the separate report/PDF.
        Quality Attributes: Clarity, honesty. Trade-offs: Detailed metrics are external.
    Evidence:
        Internal: Source tick timestamps and completed native outcome.
        External: None. Findings: Lockstep waits remain visibly qualified throughout.
        Metrics: Correct source frame/time/condition association.
        Confidence: Final frame visual review.
    Constraints: No architecture, results table, summary slide or fabricated motion.
    Reconsider When: The user requests an explanatory edit separately.
    References: stream_clip. ADR: None.
    """
    row=clip['row'];realtime=row['mode']=='realtime'
    exploratory=row.get('execute_steps')!=10 or row.get('prefetch_steps')!=0
    color='#7656a8' if exploratory else TEAL if realtime else ORANGE
    image=Image.new('RGB',SIZE,PAPER);draw=ImageDraw.Draw(image)
    heading=f"{NAMES[row['model']]}  E{row['execute_steps']}/P{row['prefetch_steps']}  |  初期状態 {row['init_state_index']}"
    label(image,heading,(24,12),font,29,width=820)
    outcome=f"終端：{'成功' if row['success'] else '失敗'}"
    label(image,outcome,(870,16),font,26,TEAL if row['success'] else ORANGE,width=190)
    schedule='実時間 1×：物理を20 Hzで継続' if realtime else'LOCKSTEP参照：推論待ち時間を省略した物理時間'
    if exploratory:schedule='探索条件 ／ '+schedule
    label(image,schedule,(24,55),font,22,color,width=1032)
    frame=Image.frombytes('RGB',(512,256),raw).resize((1080,540),Image.Resampling.BILINEAR);image.paste(frame,(0,91))
    for x,title in ((12,'外部カメラ'),(552,'手首カメラ')):
        draw.rounded_rectangle((x,102,x+146,133),radius=5,fill='#18343a');label(image,title,(x+9,108),font,17,'white',128)
    if realtime:clock=f"wall {playback_time:.2f} / {row['wall_seconds']:.2f} 秒"
    else:clock=f"物理時刻 {playback_time:.2f} / {row['sim_seconds']:.2f} 秒  ／  元のwall実行時間 {row['wall_seconds']:.2f} 秒"
    label(image,clock,(24,650),font,24,width=1032)
    label(image,'タスク：黒いボウルをつかみ、皿に置く',(24,689),font,18,MUTED,width=1032)
    return image


def stream_clip(writer,clip,font):
    """Replay real-time recordings on wall timestamps and paused references on simulation time.

    Args: writer: primed encoder; clip: validated source; font.
    Returns: Timeline/source manifest entry. Raises: ValueError for wrong decoded frame count.
    Design:
        Context: Raw20FPS video excludes any wall-clock gaps.
        Concern: A slow loop could appear real-time at unqualified raw playback speed.
        Decision: Use completed-step wall times for RT and explicitly label lockstep time compression.
        Alternatives: Assuming all source frames are50ms apart.
        Quality Attributes: Timing integrity, bounded memory. Trade-offs: Repeat frames during gaps.
    Evidence: Internal: ticks wall_time_since_start+step_wall_s.
        External: imageio-ffmpeg streaming decoder. Findings: Only current source frame is retained.
        Metrics: Source count, output count and mapped duration. Confidence: Full sequential decode.
    Constraints: No intermediate motions generated; first source frame held until its first timestamp.
    Reconsider When: Recorder stores exposure-start timestamps separately.
    References: load_clip. ADR: None.
    """
    reader=imageio_ffmpeg.read_frames(str(clip['video']),pix_fmt='rgb24');next(reader);raw=next(reader);source_index=0;count=1
    total=max(1,math.ceil(clip['duration']*FPS));realtime=clip['row']['mode']=='realtime'
    try:
        for frame_index in range(total):
            t=frame_index/FPS
            wanted=max(0,min(len(clip['ticks'])-1,bisect.bisect_right(clip['wall_times'],t)-1)) if realtime else min(len(clip['ticks'])-1,round(frame_index*clip['row']['control_hz']/FPS))
            while source_index<wanted:raw=next(reader);source_index+=1;count+=1
            writer.send(annotate(raw,clip,source_index,t,font).tobytes())
        for raw in reader:count+=1
    finally:reader.close()
    if count!=len(clip['ticks']):raise ValueError(f"decoded frame count{count} != ticks{len(clip['ticks'])}")
    return {'source':str(clip['video'].relative_to(ROOT)),'sha256':digest(clip['video']),'selection_reason':clip['reason'],'model':clip['row']['model'],'mode':clip['row']['mode'],'execute_steps':clip['row']['execute_steps'],'prefetch_steps':clip['row']['prefetch_steps'],'init_state_index':clip['row']['init_state_index'],'native_success':clip['row']['success'],'source_frames':count,'output_frames':total,'output_duration_s':total/FPS,'actual_run_wall_s':clip['row']['wall_seconds'],'source_sim_s':clip['row']['sim_seconds'],'timing':'wall-timestamp reconstruction at1x' if realtime else'simulated-time paused reference; inference/render waits omitted and labelled','wall_frame_mapping':'first frame held before first completed tick; last completed-step frame selected thereafter'}


def main():
    """Compose1–3minutes of recorded task execution without explanatory slides.

    Args: CLI report/output/font. Returns: Task-only MP4, poster and provenance manifest.
    Raises: Incomplete report, invalid media, wrong frame count or duration errors.
    Design:
        Context: Recorded task behavior is the subject of the movie.
        Concern: Cards, interpretation and metric dashboards must not occupy the movie.
        Decision: Stream seven deterministically selected complete task recordings with
            minimal persistent context and transparent real-time/reference clocks.
        Alternatives: Interleaving task footage with explanatory cards.
        Quality Attributes: Faithfulness, clarity. Trade-offs: Readers consult the report for analysis.
    Evidence:
        Internal:35trial aggregate, immutable raw rollouts and exact wall timestamps.
        External: ffmpeg full decode. Findings: No synthetic frames except repeated recorded
            frames required to reconstruct wall-clock gaps.
        Metrics: Source/final frame counts, duration and source hashes.
        Confidence: Manual task-frame review follows full decode.
    Constraints: No intro/results/settings/architecture slides; output under report only.
    Reconsider When: The user requests another edit format.
    References: summarize.py. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--report',type=Path,default=ROOT/'artifacts/realtime-vla/report/results.json');parser.add_argument('--output',type=Path,default=ROOT/'artifacts/realtime-vla/report');parser.add_argument('--font',type=Path,default=default_font());args=parser.parse_args()
    output=args.output.resolve();allowed=(ROOT/'artifacts/realtime-vla/report').resolve()
    if not output.is_relative_to(allowed):raise ValueError('output must remain under artifacts/realtime-vla/report')
    report_path=args.report.resolve();report=json.loads(report_path.read_text())
    if report['status']!='complete':raise ValueError('complete formal report required')
    for name in ('scheduler_ablation','native_horizon_smol'):
        if report.get(name) and report[name]['status']!='complete':raise ValueError(f'incomplete {name}')
    output.mkdir(parents=True,exist_ok=True);font=args.font
    if not font.is_file():raise FileNotFoundError(font)
    selected=[load_clip(row,reason) for row,reason in select_episodes(report)]
    omitted=[];expected_frames=sum(max(1,math.ceil(c['duration']*FPS)) for c in selected)
    if expected_frames/FPS>180 and selected[-1]['row']['execute_steps']==50:
        omitted.append({'source':selected[-1]['row']['source_directory'],'reason':'native clip omitted to keep task-only submission below180seconds; full result remains in report'})
        selected.pop();expected_frames=sum(max(1,math.ceil(c['duration']*FPS)) for c in selected)
    if not 60<=expected_frames/FPS<=180:raise ValueError(f'task-only video outside1–3min: {expected_frames/FPS}')
    ablation=report.get('scheduler_ablation',{});native=report.get('native_horizon_smol',{})
    movie=output/'demo.mp4';temporary=output/'demo-encoding.mp4'
    manifest={'report':str(report_path.relative_to(ROOT)),'report_sha256':digest(report_path),'fps':FPS,'size':SIZE,'presentation':'recorded task execution only; brief condition/clock/terminal labels; no explanatory slides','clips':[],'omitted_clips':omitted,'full_formal_outcomes':[{'model':g['model'],'mode':g['mode'],'successes':g['successes'],'n':g['n']} for g in report['groups']],'exploratory_scheduler_outcomes':[{'model':g['model'],'execute_steps':10,'prefetch_steps':5,'successes':g['successes'],'n':g['n']} for g in ablation.get('groups',[])],'exploratory_native_outcomes':[{'model':g['model'],'execute_steps':50,'prefetch_steps':0,'successes':g['successes'],'n':g['n']} for g in native.get('groups',[])],'total_evaluated_trials':len(report['episodes'])+len(ablation.get('episodes',[]))+len(native.get('episodes',[]))}
    writer=imageio_ffmpeg.write_frames(str(temporary),SIZE,fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','20','-preset','medium','-threads','2','-movflags','+faststart']);writer.send(None);offset=0.
    try:
        for clip in selected:
            entry=stream_clip(writer,clip,font);entry['start_s']=offset;offset+=entry['output_duration_s'];manifest['clips'].append(entry)
    finally:writer.close()
    decoder=imageio_ffmpeg.read_frames(str(temporary),pix_fmt='rgb24');metadata=next(decoder);decoded=sum(1 for _ in decoder)
    if decoded!=expected_frames or abs(metadata['duration']-expected_frames/FPS)>.08:raise ValueError('final encode frame count/duration mismatch')
    temporary.replace(movie)
    reader=imageio_ffmpeg.read_frames(str(selected[0]['video']),pix_fmt='rgb24');next(reader)
    try:annotate(next(reader),selected[0],0,0.,font).save(output/'poster.png')
    finally:reader.close()
    manifest.update(output_frames=decoded,duration_s=decoded/FPS,movie='demo.mp4',movie_sha256=digest(movie),size_bytes=movie.stat().st_size,validation='all selected source frames decoded; final task-only movie fully decoded; visual review pending')
    (output/'video-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'video':str(movie),'seconds':manifest['duration_s'],'clips':len(selected),'frames':decoded,'bytes':movie.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
