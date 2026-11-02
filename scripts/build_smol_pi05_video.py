"""Compose a task-only SmolVLA/pi05 submission video from immutable recorded runs.

Args: --output: fresh directory; --font: installed Japanese font.
Returns: MP4, poster, source/frame mapping manifest, decode validation and upload text.
Raises: ValueError for source/protocol/frame/label/time mismatches or existing outputs.
Design: Context: The final report compares two frozen VLAs and asynchronous chunk connection.
    Concern: Native rollout files omit lockstep waits and have tick-rate rather than wall-rate frames.
    Decision: Label paused references continuously and replay realtime sources on completed-tick
        wall timestamps at one-times speed; show full selected trajectories without slides.
    Alternatives: Silent raw concatenation or interpretive title/results cards.
    Quality Attributes: Timing integrity, legibility, bounded memory. Trade-offs: Chosen cases
        illustrate results rather than estimate success rates; full counts stay in the report.
Evidence: Internal: result.json and ticks.jsonl of formal and clean-RTC episodes.
    External: imageio-ffmpeg and Pillow; existing make_demo.py timing/label patterns.
    Findings: Different success outcomes and RTC conditions are retained transparently.
Constraints: No GPU/model/simulator execution, source mutation, audio or YouTube upload.
Reconsider When: Source recorder layout, task comparison or video requirements change.
References: experiments/realtime_vla/make_demo.py; submission-smol-pi05/evidence-primary.json.
ADR: None.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import re

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

OUT_ROOT = ROOT / 'artifacts/realtime-vla/submission-smol-pi05/video'
SIZE = (1080, 720)
FPS = 20
PAPER = '#f5f4ee'
INK = '#203630'
MUTED = '#58675d'
GREEN = '#146b5c'
ORANGE = '#9b4d2b'
NAMES = {'smolvla': 'SmolVLA', 'pi05': 'π0.5'}
SELECTION = (
    ('formal-smolvla-e10/lockstep-init3', 'main', 'same-state lockstep reference for first realtime disagreement'),
    ('formal-pi05-e10/lockstep-init3', 'main', 'same-state lockstep reference for first realtime disagreement'),
    ('formal-smolvla-e10/realtime-init3', 'main', 'earliest state with different realtime model outcomes; failure retained'),
    ('formal-pi05-e10/realtime-init3', 'main', 'same initial state as preceding Smol realtime failure'),
    ('formal-smolvla-e10/realtime-init1', 'main', 'earliest initial state where both realtime policies succeed'),
    ('formal-pi05-e10/realtime-init1', 'main', 'same initial state as preceding Smol realtime success'),
    ('continuity-clean/smolvla-off/realtime-init1', 'rtc-off', 'first-state same-backbone RTC control; outcome retained'),
    ('continuity-clean/smolvla-rtc-stock/realtime-init1', 'rtc-stock', 'same state/model as preceding control; unsuccessful RTC example retained'),
    ('continuity-clean/pi05-off/realtime-init1', 'rtc-off', 'first-state same-backbone RTC control; successful control retained'),
    ('continuity-clean/pi05-rtc-stock/realtime-init1', 'rtc-stock', 'same state/model as preceding control; RTC failure retained'),
)


_FONT_CACHE = {}

def draw_text(draw, text, position, font_path, size, max_width, fill=INK):
    """Fit and draw one Japanese line using the existing measured-font algorithm.

    Args: draw/text/position: Pillow text contract; font_path/size/max_width/fill: rendering options.
    Returns: None. Raises: ValueError if the line cannot fit; font IO errors.
    Design: Context: Standalone submission cannot depend on the historical Go2 demo generator.
        Concern: Importing that unrelated script makes fresh-clone video composition fail.
        Decision: Reuse its small font-fit algorithm here, retaining exact rendering behavior.
        Alternatives: Redistribute the entire Go2 video generator. Quality Attributes: Portability.
        Trade-offs: A small utility copy is kept local to this task.
    Evidence: Internal: scripts/build_demo_video.py draw_text. External: Pillow textbbox/ImageFont.
        Findings: Same font sizes, cache keys and shrink-to-fit behavior as the measured utility.
    Constraints: No silent truncation or font redistribution. Reconsider When: Caption layout changes.
    References: annotate. ADR: None.
    """
    for point_size in range(max(8, int(size)), 7, -1):
        key = (str(font_path), point_size)
        if key not in _FONT_CACHE:
            _FONT_CACHE[key] = ImageFont.truetype(key[0], point_size)
        font = _FONT_CACHE[key]
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= max_width:
            draw.text(position, text, font=font, fill=fill, anchor='lt')
            return
    raise ValueError('Text cannot fit the frame: ' + text)


def sha256(path):
    """Identify exact input/output bytes without mutating media.

    Args: path: existing regular file. Returns: Hex digest. Raises: IO errors.
    Design: Context: Paths alone cannot prove provenance. Concern: Sources could be overwritten.
        Decision: Stream SHA256 before and after composition. Alternatives: Names only.
        Quality Attributes: Traceability. Trade-offs: Sequential extra read.
    Evidence: Internal: Source manifest. External: SHA256. Findings: None.
    Constraints: Read only. Reconsider When: Content-addressed storage exists. References: main. ADR: None.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def captions(row, family):
    """Produce brief model, timing and task labels without experiment explanations.

    Args: row: completed result; family: main/rtc-off/rtc-stock.
    Returns: Display label mapping. Raises: KeyError for unsupported identities.
    Design: Context: Task motion should dominate the video. Concern: Slides or opaque timing labels.
        Decision: Fixed short annotations and explicit time basis. Alternatives: Dense result dashboards.
        Quality Attributes: Clarity. Trade-offs: Protocol detail remains in PDF/manifest.
    Evidence: Internal: Native result fields. External: None. Findings: No fabricated outcome.
    Constraints: No personal identifiers. Reconsider When: Models or task change. References: annotate. ADR: None.
    """
    realtime = row['mode'] == 'realtime'
    kind = '主比較' if family == 'main' else '接続検証：RTCなし' if family == 'rtc-off' else '接続検証：LeRobot公開RTC'
    return {
        'heading': f"{NAMES[row['model']]}  |  {'非同期' if realtime else '同期'}  |  配置 {row['init_state_index']}",
        'schedule': kind + ('  ／  非同期：実時間1倍速' if realtime else '  ／  同期：推論待ち時間を省略'),
        'outcome': '試行結果：' + ('成功' if row['success'] else '未達'),
        'task': 'タスク：黒いボウルをつかみ、皿に置く',
        'camera_left': '外部カメラ',
        'camera_right': '手首カメラ',
    }


def load_clip(relative, family, reason):
    """Check completed source results, native video geometry and step-to-time records.

    Args: relative: episode directory under realtime-vla; family/reason: selection provenance.
    Returns: Validated clip descriptor. Raises: ValueError for missing/invalid/inconsistent evidence.
    Design: Context: One recorder frame corresponds to one physics tick.
        Concern: Wrong video or partial runs can misrepresent completion and timing.
        Decision: Check result identity, settings, ticks and media metadata before encoding.
        Alternatives: Trusting filenames. Quality Attributes: Integrity. Trade-offs: Decode checked later too.
    Evidence: Internal: result.json/ticks.jsonl. External: imageio-ffmpeg metadata.
        Findings: Every selection is a complete measured episode.
    Constraints: Formal states1..5 only. Reconsider When: Source recorder schema changes.
    References: stream_clip. ADR: None.
    """
    directory = ROOT / 'artifacts/realtime-vla' / relative
    row = json.loads((directory / 'result.json').read_text())
    if row['status'] != 'completed' or row.get('error') or row['init_state_index'] not in range(1, 6):
        raise ValueError('Completed formal source required: ' + relative)
    if row['model'] not in NAMES or row['control_hz'] != FPS or not row['ros_transport']:
        raise ValueError('Source model/control/ROS mismatch: ' + relative)
    if family == 'main' and (row['execute_steps'] != 10 or row['prefetch_steps'] != 0):
        raise ValueError('Main E10/P0 mismatch')
    if family != 'main' and (row['execute_steps'] != 50 or row['prefetch_steps'] != 30):
        raise ValueError('RTC H50/R30 mismatch')
    ticks = [json.loads(line) for line in (directory / 'ticks.jsonl').read_text().splitlines() if line]
    if len(ticks) != row['ticks'] or not ticks:
        raise ValueError('Source ticks mismatch')
    times = [float(t['wall_time_since_start']) + float(t['step_wall_s']) for t in ticks]
    if any(b < a for a, b in zip(times, times[1:])) or times[-1] > row['wall_seconds'] + .01:
        raise ValueError('Source wall timestamps invalid')
    video = directory / 'rollout.mp4'
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    try:
        metadata = next(reader)
    finally:
        reader.close()
    if tuple(metadata['size']) != (512, 256) or abs(metadata['fps'] - FPS) > .01:
        raise ValueError('Source video geometry/FPS mismatch')
    if abs(metadata['duration'] - len(ticks) / FPS) > .08:
        raise ValueError('Source duration/tick count mismatch')
    duration = row['wall_seconds'] if row['mode'] == 'realtime' else len(ticks) / FPS
    return {'directory': directory, 'video': video, 'row': row, 'family': family, 'reason': reason,
            'ticks': ticks, 'wall_times': times, 'duration': duration, 'metadata': metadata,
            'labels': captions(row, family), 'original_sha256': {name: sha256(directory / name)
                for name in ('rollout.mp4', 'result.json', 'ticks.jsonl')}}


def annotate(raw, clip, local_end, font):
    """Place full recorded camera views on a constant canvas with short contextual labels.

    Args: raw: decoded RGB frame; clip: validated descriptor; local_end: time at end of frame; font.
    Returns: Annotated RGB Pillow image. Raises: Image/text fitting errors.
    Design: Context: The movie must show robot task execution throughout.
        Concern: Cropping views or interpretive slides hide the task motion.
        Decision: Preserve both full views, enlarge2x, and put labels in canvas margins.
        Alternatives: Architecture/results slides. Quality Attributes: Legibility.
        Trade-offs: No audio or detailed quantitative interpretation.
    Evidence: Internal: Native completed outcome and timestamps. External: Existing draw_text utility.
        Findings: Outcome label explicitly names the trial result, not current-frame success.
    Constraints: Do not modify or synthesize robot motion. Reconsider When: Camera layout changes.
    References: stream_clip. ADR: None.
    """
    labels = clip['labels']; row = clip['row']
    image = Image.new('RGB', SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw_text(draw, labels['heading'], (24, 17), font, 29, 805, INK)
    draw_text(draw, labels['outcome'], (860, 20), font, 25, 196, GREEN if row['success'] else ORANGE)
    draw_text(draw, labels['schedule'], (24, 62), font, 23, 1032, GREEN if row['mode'] == 'realtime' else ORANGE)
    recorded = Image.frombytes('RGB', (512, 256), raw).resize((1024, 512), Image.Resampling.BILINEAR)
    image.paste(recorded, (28, 116))
    for x, name in ((41, labels['camera_left']), (553, labels['camera_right'])):
        draw.rounded_rectangle((x, 128, x + 151, 161), radius=4, fill=INK)
        draw_text(draw, name, (x + 10, 136), font, 18, 132, 'white')
    if row['mode'] == 'realtime':
        clock = f"実経過 {min(local_end, row['wall_seconds']):.2f} / {row['wall_seconds']:.2f} 秒"
    else:
        clock = f"シミュレーション {min(local_end, row['sim_seconds']):.2f} / {row['sim_seconds']:.2f} 秒  ／  元の実行 {row['wall_seconds']:.2f} 秒"
    draw_text(draw, clock, (24, 644), font, 24, 1032, INK)
    draw_text(draw, labels['task'], (24, 687), font, 18, 1032, MUTED)
    return image


def stream_clip(writer, clip, index, offset, font, output):
    """Map each output frame to a completed source step and decode every original source frame.

    Args: writer: primed RGB encoder; clip/index/offset: validated source and timeline; font/output.
    Returns: Per-clip exact mapping/provenance record. Raises: Count/timestamp/endpoint errors.
    Design: Context: Realtime wall gaps are absent from tick-rate source videos.
        Concern: Raw concatenation would silently compress elapsed time.
        Decision: At each output frame's end boundary, select latest completed source tick;
            repeat frames during gaps and retain a complete mapping, including final source tick.
        Alternatives: Assume native20FPS equals actual realtime. Quality Attributes: Timing integrity.
        Trade-offs: Sampling quantizes state display by at most one50ms video frame.
    Evidence: Internal: wall_time_since_start+step_wall_s; source tick count.
        External: Sequential imageio-ffmpeg decoding. Findings: No interpolated robot states.
    Constraints: Last recorded step must appear; no source trajectory trims.
    Reconsider When: Capture exposure timestamps become available. References: main. ADR: None.
    """
    reader = imageio_ffmpeg.read_frames(str(clip['video']), pix_fmt='rgb24'); next(reader)
    raw = next(reader); source_index = 0; decoded = 1
    total = math.ceil(clip['duration'] * FPS); realtime = clip['row']['mode'] == 'realtime'
    mapping = []
    try:
        for frame_index in range(total):
            end = (frame_index + 1) / FPS
            target = (max(0, min(len(clip['ticks']) - 1, bisect.bisect_right(clip['wall_times'], end) - 1))
                      if realtime else min(frame_index, len(clip['ticks']) - 1))
            while source_index < target:
                raw = next(reader); source_index += 1; decoded += 1
            image = annotate(raw, clip, end, font)
            writer.send(image.tobytes())
            if frame_index in (0, total // 2, total - 1):
                position = 'start' if frame_index == 0 else 'end' if frame_index == total - 1 else 'middle'
                image.save(output / f'clip-{index:02d}-{position}.png')
            mapping.append({'output_frame': frame_index, 'output_local_end_s': end,
                            'source_frame': source_index, 'source_step_completion_wall_s': clip['wall_times'][source_index]})
        for _ in reader:
            decoded += 1
    finally:
        reader.close()
    if decoded != len(clip['ticks']) or mapping[-1]['source_frame'] != decoded - 1:
        raise ValueError('Source count or terminal frame mismatch')
    row = clip['row']
    return {'source': clip['video'].relative_to(ROOT).as_posix(), 'source_hashes': clip['original_sha256'],
            'selection_reason': clip['reason'], 'model': row['model'], 'mode': row['mode'],
            'family': clip['family'], 'init_state_index': row['init_state_index'], 'native_success': row['success'],
            'execute_steps': row['execute_steps'], 'prefetch_steps': row['prefetch_steps'],
            'source_frame_range': [0, decoded - 1], 'source_frames_decoded': decoded,
            'source_fps': clip['metadata']['fps'], 'actual_run_wall_seconds': row['wall_seconds'],
            'source_sim_seconds': row['sim_seconds'], 'output_frames': total, 'output_duration_seconds': total / FPS,
            'start_seconds': offset, 'end_seconds': offset + total / FPS, 'labels': clip['labels'],
            'time_basis': 'wall-clock1x, end-of-frame completed-step sampling' if realtime else
                          'simulation time; inference/render waiting omitted and continuously labelled',
            'time_padding_seconds': total / FPS - clip['duration'], 'source_frame_mapping': mapping}


def verify_movie(path, expected_frames):
    """Decode the completed MP4 fully and verify geometry, cadence, duration and frame count.

    Args: path: encoded MP4; expected_frames: computed exact total.
    Returns: Verification record. Raises: ValueError for mismatches.
    Design: Context: Playable media may still be incomplete. Concern: Encoder truncation.
        Decision: Full sequential decode and strict metadata checks. Alternatives: Metadata-only probing.
        Quality Attributes: Completeness. Trade-offs: One CPU decode pass.
    Evidence: Internal: Manifest frame totals. External: imageio-ffmpeg decoder.
        Findings: Structural acceptance is separate from visual review.
    Constraints: No GPU decoding. Reconsider When: Codec changes. References: main. ADR: None.
    """
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24'); metadata = next(reader)
    count = 0
    for raw in reader:
        if len(raw) != SIZE[0] * SIZE[1] * 3:
            raise ValueError('Decoded frame size mismatch')
        count += 1
    if count != expected_frames or tuple(metadata['size']) != SIZE or abs(metadata['fps'] - FPS) > .01:
        raise ValueError('Completed movie count/geometry/FPS mismatch')
    if abs(metadata['duration'] - count / FPS) > .08 or not 60 <= count / FPS <= 180:
        raise ValueError('Completed movie duration mismatch/outside submission bounds')
    return {'status': 'passed', 'all_frames_decoded': count, 'duration_seconds': count / FPS,
            'size': list(SIZE), 'fps': FPS, 'codec': metadata.get('codec'), 'audio': 'none',
            'no_intro_or_explanatory_slides': True, 'all_terminal_source_frames_shown': True}


def upload_text(manifest):
    """Describe task/video timing and documented case selection for a manual upload.

    Args: manifest: verified final timeline. Returns: Japanese title/description text.
    Raises: None.
    Design: Context: User performs upload. Concern: Timing omissions or sample selection hidden.
        Decision: Explicit concise time-basis and selected-example explanation with chapter times.
        Alternatives: Calling this video a success-rate evaluation. Quality Attributes: Honesty.
        Trade-offs: Full counts remain in PDF and repository.
    Evidence: Internal: Final clips/timing mapping. External: Assignment title convention.
        Findings: No upload action performed.
    Constraints: No personal name/contact data. Reconsider When: User chooses a different cut.
    References: main. ADR: None.
    """
    lines = ['タイトル：PAI最終課題_44055', '', 'LIBEROのPandaアームが、黒いボウルをつかんで指定された皿へ置くタスクを実行します。',
             'SmolVLAとπ0.5の公開済み学習モデルを用い、ROS 2通信、撮像、行動生成、シミュレータ内の制御を接続しました。今回の制作で追加学習は行っていません。', '',
             '非同期の映像は実時間1倍速です。同期（lockstep）の映像は推論・描画でシミュレーションが停止する待ち時間を省略し、その旨を映像内に常時表示しています。2倍速編集は行っていません。',
             '映像は比較を説明するために選んだ10試行で、成功率を推定する標本ではありません。主比較はモデル2種類×同期/非同期×5配置の20試行で、同期は両者5/5、非同期はSmolVLA3/5、π0.5が5/5でした。',
             'RTCの映像は行動列長と先読みが異なる別条件（H50/R30）の同一配置例です。RTC全30試行の成功数は、SmolVLAが補正なし/公開版/入力微分追加版で各1/5、π0.5が5/5・4/5・4/5でした。', '',
             'コード：https://github.com/kenb91311/pai-final-44055-libero-vla', '', 'チャプター：']
    for entry in manifest['clips']:
        second = int(entry['start_seconds']); timing = f'{second // 60:02d}:{second % 60:02d}'
        lines.append(f"{timing} {entry['labels']['heading']} / {entry['labels']['schedule'].split('  ／')[0]} / {entry['labels']['outcome']}")
    return '\n'.join(lines) + '\n'


def main():
    """Build only fresh task-video artifacts and preserve all source evidence byte-for-byte.

    Args: CLI output/font. Returns: None. Raises: Validation/encoding/IO errors.
    Design: Context: A new submission accompanies the SmolVLA/pi05 report.
        Concern: Existing uploaded video, report candidates or native runs could be overwritten.
        Decision: Restrict output subtree and refuse existing final files; encode CPU2threads.
        Alternatives: Replace historical YouTube submission. Quality Attributes: Reversibility/integrity.
        Trade-offs: User uploads the new movie manually.
    Evidence: Internal: Full source/output decode and before/after SHA checks.
        External: libx264 CPU encoding. Findings: No learning or simulation required.
    Constraints: No network/upload/GPU. Reconsider When: New experiment footage is requested.
    References: video-manifest.json. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0]); parser.add_argument('--output', type=Path, default=OUT_ROOT)
    parser.add_argument('--font', type=Path, default=Path('/mnt/c/Windows/Fonts/meiryo.ttc')); args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUT_ROOT.resolve()) or not args.font.is_file():
        raise ValueError('Output must stay in owned video subtree and Japanese font must exist')
    output.mkdir(parents=True, exist_ok=True)
    movie = output / 'PAI最終課題_44055.mp4'; temporary = output / 'encoding.mp4'
    if movie.exists() or temporary.exists() or (output / 'video-manifest.json').exists():
        raise ValueError('Video outputs must be fresh')
    clips = [load_clip(*entry) for entry in SELECTION]
    expected_frames = sum(math.ceil(clip['duration'] * FPS) for clip in clips)
    if not 90 <= expected_frames / FPS <= 120:
        raise ValueError('Selected video does not fit declared90–120second target')
    for clip in clips:
        text = '\n'.join(clip['labels'].values())
        if re.search(r'nkutm|/home/|Users[/\\]|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text):
            raise ValueError('Label privacy check failed')
    manifest = {'schema_version': 1, 'title': 'PAI最終課題_44055', 'task': 'LIBERO Spatial task0 black bowl to plate',
                'models': list(NAMES), 'size': list(SIZE), 'fps': FPS, 'clips': [], 'slides': 0,
                'selection_policy': 'First main-model realtime disagreement is state3; same-state lockstep references; first shared realtime success state1; first-state clean RTC off/stock examples for each backbone. Outcomes retained. This is an explanatory case selection, not a statistical sample.',
                'playback_policy': 'Realtime wall-clock1x, recorded-frame holds during gaps, end-of-frame sampling (50ms quantization); lockstep simulation-time reference with inference/render waits omitted and continuously labelled. No2x speedup.',
                'encoding': {'codec': 'libx264', 'cpu_threads': 2, 'crf': 20, 'preset': 'fast', 'audio': False},
                'source_script': {'path': 'scripts/build_smol_pi05_video.py', 'sha256': sha256(Path(__file__))}}
    writer = imageio_ffmpeg.write_frames(str(temporary), SIZE, fps=FPS, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-crf', '20', '-preset', 'fast', '-threads', '2', '-movflags', '+faststart']); writer.send(None)
    offset = 0.0
    try:
        for index, clip in enumerate(clips, 1):
            entry = stream_clip(writer, clip, index, offset, args.font, output); offset = entry['end_seconds']; manifest['clips'].append(entry)
            print(json.dumps({'clip_complete': index, 'model': entry['model'], 'duration_seconds': entry['output_duration_seconds'], 'total_seconds': offset}), flush=True)
    finally:
        writer.close()
    verification = verify_movie(temporary, expected_frames)
    for clip in clips:
        for name, digest in clip['original_sha256'].items():
            if sha256(clip['directory'] / name) != digest:
                raise ValueError('Source changed during composition')
    temporary.replace(movie)
    manifest.update({'movie': movie.name, 'movie_sha256': sha256(movie), 'bytes': movie.stat().st_size,
                     'output_frames': expected_frames, 'duration_seconds': expected_frames / FPS,
                     'all_source_files_unchanged': True, 'label_privacy_checks_passed': True,
                     'verification': verification, 'visual_review': 'Representative frames emitted for root review.'})
    (output / 'video-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'video-validation.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    (output / 'YouTube概要欄_PAI最終課題_44055.txt').write_text(upload_text(manifest))
    (output / 'YouTubeタイトル.txt').write_text('PAI最終課題_44055\n')
    (output / 'poster.png').write_bytes((output / 'clip-01-start.png').read_bytes())
    print(json.dumps({'video': movie.relative_to(ROOT).as_posix(), 'duration_seconds': expected_frames / FPS,
                      'clips': len(clips), 'frames': expected_frames, 'sha256': manifest['movie_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
