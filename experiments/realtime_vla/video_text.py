"""Fit task-video labels using an installed font and measured Pillow bounds.

Args: Text, drawing context, font path and allowed width.
Returns: Label pixels on the caller's image.
Raises: Font errors or ValueError when the line cannot fit.
Design:
    Context: Recorded cameras need short readable Japanese labels.
    Concern: Truncation can hide timing qualifications.
    Decision: Cache fonts and shrink only when textbbox exceeds the width.
    Alternatives: Character counts or unrelated video-generator imports.
    Quality Attributes: Legibility, isolation. Trade-offs: Long lines use smaller text.
Evidence:
    Internal: make_demo.py uses fixed positions and sizes.
    External: Pillow ImageFont and textbbox.
    Findings: Font binaries remain external.
    Metrics: Pixel width. Confidence: Rendered-frame review.
Constraints: No silent truncation.
Reconsider When: Canvas or language changes.
References: make_demo.py.
ADR: None.
"""
from pathlib import Path
from PIL import ImageFont

_FONT_CACHE={}


def default_font():
    """Choose a standard Japanese font while keeping CLI help usable.

    Args: None. Returns: Font path; first candidate if none exists.
    Raises: None; drawing reports a missing font.
    Design: Context: Linux and WSL font locations differ.
        Concern: Defaults must not block argument parsing.
        Decision: Probe system locations and support --font.
        Alternatives: Bundling fonts. Quality Attributes: Portability.
        Trade-offs: A host without fonts must install one.
    Evidence: Internal: make_demo --font. External: Noto/Windows system paths.
        Findings: Font data stays external. Metrics: Existence. Confidence: Filesystem check.
    Constraints: No personal paths. Reconsider When: Platforms change.
    References: README.md. ADR: None.
    """
    candidates=[Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                Path('/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf'),
                Path('/mnt/c/Windows/Fonts/meiryo.ttc')]
    return next((path for path in candidates if path.is_file()),candidates[0])


def draw_text(draw,text,position,font_path,size,max_width,fill='white'):
    """Draw a complete label after fitting its measured width.

    Args: draw/text/position/font_path/size/max_width/fill: Pillow properties.
    Returns: None. Raises: Font errors or ValueError if text cannot fit.
    Design: Context: Task cameras occupy a fixed canvas.
        Concern: Labels must remain complete.
        Decision: Measure each line and cache font/size pairs.
        Alternatives: Truncation. Quality Attributes: Predictability.
        Trade-offs: Long lines shrink to a minimum8px.
    Evidence: Internal: Video annotations. External: Pillow textbbox.
        Findings: Canvas determines size. Metrics: Width. Confidence: Deterministic.
    Constraints: One line. Reconsider When: Multiline labels are introduced.
    References: make_demo.py. ADR: None.
    """
    for point_size in range(max(8,int(size)),7,-1):
        key=(str(font_path),point_size)
        if key not in _FONT_CACHE:_FONT_CACHE[key]=ImageFont.truetype(key[0],point_size)
        font=_FONT_CACHE[key];bounds=draw.textbbox((0,0),text,font=font)
        if bounds[2]-bounds[0]<=max_width:
            draw.text(position,text,font=font,fill=fill,anchor='lt');return
    raise ValueError(f'Text cannot fit the frame: {text}')
