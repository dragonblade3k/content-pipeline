"""
Stage 4: video.py

Turns synced voice lines into a finished vertical video, one caption
card per line, timed to that line's narration, concatenated into a
single mp4.

Same interface pattern as the rest of the pipeline: BackgroundRenderer
is the contract, SolidBackground is the default (a flat color card,
zero motion, zero risk) and VoxelDropBackground is the animated
upgrade, a procedurally generated field of colored blocks, drawn
entirely with Pillow and basic trig, no external assets. That
"colored blocks bouncing behind the text" look is deliberately generic
rather than a copy of any specific game's textures, characters, or
branding: real Minecraft footage (or any other copyrighted stock
footage) is exactly the exposure flagged as a real risk once a
pipeline like this runs across many clips, scraping "no copyright"
reuploads off YouTube doesn't actually clear that risk, it just hides
it. Generating the motion instead of sourcing it sidesteps the
question entirely, and it's also just a better fit for a from-scratch
portfolio piece than an unattributed clip would be.

assemble_video also optionally mixes a background music track under
the narration via ffmpeg. No track ships with this repo, on purpose:
music is exactly the kind of asset that's either real money
(licensing) or a real name attached (attribution), and neither of
those is something to bundle silently into someone else's repo. See
README "Adding background music" for where to get one for free and
how to point PIPELINE_MUSIC_PATH at it.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import math
import random
import subprocess
import warnings

from PIL import Image, ImageDraw, ImageFont

from .tts import VoiceLine

WIDTH, HEIGHT = 1080, 1920
BG_COLOR = (12, 14, 20)
ACCENT_COLOR = (226, 149, 77)
TEXT_COLOR = (237, 235, 230)

_FONT_DIR = "/usr/share/fonts/truetype/dejavu"

_BLOCK_COLORS = [
    (76, 175, 80), (33, 150, 243), (255, 152, 0),
    (156, 39, 176), (244, 67, 54), (0, 188, 212),
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    DejaVu is the intended caption face. If the absolute path misses,
    Pillow's truetype() also searches the system font directories by
    file name, so this only falls through when DejaVu isn't installed
    anywhere Pillow looks, which is the normal state of a stock Windows
    or macOS machine. The fallback must keep the requested size:
    load_default() with no argument is a roughly 10px face, which on a
    1080x1920 frame yields a finished mp4 whose captions nobody can
    read, with nothing reporting that anything went wrong.
    """
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{_FONT_DIR}/{name}", size)
    except OSError:
        pass
    warnings.warn(
        f"{name} not found in {_FONT_DIR} or the system font directories, "
        "falling back to Pillow's built-in font. Captions will render, but "
        "install DejaVu (e.g. the fonts-dejavu-core package) for the "
        "intended typeface.",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10.1 has no size parameter; requirements.txt still
        # allows 10.0, so degrade to the small bitmap face there.
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class BackgroundRenderer(ABC):
    """
    Contract for whatever sits behind the caption text. `frame_rate()`
    is the signal assemble_video uses to decide how to render a line:
    1 means "static", call frame() once and hold it for the whole
    line's audio; anything higher means "animated", render that many
    frames per second for the line's duration and encode them as an
    image sequence.
    """

    @abstractmethod
    def frame(self, frame_index: int, elapsed_seconds: float) -> Image.Image:
        raise NotImplementedError

    def frame_rate(self) -> int:
        return 1


class SolidBackground(BackgroundRenderer):
    """The original, default background. Flat color, no motion, no risk."""

    def frame(self, frame_index: int, elapsed_seconds: float) -> Image.Image:
        return Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)


class VoxelDropBackground(BackgroundRenderer):
    """
    An original, procedurally animated block field: colored squares
    drop in from above, settle into a shifting skyline unique to each
    line (seeded by line index, so it's reproducible, not random every
    run), and idle with a small sine wave bounce once settled. That's
    the "something moving behind the text" energy the solid card
    doesn't have, generated from scratch rather than sourced from
    anywhere, so there's no footage, texture, or trademark to reason
    about. A dark overlay is blended on top so the caption text stays
    readable over the motion.
    """

    _COLS = 9
    _ROWS = 17
    _CELL = WIDTH // _COLS
    _FPS = 12
    _DROP_SECONDS = 1.2

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self._targets = [rng.randint(3, self._ROWS - 1) for _ in range(self._COLS)]
        self._colors = [rng.choice(_BLOCK_COLORS) for _ in range(self._COLS)]

    def frame_rate(self) -> int:
        return self._FPS

    def frame(self, frame_index: int, elapsed_seconds: float) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        drop_progress = min(1.0, elapsed_seconds / self._DROP_SECONDS)
        eased = 1 - (1 - drop_progress) ** 3
        for col in range(self._COLS):
            target_row = self._targets[col]
            row = -2 + eased * (target_row + 2)
            if drop_progress >= 1.0:
                row = target_row + math.sin(elapsed_seconds * 2.0 + col) * 0.15
            x0 = col * self._CELL
            y0 = HEIGHT - int(row * self._CELL) - self._CELL
            x1, y1 = x0 + self._CELL - 6, y0 + self._CELL - 6
            draw.rectangle([x0, y0, x1, y1], fill=self._colors[col])
        overlay = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        return Image.blend(img, overlay, 0.45)


def _draw_caption(img: Image.Image, text: str, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)

    label_font = _load_font(40, bold=True)
    draw.text((80, 140), label.upper(), font=label_font, fill=ACCENT_COLOR)
    draw.rectangle([80, 200, 280, 205], fill=ACCENT_COLOR)

    body_font = _load_font(64, bold=True)
    max_width = WIDTH - 160
    lines = _wrap_text(draw, text, body_font, max_width)
    line_height = 84
    total_height = line_height * len(lines)
    y = (HEIGHT - total_height) // 2
    for line in lines:
        draw.text((80, y), line, font=body_font, fill=TEXT_COLOR)
        y += line_height
    return img


def render_caption_frame(text: str, label: str, out_path: Path) -> Path:
    """Kept for the static/default path: one solid-background frame, captioned."""
    img = _draw_caption(SolidBackground().frame(0, 0.0), text, label)
    img.save(out_path)
    return out_path


def _render_line_segment(vl: VoiceLine, label: str, background: BackgroundRenderer,
                          work_dir: Path, index: int) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    segment_path = work_dir / f"segment_{index:02d}.mp4"
    fps = background.frame_rate()

    if fps <= 1:
        frame_path = work_dir / f"frame_{index:02d}.png"
        _draw_caption(background.frame(0, 0.0), vl.text, label).save(frame_path)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(frame_path),
                "-i", str(vl.wav_path),
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(segment_path),
            ],
            check=True, capture_output=True,
        )
        return segment_path

    frame_dir = work_dir / f"anim_{index:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, round(vl.duration_seconds * fps))
    for f in range(n_frames):
        elapsed = f / fps
        img = _draw_caption(background.frame(f, elapsed), vl.text, label)
        img.save(frame_dir / f"f_{f:04d}.png")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps), "-i", str(frame_dir / "f_%04d.png"),
            "-i", str(vl.wav_path),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(segment_path),
        ],
        check=True, capture_output=True,
    )
    return segment_path


def _resolve_music_path(music_path) -> Path:
    """
    Validated up front by assemble_video, not at mix time. Mixing is the
    last thing assemble_video does, so checking the path there means a
    typo in PIPELINE_MUSIC_PATH is only reported after every caption
    frame has been drawn, encoded and concatenated. On the voxel
    background that is hundreds of full 1080x1920 frames per line, all
    of it discarded for a path that was knowable before the first frame.

    is_file() rather than exists(): a directory exists, so the old check
    passed one straight through to ffmpeg, which then failed at the end
    of the run with its own message about an invalid input instead.
    """
    path = Path(music_path)
    if not path.is_file():
        detail = "is a directory, not a file" if path.is_dir() else "doesn't exist"
        raise FileNotFoundError(
            f"PIPELINE_MUSIC_PATH points at {path}, which {detail}. "
            "See README 'Adding background music' for where to get a free track."
        )
    return path


def _mix_in_music(video_path: Path, music_path: Path, out_path: Path) -> Path:
    """
    Loops the track if it's shorter than the video, ducks it to well
    under the narration, and mixes rather than replaces the narration
    track. Video stream is copied untouched, only audio is re-encoded.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            "[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path


def assemble_video(
    voice_lines: List[VoiceLine],
    label: str,
    work_dir: Path,
    out_path: Path,
    background: Optional[BackgroundRenderer] = None,
    music_path: Optional[Path] = None,
) -> Path:
    background = background or SolidBackground()
    # Before anything is rendered, see _resolve_music_path.
    music_path = _resolve_music_path(music_path) if music_path else None
    work_dir.mkdir(parents=True, exist_ok=True)

    segment_paths = [
        _render_line_segment(vl, label, background, work_dir, i)
        for i, vl in enumerate(voice_lines)
    ]

    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p.resolve()}'\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_target = out_path if not music_path else work_dir / "_no_music.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(concat_target)],
        check=True, capture_output=True,
    )

    if music_path:
        _mix_in_music(concat_target, music_path, out_path)

    return out_path
