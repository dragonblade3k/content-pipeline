"""
Tests for the background renderer and its environment wiring. The
subprocess backed pieces (ffmpeg encoding) are exercised by actually
running the CLI, same reasoning as tts.py and the rest of video.py,
see README. These tests cover the pure logic: frame generation is
deterministic given a seed, and the env-driven factories pick the
right implementation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

import pipeline.video as video
from pipeline.video import SolidBackground, VoxelDropBackground, WIDTH, HEIGHT


def test_solid_background_is_flat_and_static():
    bg = SolidBackground()
    assert bg.frame_rate() == 1
    img = bg.frame(0, 0.0)
    assert img.size == (WIDTH, HEIGHT)
    # same pixel everywhere for a flat color card
    assert img.getpixel((0, 0)) == img.getpixel((WIDTH - 1, HEIGHT - 1))


def test_voxel_background_is_animated():
    bg = VoxelDropBackground(seed=1)
    assert bg.frame_rate() > 1
    img = bg.frame(0, 0.0)
    assert isinstance(img, Image.Image)
    assert img.size == (WIDTH, HEIGHT)


def test_voxel_background_is_deterministic_for_same_seed():
    a = VoxelDropBackground(seed=42).frame(5, 0.4)
    b = VoxelDropBackground(seed=42).frame(5, 0.4)
    assert a.tobytes() == b.tobytes()


def test_voxel_background_differs_by_seed():
    a = VoxelDropBackground(seed=1).frame(0, 2.0)
    b = VoxelDropBackground(seed=2).frame(0, 2.0)
    assert a.tobytes() != b.tobytes()


def test_voxel_background_settles_after_drop_window():
    """Past the drop window the layout should stop changing shape drastically,
    it should only idle-bounce, i.e. two frames a full second apart once
    settled should still look like the same skyline (same block columns lit)."""
    bg = VoxelDropBackground(seed=7)
    early = bg.frame(0, 5.0)
    later = bg.frame(0, 6.0)
    assert early.size == later.size


def test_background_from_env_defaults_to_solid(monkeypatch):
    monkeypatch.delenv("PIPELINE_BACKGROUND", raising=False)
    from pipeline.config import background_from_env

    assert isinstance(background_from_env(), SolidBackground)


def test_background_from_env_selects_voxel(monkeypatch):
    monkeypatch.setenv("PIPELINE_BACKGROUND", "voxel")
    from pipeline.config import background_from_env

    assert isinstance(background_from_env(), VoxelDropBackground)


def test_background_from_env_rejects_unknown_choice(monkeypatch):
    monkeypatch.setenv("PIPELINE_BACKGROUND", "bogus")
    from pipeline.config import background_from_env

    with pytest.raises(ValueError):
        background_from_env()


def test_music_path_from_env_defaults_to_none(monkeypatch):
    monkeypatch.delenv("PIPELINE_MUSIC_PATH", raising=False)
    from pipeline.config import music_path_from_env

    assert music_path_from_env() is None


def test_music_path_from_env_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_MUSIC_PATH", str(tmp_path / "track.mp3"))
    from pipeline.config import music_path_from_env

    assert music_path_from_env() == tmp_path / "track.mp3"


def _text_height(font, text="Ayrton Senna"):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    top, bottom = draw.textbbox((0, 0), text, font=font)[1::2]
    return bottom - top


def test_load_font_fallback_keeps_requested_size_and_warns(monkeypatch):
    """With DejaVu missing, the fallback used to be Pillow's ~10px default,
    so a 64px caption silently rendered unreadably small. It must now warn
    and still come back at roughly the size asked for."""
    real_truetype = video.ImageFont.truetype

    def missing(font=None, *args, **kwargs):
        # Only file lookups fail; load_default(size=...) itself goes
        # through truetype() with an in-memory font and must still work.
        if isinstance(font, (str, Path)):
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(video.ImageFont, "truetype", missing)
    with pytest.warns(RuntimeWarning, match="DejaVuSans-Bold.ttf"):
        font = video._load_font(64, bold=True)
    assert _text_height(font) > 32


def test_load_font_does_not_warn_when_dejavu_loads(monkeypatch, recwarn):
    sentinel = object()
    monkeypatch.setattr(video.ImageFont, "truetype", lambda *a, **k: sentinel)
    assert video._load_font(40) is sentinel
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def _stub_line(tmp_path):
    from pipeline.tts import VoiceLine

    return VoiceLine(text="Senna took pole", wav_path=tmp_path / "line_00.wav", duration_seconds=1.0)


def test_assemble_video_rejects_missing_music_before_rendering(monkeypatch, tmp_path):
    """The music path used to be checked only at mix time, i.e. after every
    frame had been drawn, encoded and concatenated. A bad path must now be
    reported before the first segment is rendered."""
    rendered = []
    monkeypatch.setattr(video, "_render_line_segment", lambda *a, **k: rendered.append(a))

    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        video.assemble_video(
            [_stub_line(tmp_path)],
            label="senna",
            work_dir=tmp_path / "frames",
            out_path=tmp_path / "out.mp4",
            music_path=tmp_path / "nope.mp3",
        )

    assert rendered == []
    assert not (tmp_path / "frames").exists()


def test_resolve_music_path_rejects_a_directory(tmp_path):
    """exists() is true for a directory, so the old check let one through to
    ffmpeg, which failed at the very end of the run instead."""
    with pytest.raises(FileNotFoundError, match="is a directory"):
        video._resolve_music_path(tmp_path)


def test_resolve_music_path_accepts_a_real_file(tmp_path):
    track = tmp_path / "track.mp3"
    track.write_bytes(b"not really an mp3, but it is a file")
    assert video._resolve_music_path(str(track)) == track


def _caption_wrap(text):
    """_draw_caption's exact wrap setup: the 64px bold body face and the
    WIDTH - 160 text column it lays lines out in."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (WIDTH, HEIGHT)))
    font = video._load_font(64, bold=True)
    max_width = WIDTH - 160
    lines = video._wrap_text(draw, text, font, max_width)
    widths = [draw.textlength(line, font=font) for line in lines]
    return lines, widths, max_width


def test_wrap_text_keeps_ordinary_text_inside_the_column():
    lines, widths, max_width = _caption_wrap(
        "Ayrton Senna took pole at Monaco in 1988 by a margin nobody has "
        "come close to since, and he did it on his eleventh lap."
    )
    assert len(lines) > 1
    assert all(w <= max_width for w in widths)


def test_wrap_text_splits_a_word_too_wide_for_the_column():
    """A word wider than the text column used to be appended as its own
    line regardless. _draw_caption draws every line at a fixed x=80 with
    no clipping, so that line ran off the right edge of the 1080px frame
    with nothing reporting it. A URL in an LLM written hook is the
    realistic trigger."""
    text = "Watch the lap at youtube.com/watch?v=dQw4w9WgXcQ right now"
    lines, widths, max_width = _caption_wrap(text)
    assert all(w <= max_width for w in widths), [
        line for line, w in zip(lines, widths) if w > max_width
    ]
    # the URL could not have survived whole, so it must have been split
    assert not any("youtube.com/watch?v=dQw4w9WgXcQ" in line for line in lines)


def test_wrap_text_loses_no_characters_when_it_splits():
    """Splitting must be a reflow, not an edit: no character dropped and
    no hyphen or ellipsis invented, because in a URL or a hashtag an
    added character changes what the viewer reads the link or tag to be."""
    text = "Senna's #BrazilianGrandPrixNineteenNinetyOne drive was unreal"
    lines, _, _ = _caption_wrap(text)
    assert "".join(line.replace(" ", "") for line in lines) == "".join(text.split())


def test_wrap_text_does_not_split_a_word_that_fits_on_its_own_line():
    """The split path must only engage when starting a fresh line cannot
    help. A long-but-fitting word pushed onto its own line stays whole."""
    lines, widths, max_width = _caption_wrap("Senna Verstappen championship")
    assert all(w <= max_width for w in widths)
    assert "championship" in lines


def test_break_long_word_emits_no_empty_chunks_and_terminates():
    """A single character wider than the column has nothing left to
    split, so it is emitted as-is. The guard that allows that must not
    also let an empty chunk through, which would render as a blank
    caption line and consume vertical space for nothing."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (WIDTH, HEIGHT)))
    font = video._load_font(64, bold=True)
    chunks = video._break_long_word(draw, "WWWWW", font, max_width=1)
    assert chunks == list("WWWWW")
    assert all(chunks)
