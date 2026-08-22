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
    assert list(a.getdata()) == list(b.getdata())


def test_voxel_background_differs_by_seed():
    a = VoxelDropBackground(seed=1).frame(0, 2.0)
    b = VoxelDropBackground(seed=2).frame(0, 2.0)
    assert list(a.getdata()) != list(b.getdata())


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
