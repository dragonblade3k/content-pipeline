"""
Tests for how the voice stage reports a backend that will not run.

The synthesis itself still is not mocked, same reasoning as the rest of
the subprocess backed stages, see README. What is covered here is the
error contract around it: when espeak-ng or piper is missing or exits
non-zero, the caller has to be told what happened and what to do, the
way every other external dependency in this project already does.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import pipeline.tts as tts
from pipeline.tts import EspeakVoice, _run_voice_tool


def test_run_voice_tool_surfaces_the_tools_own_stderr():
    with pytest.raises(RuntimeError) as excinfo:
        _run_voice_tool(
            ["sh", "-c", "echo 'unknown voice en-zz' >&2; exit 3"],
            install_hint="unused here",
        )
    message = str(excinfo.value)
    assert "unknown voice en-zz" in message
    assert "3" in message


def test_run_voice_tool_names_a_missing_binary_and_how_to_install_it():
    with pytest.raises(FileNotFoundError) as excinfo:
        _run_voice_tool(
            ["definitely-not-a-real-binary-xyz"],
            install_hint="Install it with: sudo apt-get install -y espeak-ng",
        )
    message = str(excinfo.value)
    assert "definitely-not-a-real-binary-xyz" in message
    assert "sudo apt-get install -y espeak-ng" in message


def test_run_voice_tool_still_passes_stdin_to_the_process(tmp_path):
    """
    Piper takes its line on stdin rather than in argv. If that route
    ever stopped being wired through, the failure would be silent
    narration rather than an error, so it gets its own guard.
    """
    target = tmp_path / "captured.txt"
    _run_voice_tool(
        ["sh", "-c", f"cat > {target}"],
        install_hint="unused here",
        stdin=b"the line to speak",
    )
    assert target.read_text() == "the line to speak"


def test_espeak_voice_tells_you_to_install_espeak_when_it_is_absent(monkeypatch, tmp_path):
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "espeak-ng")

    monkeypatch.setattr(subprocess, "run", missing_binary)

    with pytest.raises(FileNotFoundError) as excinfo:
        EspeakVoice().synthesize_lines(["one line"], tmp_path / "audio")
    assert "apt-get install -y espeak-ng" in str(excinfo.value)
