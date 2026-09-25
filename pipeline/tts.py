"""
Stage 3: tts.py

Turns spoken lines into narration audio.

Same interface pattern as the earlier stages: VoiceSynth is the
contract. EspeakVoice is the default and the one actually exercised in
this sandbox, it needs nothing beyond the espeak-ng binary (installed
via apt, no internet beyond the package mirror), no model download, no
account, no key. Quality is dated formant synthesis and it sounds
robotic, but it is real generated audio, not a stub, and it runs
anywhere including fully offline. PiperVoice is the quality upgrade
path: real neural TTS, wired to a local .onnx model file, verified end
to end on real hardware, see its own docstring. ElevenLabsVoice is a
stub for when a paid key is available.

Every implementation returns per line durations so video.py can sync
captions to the audio without a separate forced alignment step.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
import subprocess
import wave


@dataclass
class VoiceLine:
    text: str
    wav_path: Path
    duration_seconds: float


class VoiceSynth(ABC):
    @abstractmethod
    def synthesize_lines(self, lines: List[str], out_dir: Path) -> List[VoiceLine]:
        """Render each line to its own wav file, in order, in out_dir."""
        raise NotImplementedError


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _run_voice_tool(cmd: List[str], install_hint: str, stdin: Optional[bytes] = None) -> None:
    """
    Run a voice backend's binary and turn its two failure modes into the
    kind of error the rest of this project raises.

    Every other stage that depends on something external says what went
    wrong and what to do about it: a missing ANTHROPIC_API_KEY, an
    unreachable Ollama, a missing Piper model. The subprocess backed
    stages were the exception. `capture_output=True` sends the tool's
    own stderr somewhere nobody ever reads it, so a failing espeak-ng
    surfaced only as "returned non-zero exit status 1" with its actual
    complaint discarded, and a missing binary surfaced as a bare
    FileNotFoundError naming the executable and nothing else.

    stderr is still captured rather than inherited, so a successful run
    stays quiet on the terminal. It is now attached to the exception
    instead of dropped.
    """
    try:
        subprocess.run(cmd, input=stdin, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"'{cmd[0]}' is not installed or not on PATH. {install_hint}"
        ) from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace").strip()
        tail = "\n".join(detail.splitlines()[-10:])
        raise RuntimeError(
            f"'{cmd[0]}' failed with exit status {e.returncode}."
            + (f" It reported:\n{tail}" if tail else " It wrote nothing to stderr.")
        ) from e


class EspeakVoice(VoiceSynth):
    """Local, offline, no key, no model download. The default for this project."""

    def __init__(self, voice: str = "en-us", speed_wpm: int = 165):
        self._voice = voice
        self._speed = speed_wpm

    def synthesize_lines(self, lines: List[str], out_dir: Path) -> List[VoiceLine]:
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, text in enumerate(lines):
            wav_path = out_dir / f"line_{i:02d}.wav"
            _run_voice_tool(
                ["espeak-ng", "-v", self._voice, "-s", str(self._speed), "-w", str(wav_path), text],
                install_hint="Install it with: sudo apt-get install -y espeak-ng",
            )
            results.append(
                VoiceLine(text=text, wav_path=wav_path, duration_seconds=_wav_duration(wav_path))
            )
        return results


class PiperVoice(VoiceSynth):
    """
    Neural TTS, much better quality than espeak. Needs a local .onnx
    voice model plus its .json config. Download one from
    https://huggingface.co/rhasspy/piper-voices, see the README's
    "connecting the free upgrades" section for the exact commands.
    Verified end to end with the en_US-lessac-medium voice.
    """

    def __init__(self, model_path: Path, config_path: Optional[Path] = None):
        self._model_path = Path(model_path)
        self._config_path = Path(config_path) if config_path else Path(str(model_path) + ".json")
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at {self._model_path}. Download one from "
                "https://huggingface.co/rhasspy/piper-voices and point PiperVoice at it."
            )

    def synthesize_lines(self, lines: List[str], out_dir: Path) -> List[VoiceLine]:
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, text in enumerate(lines):
            wav_path = out_dir / f"line_{i:02d}.wav"
            _run_voice_tool(
                ["piper", "-m", str(self._model_path), "-c", str(self._config_path), "-f", str(wav_path)],
                install_hint="Install it with: pip install piper-tts (it is in requirements.txt).",
                stdin=text.encode("utf-8"),
            )
            results.append(
                VoiceLine(text=text, wav_path=wav_path, duration_seconds=_wav_duration(wav_path))
            )
        return results


class ElevenLabsVoice(VoiceSynth):
    """Stub. Wire this to the ElevenLabs API once a key is available, see README."""

    def __init__(self, voice_id: str = "default"):
        self._voice_id = voice_id

    def synthesize_lines(self, lines: List[str], out_dir: Path) -> List[VoiceLine]:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set. See README for wiring instructions.")
        raise NotImplementedError("Wire this up to the ElevenLabs API, see README.")
