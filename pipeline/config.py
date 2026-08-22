"""
config.py

Environment driven provider selection. Swapping which implementation
each stage uses is a shell variable, not a code edit, that's the whole
payoff of building every stage behind an interface in the first place.

    PIPELINE_FACTS=static|live          default: static
    PIPELINE_SCRIPT=template|ollama|anthropic   default: template
    PIPELINE_VOICE=espeak|piper         default: espeak
    PIPELINE_BACKGROUND=solid|voxel     default: solid

    OLLAMA_MODEL        model name for the ollama backend, default llama3.1
    OLLAMA_HOST         default http://localhost:11434
    ANTHROPIC_API_KEY   required if PIPELINE_SCRIPT=anthropic
    ANTHROPIC_MODEL      optional override, see script_gen.py for the default
    PIPER_MODEL_PATH    path to a downloaded .onnx voice model,
                         required if PIPELINE_VOICE=piper
    PIPELINE_MUSIC_PATH  path to a background music file (mp3/wav), mixed
                         under the narration if set, silent if unset

Every default here matches the original zero setup, zero cost path, so
running the CLI with no environment variables set behaves exactly as
it did before this file existed.
"""

import os
from pathlib import Path
from typing import Optional

from .research import FactSource, StaticF1FactSource, LiveF1ApiFactSource
from .script_gen import (
    ScriptGenerator,
    TemplateScriptGenerator,
    AnthropicScriptGenerator,
    OllamaScriptGenerator,
)
from .tts import VoiceSynth, EspeakVoice, PiperVoice
from .video import BackgroundRenderer, SolidBackground, VoxelDropBackground


def fact_source_from_env() -> FactSource:
    choice = os.environ.get("PIPELINE_FACTS", "static")
    if choice == "live":
        return LiveF1ApiFactSource()
    if choice != "static":
        raise ValueError(f"Unknown PIPELINE_FACTS='{choice}', expected 'static' or 'live'")
    return StaticF1FactSource()


def script_generator_from_env() -> ScriptGenerator:
    choice = os.environ.get("PIPELINE_SCRIPT", "template")
    if choice == "ollama":
        return OllamaScriptGenerator()
    if choice == "anthropic":
        return AnthropicScriptGenerator()
    if choice != "template":
        raise ValueError(f"Unknown PIPELINE_SCRIPT='{choice}', expected 'template', 'ollama', or 'anthropic'")
    return TemplateScriptGenerator()


def voice_from_env() -> VoiceSynth:
    choice = os.environ.get("PIPELINE_VOICE", "espeak")
    if choice == "piper":
        model_path = os.environ.get("PIPER_MODEL_PATH")
        if not model_path:
            raise RuntimeError("Set PIPER_MODEL_PATH to a downloaded .onnx voice model to use PIPELINE_VOICE=piper")
        return PiperVoice(Path(model_path))
    if choice != "espeak":
        raise ValueError(f"Unknown PIPELINE_VOICE='{choice}', expected 'espeak' or 'piper'")
    return EspeakVoice()


def background_from_env(seed: int = 0) -> BackgroundRenderer:
    choice = os.environ.get("PIPELINE_BACKGROUND", "solid")
    if choice == "voxel":
        return VoxelDropBackground(seed=seed)
    if choice != "solid":
        raise ValueError(f"Unknown PIPELINE_BACKGROUND='{choice}', expected 'solid' or 'voxel'")
    return SolidBackground()


def music_path_from_env() -> Optional[Path]:
    raw = os.environ.get("PIPELINE_MUSIC_PATH")
    return Path(raw) if raw else None
