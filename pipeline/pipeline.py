"""
pipeline.py

Wires the five stages together: research, script, voice, video, metadata.

This is the only file in the project that knows about all five stages.
Every stage module above only knows its own inputs and outputs, which
is the actual payoff of the interface pattern used throughout: dropping
in a live fact source, a real LLM, or a paid voice later is a one line
change here, not a rewrite spread across the codebase.
"""

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .research import FactSource, StaticF1FactSource
from .script_gen import ScriptGenerator, TemplateScriptGenerator, Script
from .tts import VoiceSynth, EspeakVoice
from .video import assemble_video, BackgroundRenderer, SolidBackground, VoxelDropBackground
from .metadata import generate_metadata, Metadata


@dataclass
class PipelineResult:
    topic: str
    script: Script
    metadata: Metadata
    video_path: Path


class Pipeline:
    def __init__(
        self,
        fact_source: Optional[FactSource] = None,
        script_generator: Optional[ScriptGenerator] = None,
        voice_synth: Optional[VoiceSynth] = None,
        output_dir: Path = Path("outputs"),
        background: Optional[BackgroundRenderer] = None,
        music_path: Optional[Path] = None,
    ):
        self.fact_source = fact_source or StaticF1FactSource()
        self.script_generator = script_generator or TemplateScriptGenerator()
        self.voice_synth = voice_synth or EspeakVoice()
        self.output_dir = Path(output_dir)
        self.background = background or SolidBackground()
        self.music_path = music_path

    @classmethod
    def from_env(cls, output_dir: Path = Path("outputs")) -> "Pipeline":
        """
        Build a Pipeline from PIPELINE_FACTS / PIPELINE_SCRIPT / PIPELINE_VOICE /
        PIPELINE_BACKGROUND / PIPELINE_MUSIC_PATH environment variables, see
        config.py for the full list and defaults. This is how you connect
        the free upgrades (live facts, a local Ollama model, neural voice,
        an animated background, background music) without editing any code.
        """
        from .config import (
            fact_source_from_env,
            script_generator_from_env,
            voice_from_env,
            background_from_env,
            music_path_from_env,
        )

        return cls(
            fact_source=fact_source_from_env(),
            script_generator=script_generator_from_env(),
            voice_synth=voice_from_env(),
            output_dir=output_dir,
            background=background_from_env(),
            music_path=music_path_from_env(),
        )

    def run(self, topic: str) -> PipelineResult:
        facts = self.fact_source.get_facts(topic)
        script = self.script_generator.generate(topic, facts)
        metadata = generate_metadata(topic, script)

        work_dir = self.output_dir / "_work" / topic
        voice_lines = self.voice_synth.synthesize_lines(script.all_lines(), work_dir / "audio")

        # Re-seed a per-topic layout so different topics don't all get the
        # identical block skyline. zlib.crc32 rather than hash() because
        # str hashing is randomized per process by default, this needs to
        # be the same seed on every run for the same topic.
        background = self.background
        if isinstance(background, VoxelDropBackground):
            background = VoxelDropBackground(seed=zlib.crc32(topic.encode()))

        out_path = self.output_dir / f"{topic}.mp4"
        video_path = assemble_video(
            voice_lines,
            label=topic.replace("-", " "),
            work_dir=work_dir / "frames",
            out_path=out_path,
            background=background,
            music_path=self.music_path,
        )

        return PipelineResult(topic=topic, script=script, metadata=metadata, video_path=video_path)
