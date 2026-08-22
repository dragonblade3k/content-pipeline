"""
Unit tests for the parts of the pipeline that don't need a subprocess
call (espeak, ffmpeg). Run with: python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from pipeline.research import StaticF1FactSource, Fact
from pipeline.script_gen import TemplateScriptGenerator, Script
from pipeline.metadata import generate_metadata


def test_static_fact_source_lists_topics():
    source = StaticF1FactSource()
    topics = source.topics()
    assert "senna" in topics
    assert topics == sorted(topics)


def test_static_fact_source_returns_facts_for_known_topic():
    source = StaticF1FactSource()
    facts = source.get_facts("senna", limit=3)
    assert len(facts) == 3
    assert all(isinstance(f, Fact) for f in facts)
    assert all(f.text for f in facts)


def test_static_fact_source_raises_on_unknown_topic():
    source = StaticF1FactSource()
    with pytest.raises(KeyError):
        source.get_facts("nonexistent-topic")


def test_template_script_generator_builds_full_script():
    source = StaticF1FactSource()
    facts = source.get_facts("schumacher")
    script = TemplateScriptGenerator().generate("schumacher", facts)

    assert isinstance(script, Script)
    assert script.hook
    assert len(script.lines) == len(facts)
    assert script.cta
    assert len(script.all_lines()) == len(facts) + 2  # hook + facts + cta


def test_template_script_generator_raises_on_empty_facts():
    with pytest.raises(ValueError):
        TemplateScriptGenerator().generate("empty-topic", [])


def test_metadata_title_is_never_longer_than_60_chars():
    source = StaticF1FactSource()
    for topic in source.topics():
        facts = source.get_facts(topic)
        script = TemplateScriptGenerator().generate(topic, facts)
        meta = generate_metadata(topic, script)
        assert len(meta.title) <= 60
        assert meta.hashtags[0] == "#F1"


def test_metadata_includes_topic_specific_hashtag():
    source = StaticF1FactSource()
    facts = source.get_facts("hamilton-verstappen-2021")
    script = TemplateScriptGenerator().generate("hamilton-verstappen-2021", facts)
    meta = generate_metadata("hamilton-verstappen-2021", script)
    assert "#HamiltonVerstappen2021" in meta.hashtags
