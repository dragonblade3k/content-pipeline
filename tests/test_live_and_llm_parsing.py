"""
Tests for the parsing logic behind the network dependent stages
(LiveF1ApiFactSource, the LLM backed script generators). These can't
hit a real network from this sandbox, so instead they test the pure
parsing functions directly against a fixture shaped like the
documented API response. If the real API's shape has drifted from
what's assumed here, these are the first tests that should fail once
run somewhere with real network access, which is exactly the point of
separating parsing from the network call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from pipeline.research import LiveF1ApiFactSource
from pipeline.script_gen import (
    _split_script_response,
    _script_prompt,
    _script_from_ollama_response,
    _ollama_script_prompt,
)
from pipeline.research import Fact


# Shaped to match the documented Ergast / Jolpica driverStandings.json response.
_FIXTURE_STANDINGS_RESPONSE = {
    "MRData": {
        "StandingsTable": {
            "season": "2024",
            "StandingsLists": [
                {
                    "season": "2024",
                    "DriverStandings": [
                        {
                            "position": "1",
                            "points": "437",
                            "wins": "9",
                            "Driver": {"givenName": "Max", "familyName": "Verstappen"},
                            "Constructors": [{"name": "Red Bull"}],
                        }
                    ],
                }
            ],
        }
    }
}


def test_parse_standings_produces_two_facts():
    facts = LiveF1ApiFactSource._parse_standings(_FIXTURE_STANDINGS_RESPONSE, limit=5)
    assert len(facts) == 2
    assert all(isinstance(f, Fact) for f in facts)
    assert "Max Verstappen" in facts[0].text
    assert "P1" in facts[0].text
    assert "437 points" in facts[0].text
    assert "9 races" in facts[1].text
    assert "Red Bull" in facts[1].text


def test_parse_standings_raises_on_empty_response():
    with pytest.raises(ValueError):
        LiveF1ApiFactSource._parse_standings({"MRData": {"StandingsTable": {"StandingsLists": []}}}, limit=5)


def test_get_facts_rejects_malformed_topic():
    source = LiveF1ApiFactSource()
    with pytest.raises(ValueError):
        source.get_facts("just-a-name-no-year")


def test_split_script_response_parses_labeled_sections():
    raw = "HOOK: A great hook.\nBODY: Fact one.\nFact two.\nCTA: Follow for more."
    script = _split_script_response(raw)
    assert script.hook == "A great hook."
    assert script.lines == ["Fact one.", "Fact two."]
    assert script.cta == "Follow for more."


def test_split_script_response_ignores_chatty_preamble():
    """
    Regression test for the real failure hit running this against a
    local Ollama model: it prepended commentary before the labeled
    sections instead of starting cleanly with HOOK:, unlike Claude.
    """
    raw = (
        "Here is a spoken script for a short form video:\n\n"
        "HOOK: The king of F1 has been crowned again.\n"
        "BODY: He won nine races.\nHe finished P1.\n"
        "CTA: Follow for more."
    )
    script = _split_script_response(raw)
    assert script.hook == "The king of F1 has been crowned again."
    assert "Here is a spoken script" not in script.hook
    assert script.lines == ["He won nine races.", "He finished P1."]


def test_split_script_response_rejects_missing_sections():
    with pytest.raises(ValueError):
        _split_script_response("just some text with no labels at all")


def test_script_prompt_includes_every_fact():
    facts = [Fact(text="Fact A"), Fact(text="Fact B")]
    prompt = _script_prompt("verstappen-2024", facts)
    assert "Fact A" in prompt
    assert "Fact B" in prompt
    assert "verstappen-2024" in prompt


def test_script_from_ollama_response_parses_valid_json():
    raw = '{"hook": "A great hook.", "body": ["Fact one.", "Fact two."], "cta": "Follow for more."}'
    script = _script_from_ollama_response(raw)
    assert script.hook == "A great hook."
    assert script.lines == ["Fact one.", "Fact two."]
    assert script.cta == "Follow for more."


def test_script_from_ollama_response_rejects_invalid_json():
    """
    Regression test for the second real failure hit against llama3: it
    ignored the requested format entirely and wrote free prose instead
    of JSON. Schema constrained decoding (see OllamaScriptGenerator's
    `format` field) is what actually prevents this in practice, this
    test just confirms the parser fails loudly instead of silently if
    it ever does get malformed input.
    """
    with pytest.raises(ValueError):
        _script_from_ollama_response("Here is the script you requested:\n\nNot JSON at all.")


def test_script_from_ollama_response_rejects_incomplete_json():
    with pytest.raises(ValueError):
        _script_from_ollama_response('{"hook": "A hook.", "body": [], "cta": "Follow."}')


def test_ollama_script_prompt_requests_exact_fact_count():
    facts = [Fact(text="Fact A"), Fact(text="Fact B"), Fact(text="Fact C")]
    prompt = _ollama_script_prompt("senna", facts)
    assert "exactly 3 sentences" in prompt
    assert "Fact A" in prompt and "Fact B" in prompt and "Fact C" in prompt
