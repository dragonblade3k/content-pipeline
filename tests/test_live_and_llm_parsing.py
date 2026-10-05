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


def test_split_script_response_does_not_absorb_the_next_label_as_content():
    r"""
    Regression test for a bare "HOOK:" line. The separator between a
    label and its content used to be \s*, which swallowed the newline
    in front of "BODY:" and so hid that label from the section
    boundary. The hook came back as the literal text "BODY: Senna won
    three titles.", which is then narrated as the opening line and
    reused as the clip's title and caption, and the same sentence was
    parsed a second time as the body.
    """
    raw = "HOOK:\nBODY: Senna won three titles.\nCTA: Follow for more."
    with pytest.raises(ValueError) as excinfo:
        _split_script_response(raw)
    assert "HOOK" in str(excinfo.value)


def test_split_script_response_rejects_present_but_empty_sections():
    """
    An empty hook or cta reaches the voice stage as a line with nothing
    to say. _script_from_ollama_response already rejects this shape on
    the JSON path; the labeled text path has to agree.
    """
    for raw in (
        "HOOK:\nBODY: Senna won three titles.\nCTA: Follow for more.",
        "HOOK: Here is a fact.\nBODY: Senna won three titles.\nCTA:",
    ):
        with pytest.raises(ValueError):
            _split_script_response(raw)


def test_split_script_response_allows_content_on_the_line_after_a_label():
    """
    Tightening the label separator to spaces and tabs must not stop a
    model from putting a section's text on the following line, since
    the capture itself still spans newlines.
    """
    raw = "HOOK:\n  The king of F1 is crowned again.\nBODY: He won nine races.\nCTA: Follow."
    script = _split_script_response(raw)
    assert script.hook == "The king of F1 is crowned again."
    assert script.lines == ["He won nine races."]
    assert script.cta == "Follow."


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


def test_script_from_ollama_response_rejects_string_body():
    """
    The real hazard behind the schema. A model writing the body as one
    string instead of an array is valid JSON, and iterating a string
    yields its characters, so this used to parse "happily" into one
    spoken line per character: a script of 28 single letters, each one
    narrated and encoded as its own video segment. Nothing downstream
    validates line count or length, so the only place this can be
    caught is here.
    """
    raw = '{"hook": "A hook.", "body": "He won nine races. He finished P1.", "cta": "Follow."}'
    with pytest.raises(ValueError, match="list of strings"):
        _script_from_ollama_response(raw)


def test_script_from_ollama_response_rejects_non_string_body_entries():
    with pytest.raises(ValueError, match="list of strings"):
        _script_from_ollama_response('{"hook": "A hook.", "body": [1, 2], "cta": "Follow."}')


def test_script_from_ollama_response_rejects_non_object_json():
    """
    Valid JSON that isn't an object at all: parsed.get() used to raise a
    bare AttributeError from inside the parser rather than this module's
    own error, which told you nothing about what the model did wrong.
    """
    for raw in ['"just a sentence"', "null", '["a", "b"]']:
        with pytest.raises(ValueError, match="asks for an object"):
            _script_from_ollama_response(raw)


def test_script_from_ollama_response_rejects_non_string_hook():
    with pytest.raises(ValueError, match="'hook' must be a string"):
        _script_from_ollama_response('{"hook": 5, "body": ["Fact one."], "cta": "Follow."}')


def test_script_from_ollama_response_keeps_accepting_a_conforming_body():
    """
    The guard must not narrow what already worked: blank and whitespace
    only entries are still dropped rather than rejected, since the
    previous parser tolerated them and a model padding its array is not
    a malformed response.
    """
    raw = '{"hook": "A hook.", "body": ["Fact one.", "", "   ", "Fact two."], "cta": "Follow."}'
    script = _script_from_ollama_response(raw)
    assert script.lines == ["Fact one.", "Fact two."]


# --- The live source's network boundary -------------------------------
#
# urllib.error.HTTPError subclasses URLError, so a single
# `except (URLError, TimeoutError)` clause catches every HTTP status the
# API returns and reports it as an unreachable network. The mistake the
# README itself warns about, a driverId that is not simply the lowercase
# surname, arrives as a 404 and used to send the caller off to debug
# their network instead of their topic. These pin each failure to the
# thing that actually went wrong.


class _FakeResponse:
    """Minimal stand in for the object urlopen yields as a context manager."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, reason):
    import urllib.error

    return urllib.error.HTTPError("http://example.invalid", code, reason, {}, None)


def _get_facts_raising(raised):
    """Call get_facts with urlopen replaced, and return the exception raised."""
    from unittest import mock

    source = LiveF1ApiFactSource()
    with mock.patch("urllib.request.urlopen", side_effect=raised):
        with pytest.raises(Exception) as caught:
            source.get_facts("verstapen-2024")
    return caught.value


def test_unknown_driver_id_is_reported_as_a_bad_topic_not_a_bad_network():
    err = _get_facts_raising(_http_error(404, "Not Found"))
    assert isinstance(err, ValueError)
    assert "verstapen" in str(err)
    assert "drivers.json" in str(err)
    assert "sandbox" not in str(err).lower()


def test_rate_limiting_is_reported_as_rate_limiting():
    err = _get_facts_raising(_http_error(429, "Too Many Requests"))
    assert isinstance(err, RuntimeError)
    assert "rate limit" in str(err).lower()
    assert "sandbox" not in str(err).lower()


def test_other_http_statuses_name_the_status():
    err = _get_facts_raising(_http_error(500, "Internal Server Error"))
    assert isinstance(err, RuntimeError)
    assert "500" in str(err)
    assert "sandbox" not in str(err).lower()


def test_a_genuine_network_failure_still_names_the_sandbox_limitation():
    import urllib.error

    err = _get_facts_raising(urllib.error.URLError("Name or service not known"))
    assert isinstance(err, RuntimeError)
    assert "sandbox" in str(err).lower()


def test_a_non_json_body_is_reported_as_a_non_json_body():
    from unittest import mock

    source = LiveF1ApiFactSource()
    html = b"<html><head><title>403 Forbidden</title></head></html>"
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(html)):
        with pytest.raises(RuntimeError) as caught:
            source.get_facts("norris-2024")
    assert "not JSON" in str(caught.value)


def test_empty_standings_also_points_at_the_driver_id_lookup():
    with pytest.raises(ValueError) as caught:
        LiveF1ApiFactSource._parse_standings(
            {"MRData": {"StandingsTable": {"StandingsLists": []}}}, limit=5
        )
    assert "drivers.json" in str(caught.value)


# --- The Ollama backend's network boundary -----------------------------
#
# The same HTTPError-subclasses-URLError trap the live fact source fell
# into, in a second place. A single `except (URLError, TimeoutError)`
# clause reported every status Ollama answered with as "is Ollama
# running?", and the two statuses that actually happen both mean it is:
# a 404 for a model that was never pulled, and a 400 from an Ollama too
# old to accept a JSON schema in `format`, the limitation the generator's
# own docstring warns about. Ollama's own explanation in the response
# body was discarded along with the status.


def _ollama_http_error(code, reason, body=b""):
    import io
    import urllib.error

    return urllib.error.HTTPError(
        "http://localhost:11434/api/generate", code, reason, {}, io.BytesIO(body)
    )


def _generate_raising(raised):
    """Call generate() with urlopen replaced, and return the exception raised."""
    from unittest import mock

    from pipeline.script_gen import OllamaScriptGenerator

    generator = OllamaScriptGenerator(model="llama3.1", host="http://localhost:11434")
    facts = [Fact(text="Senna took 65 pole positions.", year=1994, category="record")]
    with mock.patch("urllib.request.urlopen", side_effect=raised):
        with pytest.raises(Exception) as caught:
            generator.generate("senna", facts)
    return caught.value


def test_a_model_that_was_never_pulled_is_not_reported_as_a_dead_server():
    err = _generate_raising(
        _ollama_http_error(
            404, "Not Found", b'{"error":"model \'llama3.1\' not found"}'
        )
    )
    assert isinstance(err, RuntimeError)
    assert "ollama pull llama3.1" in str(err)
    assert "model 'llama3.1' not found" in str(err)
    assert "is it running" not in str(err).lower()


def test_a_rejected_request_points_at_the_format_field_not_the_connection():
    err = _generate_raising(
        _ollama_http_error(400, "Bad Request", b'{"error":"invalid format"}')
    )
    assert isinstance(err, RuntimeError)
    assert "400" in str(err)
    assert "invalid format" in str(err)
    assert "is it running" not in str(err).lower()


def test_other_ollama_statuses_name_the_status_and_quote_the_body():
    err = _generate_raising(
        _ollama_http_error(500, "Internal Server Error", b'{"error":"out of memory"}')
    )
    assert isinstance(err, RuntimeError)
    assert "500" in str(err)
    assert "out of memory" in str(err)
    assert "is it running" not in str(err).lower()


def test_an_unreadable_error_body_still_yields_the_status():
    err = _generate_raising(_ollama_http_error(503, "Service Unavailable", b"<html>"))
    assert isinstance(err, RuntimeError)
    assert "503" in str(err)
    assert "<html>" in str(err)


def test_an_empty_error_body_still_yields_the_status():
    err = _generate_raising(_ollama_http_error(502, "Bad Gateway"))
    assert isinstance(err, RuntimeError)
    assert "502" in str(err)


def test_a_genuine_connection_failure_still_asks_whether_ollama_is_running():
    import urllib.error

    err = _generate_raising(urllib.error.URLError("Connection refused"))
    assert isinstance(err, RuntimeError)
    assert "is it running" in str(err).lower()
    assert "ollama serve" in str(err)


# --- The standings response's own shape --------------------------------
#
# The third place here where "it answered" was confused with "it
# answered the way this code expects". The HTTP layer, the non JSON
# body and Ollama's decoded script are all checked now, but a
# driverStandings payload that parsed as JSON was still indexed blind:
# a renamed or dropped field came back as a bare KeyError naming only
# the key, from the one function the README sends the reader to when
# the schema drifts.


_MISSING = object()


def _standings(season="2024", **overrides):
    """A valid standings entry, with whatever the test wants broken."""
    entry = {
        "position": "1",
        "points": "437",
        "wins": "9",
        "Driver": {"givenName": "Max", "familyName": "Verstappen"},
        "Constructors": [{"name": "Red Bull"}],
    }
    for key, value in overrides.items():
        if value is _MISSING:
            entry.pop(key, None)
        else:
            entry[key] = value
    standings_list = {"DriverStandings": [entry]}
    if season is not _MISSING:
        standings_list["season"] = season
    return {"MRData": {"StandingsTable": {"StandingsLists": [standings_list]}}}


def _parse_raising(data):
    with pytest.raises(Exception) as caught:
        LiveF1ApiFactSource._parse_standings(data, limit=5)
    return caught.value


def test_a_renamed_constructors_field_names_the_field_not_a_bare_keyerror():
    err = _parse_raising(_standings(Constructors=_MISSING))
    assert isinstance(err, RuntimeError)
    assert "Constructors" in str(err)
    assert "_parse_standings" in str(err)


def test_a_missing_field_lists_the_fields_that_are_there():
    err = _parse_raising(_standings(points=_MISSING))
    assert "points" in str(err)
    assert "wins" in str(err)  # the fields it does have, to compare against


def test_an_empty_constructors_list_is_reported_as_a_missing_team_name():
    err = _parse_raising(_standings(Constructors=[]))
    assert isinstance(err, RuntimeError)
    assert "no team name" in str(err)


def test_a_driver_object_missing_a_name_part_names_that_part():
    err = _parse_raising(_standings(Driver={"givenName": "Max"}))
    assert isinstance(err, RuntimeError)
    assert "familyName" in str(err)
    assert "Driver object" in str(err)


def test_a_standings_entry_that_is_not_an_object_is_reported_as_a_shape_problem():
    data = {"MRData": {"StandingsTable": {"StandingsLists": [
        {"season": "2024", "DriverStandings": ["Max Verstappen"]}
    ]}}}
    err = _parse_raising(data)
    assert isinstance(err, RuntimeError)
    assert "str" in str(err)


def test_a_non_numeric_season_is_reported_instead_of_crashing_on_int():
    err = _parse_raising(_standings(season="twenty twenty four"))
    assert isinstance(err, RuntimeError)
    assert "not a year" in str(err)


def test_a_schema_drift_never_sends_the_caller_to_check_their_network():
    err = _parse_raising(_standings(Constructors=_MISSING))
    lowered = str(err).lower()
    assert "network" not in lowered
    assert "sandbox" not in lowered
    assert "reach" not in lowered


def test_an_empty_field_value_is_not_treated_as_drift():
    facts = LiveF1ApiFactSource._parse_standings(_standings(points=""), limit=5)
    assert "with  points" in facts[0].text  # reported as sent, not rejected


def test_a_single_win_stays_singular_even_if_the_count_arrives_as_a_number():
    facts = LiveF1ApiFactSource._parse_standings(_standings(wins=1), limit=5)
    assert "won 1 race in" in facts[1].text



def test_a_standings_list_with_no_season_at_all_names_the_season_field():
    err = _parse_raising(_standings(season=_MISSING))
    assert isinstance(err, RuntimeError)
    assert "season" in str(err)
    assert "standings list" in str(err)
