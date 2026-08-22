"""
Stage 2: script_gen.py

Turns a topic's facts into a short spoken script for a 30 to 45 second clip.

Same interface pattern as research.py: ScriptGenerator is the contract.
TemplateScriptGenerator is a deterministic, fully offline implementation
that works today with zero setup, useful for testing the rest of the
pipeline without spending API credits. AnthropicScriptGenerator is a
real LLM backed implementation wired to an environment variable so it
drops in the moment a key is available. Nothing in tts.py or video.py
needs to know or care which one produced the script, they only see a
Script object.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import os

from .research import Fact


@dataclass
class Script:
    hook: str
    lines: List[str]
    cta: str

    def full_text(self) -> str:
        return " ".join([self.hook, *self.lines, self.cta])

    def all_lines(self) -> List[str]:
        """Every spoken line in order, used for caption timing in video.py."""
        return [self.hook, *self.lines, self.cta]


class ScriptGenerator(ABC):
    @abstractmethod
    def generate(self, topic: str, facts: List[Fact]) -> Script:
        raise NotImplementedError


class TemplateScriptGenerator(ScriptGenerator):
    """
    Deterministic, offline, no API key required. The hook isn't going to
    be as sharp as an LLM written one, but the output is honest, fully
    inspectable, and it means the pipeline runs end to end today with
    nothing to sign up for.
    """

    _HOOKS = {
        "record": "Here's a Formula 1 record most fans get wrong.",
        "history": "This moment changed Formula 1 forever.",
        "rivalry": "This rivalry defined a generation of Formula 1.",
        "controversy": "This is still one of F1's most argued about moments.",
        "engineering": "Here's the engineering story behind this.",
        "legacy": "Here's why this still matters in Formula 1 today.",
    }
    _DEFAULT_HOOK = "Here's something worth knowing about Formula 1."

    def generate(self, topic: str, facts: List[Fact]) -> Script:
        if not facts:
            raise ValueError(f"No facts to build a script from for '{topic}'")
        lead_category = facts[0].category
        hook = self._HOOKS.get(lead_category, self._DEFAULT_HOOK)
        lines = [fact.text for fact in facts]
        cta = "Follow for more Formula 1 history, one fact at a time."
        return Script(hook=hook, lines=lines, cta=cta)


class AnthropicScriptGenerator(ScriptGenerator):
    """
    Real LLM backed script writer. Reads ANTHROPIC_API_KEY from the
    environment, a key is never hardcoded in source. Raises a clear
    error if the key is missing instead of failing silently or quietly
    falling back to something else.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self._model = os.environ.get("ANTHROPIC_MODEL", model)

    def generate(self, topic: str, facts: List[Fact]) -> Script:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using "
                "AnthropicScriptGenerator, or pass TemplateScriptGenerator "
                "to the pipeline instead."
            )
        import anthropic  # imported lazily, this module has no hard dependency on it

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": _script_prompt(topic, facts)}],
        )
        return _split_script_response(response.content[0].text)


def _split_script_response(text: str) -> Script:
    """
    Looks for HOOK: / BODY: / CTA: labels anywhere in the text rather
    than assuming the whole response is exactly three '---' separated
    parts.

    That stricter version was the first cut and it broke the first
    time this was run against a real local model: Claude follows a
    "return exactly this, nothing else" instruction reliably, llama3
    does not, it prepends chatty preamble like "Here is a spoken
    script for a short form video:" before the actual content. Rather
    than fight every model into never doing that, the parser just
    finds the labeled sections and ignores whatever comes before them.
    """
    import re

    def _section(label: str, next_labels: List[str]) -> str:
        boundary = "|".join(next_labels) if next_labels else r"$(?!)"
        pattern = rf"{label}:\s*(.*?)(?=\n(?:{boundary}):|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(f"Could not find a '{label}:' section in the model's response:\n{text}")
        return match.group(1).strip()

    hook = _section("HOOK", ["BODY", "CTA"])
    body = _section("BODY", ["CTA"])
    cta = _section("CTA", [])

    lines = [line.strip("-• ").strip() for line in body.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"BODY section parsed empty, raw response:\n{text}")
    return Script(hook=hook, lines=lines, cta=cta)


def _script_prompt(topic: str, facts: List[Fact]) -> str:
    fact_block = "\n".join(f"- {f.text}" for f in facts)
    return (
        f"Write a punchy 30 to 45 second spoken script about {topic} for a "
        f"short form video, using only these verified facts, do not invent "
        f"any new facts:\n{fact_block}\n\n"
        "Respond with exactly this format and nothing else, no preamble, "
        "no introduction, no commentary before or after:\n\n"
        "HOOK: <one sentence hook>\n"
        "BODY: <one short sentence per fact, same order, one per line>\n"
        "CTA: <one sentence call to action to follow for more>"
    )


_OLLAMA_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string"},
        "body": {"type": "array", "items": {"type": "string"}},
        "cta": {"type": "string"},
    },
    "required": ["hook", "body", "cta"],
}


def _ollama_script_prompt(topic: str, facts: List[Fact]) -> str:
    fact_block = "\n".join(f"{i + 1}. {f.text}" for i, f in enumerate(facts))
    return (
        f"Write a punchy 30 to 45 second spoken script about {topic} for a "
        f"short form video, using only these verified facts, do not invent "
        f"any new facts and do not skip any of them:\n{fact_block}\n\n"
        "hook: a one sentence hook.\n"
        f"body: exactly {len(facts)} sentences, one per fact above, same order.\n"
        "cta: a one sentence call to action to follow for more."
    )


class OllamaScriptGenerator(ScriptGenerator):
    """
    Real LLM backed script writer that runs fully local through Ollama,
    zero cost per call since it's your own hardware doing the inference,
    not a metered API. This is the best quality for free option in this
    project: better than the template, no key, no per token cost, and
    it reuses the exact tool you already run for the F1 discovery
    platform and AuraOS instead of introducing a new dependency.

    Needs Ollama running locally (`ollama serve`, or it's already
    running if you use the desktop app) with a model pulled first,
    e.g. `ollama pull llama3.1`. Uses only the standard library for the
    HTTP call, no extra pip install needed for this class specifically.

    This went through two real failed attempts before landing here,
    worth knowing for an interview. First cut asked for three '---'
    separated parts: llama3 prepended chatty preamble before the
    content. Second cut switched to HOOK:/BODY:/CTA: labels with a
    parser that ignores anything before them: on the very next run
    llama3 skipped the labels entirely and just wrote free prose, a
    different failure, not a smaller version of the first one. That
    ruled out "parse more leniently" as the fix, the model just isn't
    reliable at following a text formatting instruction at all.

    The actual fix: Ollama's `format` field accepts a JSON schema and
    constrains decoding to match it, so the model cannot produce
    something that fails to parse, it's not a request the model can
    ignore. Verified end to end, valid JSON on every run. Claude, by
    contrast, followed the plain text instruction correctly every
    time in this project, so AnthropicScriptGenerator above never
    needed this, that gap between the two is itself worth remembering.
    """

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self._host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, topic: str, facts: List[Fact]) -> Script:
        import json
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self._model,
            "prompt": _ollama_script_prompt(topic, facts),
            "format": _OLLAMA_SCRIPT_SCHEMA,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}. Is it running, and "
                f"have you pulled the model? (`ollama pull {self._model}`). "
                f"Original error: {e}"
            ) from e

        return _script_from_ollama_response(result.get("response", ""))


def _script_from_ollama_response(raw_response: str) -> Script:
    """Pulled out of generate() so it's testable without a real Ollama call."""
    import json

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Ollama did not return valid JSON despite the format schema, "
            f"raw response:\n{raw_response}"
        ) from e

    hook = (parsed.get("hook") or "").strip()
    lines = [line.strip() for line in parsed.get("body", []) if line and line.strip()]
    cta = (parsed.get("cta") or "").strip()
    if not hook or not lines or not cta:
        raise ValueError(f"Incomplete script from Ollama: {parsed}")
    return Script(hook=hook, lines=lines, cta=cta)
