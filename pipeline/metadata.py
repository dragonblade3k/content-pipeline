"""
Stage 5: metadata.py

Generates the title, caption, and hashtags that ship with the finished
clip. Template based and deterministic on purpose, a good caption here
is mostly consistent formatting and a stable hashtag set, not creative
writing, so it doesn't need an LLM call to do a solid job.
"""

from dataclasses import dataclass
from typing import List

from .script_gen import Script

_BASE_HASHTAGS = ["#F1", "#Formula1", "#F1Facts", "#Motorsport"]


@dataclass
class Metadata:
    title: str
    caption: str
    hashtags: List[str]

    def caption_with_hashtags(self) -> str:
        return f"{self.caption}\n\n{' '.join(self.hashtags)}"


def generate_metadata(topic: str, script: Script) -> Metadata:
    title = script.hook.rstrip(".!?")
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "..."

    topic_tag = "#" + "".join(part.capitalize() for part in topic.split("-"))
    hashtags = _BASE_HASHTAGS + [topic_tag]

    return Metadata(title=title, caption=script.hook, hashtags=hashtags)
