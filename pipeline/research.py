"""
Stage 1: research.py

Turns a topic into a small set of facts a script can be built from.

Design note, worth remembering for an interview: this sits behind a
FactSource interface on purpose. The sandbox this was first built in
has network access locked to package registries only, so the live F1
API (Jolpica-F1, a free keyless REST API that is endpoint compatible
with the old Ergast API) is not reachable from here. Rather than fake
a network call and pretend, stage 1 ships with a curated static
dataset and a documented, unimplemented extension point for the live
source. Swapping StaticF1FactSource for LiveF1ApiFactSource later is a
one line change in pipeline.py, nothing downstream has to know the
difference. That is the point of coding to an interface instead of a
concrete class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json


@dataclass(frozen=True)
class Fact:
    text: str
    year: Optional[int] = None
    category: str = "general"


class FactSource(ABC):
    @abstractmethod
    def get_facts(self, topic: str, limit: int = 5) -> List[Fact]:
        """Return up to `limit` facts about `topic`."""
        raise NotImplementedError

    @abstractmethod
    def topics(self) -> List[str]:
        """Return the topics this source currently knows about."""
        raise NotImplementedError


class StaticF1FactSource(FactSource):
    """Curated facts shipped in pipeline/data/f1_facts.json. No network required."""

    _DATA_PATH = Path(__file__).parent / "data" / "f1_facts.json"

    def __init__(self, data_path: Optional[Path] = None):
        path = data_path or self._DATA_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._facts_by_topic = {
            topic: [Fact(**fact) for fact in facts] for topic, facts in raw.items()
        }

    def topics(self) -> List[str]:
        return sorted(self._facts_by_topic.keys())

    def get_facts(self, topic: str, limit: int = 5) -> List[Fact]:
        key = topic.strip().lower()
        if key not in self._facts_by_topic:
            raise KeyError(
                f"No facts for '{topic}'. Available topics: {', '.join(self.topics())}"
            )
        return self._facts_by_topic[key][:limit]


class LiveF1ApiFactSource(FactSource):
    """
    Pulls real driver standings from the Jolpica F1 API, a free, keyless
    REST API that is endpoint compatible with the old Ergast API, and
    turns them into Fact objects.

    Topic format is different from the static source on purpose, since
    this isn't a fixed curated list: pass "<driverId>-<season>", for
    example "norris-2024" or "hamilton-2020". driverId is usually the
    driver's lowercase surname, but not always, verified live: it's
    "max_verstappen", not "verstappen", because Ergast disambiguates
    against his father Jos Verstappen who also raced in F1. If you're
    not sure of one, look it up first with:

        curl https://api.jolpi.ca/ergast/f1/2024/drivers.json

    This was actually run and verified against the live endpoint
    (see the README's "connecting the free upgrades" section for the
    exact commands and what broke on the first try). If the schema
    drifts in the future, _parse_standings below is the only place to
    fix it, nothing else in the pipeline needs to change.

    Scoped to one endpoint (season standings) on purpose, that's enough
    for two solid facts, position and wins, reliably. Pulling in race
    by race results or qualifying data for a richer fact set is the
    natural next step once this is confirmed working.
    """

    _BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def topics(self) -> List[str]:
        raise NotImplementedError(
            "Live topics are open ended driverId-season pairs, not a fixed "
            "list, there's nothing meaningful to enumerate here. Pass a "
            "topic like 'verstappen-2024' straight to get_facts."
        )

    def get_facts(self, topic: str, limit: int = 5) -> List[Fact]:
        import json
        import urllib.request
        import urllib.error

        driver_id, sep, season = topic.rpartition("-")
        if not sep or not season.isdigit():
            raise ValueError(
                f"Expected topic as '<driverId>-<season>', e.g. 'verstappen-2024', got '{topic}'"
            )

        url = f"{self._BASE_URL}/{season}/drivers/{driver_id}/driverStandings.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Could not reach the F1 API at {url}: {e}. If this is running "
                "in a sandboxed or restricted network environment, that's the "
                "known limitation, see the class docstring."
            ) from e

        return self._parse_standings(data, limit)

    @staticmethod
    def _parse_standings(data: dict, limit: int) -> List[Fact]:
        lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if not lists or not lists[0].get("DriverStandings"):
            raise ValueError(
                "No standings data in the API response, double check the "
                "driverId and season are correct and that driver actually "
                "raced that season."
            )

        standing = lists[0]["DriverStandings"][0]
        season = lists[0]["season"]
        driver = standing["Driver"]
        name = f"{driver['givenName']} {driver['familyName']}"
        constructor = standing["Constructors"][0]["name"]
        wins = standing["wins"]

        facts = [
            Fact(
                text=f"{name} finished the {season} season in P{standing['position']} "
                     f"with {standing['points']} points.",
                year=int(season), category="record",
            ),
            Fact(
                text=f"{name} won {wins} race{'s' if wins != '1' else ''} in {season}, "
                     f"driving for {constructor}.",
                year=int(season), category="history",
            ),
        ]
        return facts[:limit]
