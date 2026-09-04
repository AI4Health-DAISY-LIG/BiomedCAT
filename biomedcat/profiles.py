"""User preference profiles for the context-graph stage.

A profile tells BiomedCAT how much each kind of knowledge-graph edge matters to the user. It
is a JSON file with up to four blocks; a flat {predicate: weight} file (the legacy format
produced by MEDiQ's probabilities_maps.py) is accepted and read as the `predicates` block.

    {
      "name": "clinical mechanisms",
      "predicates": {"biolink:has_phenotype": 0.7, ...},            # required
      "sources": {"infores:diseases": 0.2, ...},                     # default 1.0
      "knowledge_levels": {"prediction": 0.3},                       # default 1.0
      "prefix_restricted_predicates": {                              # optional
          "biolink:subclass_of": ["MONDO:", "HP:", "DOID:", "EFO:", "Orphanet:"]
      },
      "entity_scope": ["gene", "protein", "disease", ...],          # optional, drives the reading prompt
      "reading_focus": "free text appended to the reading prompt"   # optional
    }

The weight of an edge is the product predicate x source x knowledge level. A predicate absent
from `predicates` has weight 0 and is never traversed. A prefix-restricted predicate is kept
only when both endpoints carry one of the listed CURIE prefixes (e.g. subclass_of between
diseases, not between chemicals or taxa).

`entity_scope` and `reading_focus` keep the domain knowledge out of the code: the slide-reading
prompt in biomedcat.prompts is domain-neutral and takes the list of entity kinds to name from
the active profile. Profiles live in data/, which is not distributed with the code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Profile:
    name: str
    predicates: dict[str, float]
    sources: dict[str, float] = field(default_factory=dict)
    knowledge_levels: dict[str, float] = field(default_factory=dict)
    prefix_restricted_predicates: dict[str, list[str]] = field(default_factory=dict)
    entity_scope: list[str] = field(default_factory=list)
    reading_focus: str = ""
    stratum: str = ""
    path: str = ""

    def predicate_weight(self, predicate: str) -> float:
        return float(self.predicates.get(predicate, 0.0))

    def source_weight(self, source: str) -> float:
        return float(self.sources.get(source, 1.0))

    def knowledge_level_weight(self, level: str) -> float:
        return float(self.knowledge_levels.get(level, 1.0))

    def edge_weight(self, predicate: str, source: str, level: str) -> float:
        return self.predicate_weight(predicate) * self.source_weight(source) * self.knowledge_level_weight(level)

    def endpoints_allowed(self, predicate: str, subject: str, obj: str) -> bool:
        prefixes = self.prefix_restricted_predicates.get(predicate)
        if not prefixes:
            return True
        return any(subject.startswith(p) for p in prefixes) and any(obj.startswith(p) for p in prefixes)


def load_profile(path: str | Path) -> Profile:
    """Read a profile file; a flat {predicate: weight} JSON is accepted as legacy format."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "predicates" not in raw:
        # Legacy flat format: every key is a predicate.
        return Profile(name=path.stem, predicates={k: float(v) for k, v in raw.items()}, path=str(path))
    profile = Profile(
        name=raw.get("name", path.stem),
        predicates={k: float(v) for k, v in raw["predicates"].items()},
        sources={k: float(v) for k, v in raw.get("sources", {}).items()},
        knowledge_levels={k: float(v) for k, v in raw.get("knowledge_levels", {}).items()},
        prefix_restricted_predicates={k: list(v) for k, v in raw.get("prefix_restricted_predicates", {}).items()},
        entity_scope=[str(x) for x in raw.get("entity_scope", [])],
        reading_focus=str(raw.get("reading_focus", "") or ""),
        stratum=str(raw.get("stratum", "") or ""),
        path=str(path),
    )
    # A profile tied to a stratum takes its reading scope from the derived strata file unless
    # the profile overrides it explicitly.
    if profile.stratum and not (profile.entity_scope and profile.reading_focus):
        strata_path = path.parent.parent / "biolink_strata.json"
        if strata_path.is_file():
            strata = json.loads(strata_path.read_text(encoding="utf-8")).get("strata", {})
            derived = strata.get(profile.stratum, {})
            if not profile.entity_scope:
                profile.entity_scope = list(derived.get("entity_scope", []))
            if not profile.reading_focus:
                profile.reading_focus = str(derived.get("reading_focus", ""))
    return profile


def register_in_duckdb(con, profile: Profile) -> None:
    """Create the lookup tables prof_pred, prof_src, prof_kl so SQL can compute edge weights.

    Only listed entries are stored; SQL callers must COALESCE missing sources and levels to 1.0
    and treat a missing predicate as 0 (LEFT JOIN + COALESCE(..., 0)).
    """
    for table, col, rows in (
        ("prof_pred", "predicate", list(profile.predicates.items())),
        ("prof_src", "source", list(profile.sources.items())),
        ("prof_kl", "level", list(profile.knowledge_levels.items())),
    ):
        con.execute(f"CREATE OR REPLACE TABLE {table} ({col} VARCHAR, w DOUBLE)")
        if rows:  # DuckDB's executemany rejects an empty parameter list
            con.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
