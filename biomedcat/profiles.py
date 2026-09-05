"""User preference profiles for the context-graph stage.

A profile tells BiomedCAT which parts of the knowledge graph matter to the user. Since the
entity-branch scheme (biomedcat.weights) a profile is a small JSON file:

    {
      "name": "biochemical actions",
      "stratum": "biochemical",                                      # optional, reading scope
      "branch_weights": {"gene": 1, "chemical entity": 1, "disease or phenotypic feature": 0.5, ...},
      "directionality": {"mode": "on", "inverse_factor": 0.5},      # default: on, 0.5
      "weighting": {"alpha": 0.7, "scope_rule": "max"},             # optional
      "sources": {"infores:diseases": 0.2, ...},                     # default 1.0
      "knowledge_levels": {"prediction": 0.3},                       # default 1.0
      "prefix_restricted_predicates": {"biolink:subclass_of": ["MONDO:", "HP:"]},   # optional
      "entity_scope": [...], "reading_focus": "..."                  # optional, reading prompt
    }

`branch_weights` gives one priority (0, 0.5 or 1) per entity branch of the Biolink class
hierarchy; the class scope and the predicate weights are derived from it (and cached under
`derived/` next to the file) by biomedcat.weights. A file that carries an explicit
`predicates` block (the uniform baseline, legacy hand-written profiles, a flat
{predicate: weight} file from MEDiQ) uses it as is; `branch_weights` then only define the
class scope.

The weight of an edge is the product predicate x source x knowledge level. A predicate absent
from `predicates` has weight 0 and is never traversed. A prefix-restricted predicate is kept
only when both endpoints carry one of the listed CURIE prefixes (e.g. subclass_of between
diseases, not between chemicals or taxa). A node whose Biolink category has class weight 0 is
out of scope: never expanded, never kept. Directionality is applied by the context-graph
stage (see biomedcat.weights for the semantics).

`entity_scope` and `reading_focus` keep the domain knowledge out of the code: the slide-reading
prompt in biomedcat.prompts is domain-neutral and takes the list of entity kinds to name from
the active profile. Profiles live in data/, which is not distributed with the code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


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
    # Entity-branch scheme (biomedcat.weights); empty for legacy flat profiles.
    branch_weights: dict[str, float] = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)   # biolink:SmallMolecule -> 0 / 0.5 / 1
    directionality: dict = field(default_factory=lambda: {"mode": "on", "inverse_factor": 0.5})
    symmetric_predicates: list[str] = field(default_factory=list)
    derived_path: str = ""

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

    def category_weight(self, category: str) -> float:
        """Scope weight of a KG2c node category; categories unknown to the model stay in scope."""
        if not self.category_weights:
            return 1.0
        return float(self.category_weights.get(category, 1.0))

    @property
    def directed(self) -> bool:
        return str(self.directionality.get("mode", "on")) == "on"

    @property
    def inverse_factor(self) -> float:
        return float(self.directionality.get("inverse_factor", 0.5)) if self.directed else 1.0


def _strata_path(profile_path: Path) -> Path:
    """data/biolink_strata.json, whether the profile lives in data/profiles or in a job directory."""
    from biomedcat.config import settings

    candidate = profile_path.parent.parent / "biolink_strata.json"
    if candidate.is_file():
        return candidate
    return Path(settings.internal_data_path) / "biolink_strata.json"


def load_profile(path: str | Path, derive: bool = True, presence: str = "filtered") -> Profile:
    """Read a profile file; derive its weights from `branch_weights` when it has no `predicates` block.

    A flat {predicate: weight} JSON is accepted as legacy format. `derive=False` skips the
    derivation (the profile then has no predicate weights), for listings that only need names.
    `presence="observed"` derives against every predicate of the raw KG2c instead of the current
    Parquet build (used when building that Parquet, see biomedcat.weights.derive_for_profile).
    """
    from biomedcat import weights as weights_mod

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "predicates" not in raw and "branch_weights" not in raw:
        # Legacy flat format: every key is a predicate; undirected, as it was traversed before.
        return Profile(name=path.stem, predicates={k: float(v) for k, v in raw.items()}, path=str(path),
                       directionality={"mode": "off", "inverse_factor": 1.0})

    profile = Profile(
        name=raw.get("name", path.stem),
        predicates={k: float(v) for k, v in (raw.get("predicates") or {}).items()},
        sources={k: float(v) for k, v in raw.get("sources", {}).items()},
        knowledge_levels={k: float(v) for k, v in raw.get("knowledge_levels", {}).items()},
        prefix_restricted_predicates={k: list(v) for k, v in raw.get("prefix_restricted_predicates", {}).items()},
        entity_scope=[str(x) for x in raw.get("entity_scope", [])],
        reading_focus=str(raw.get("reading_focus", "") or ""),
        stratum=str(raw.get("stratum", "") or ""),
        path=str(path),
        directionality=weights_mod.directionality_of(raw),
    )

    if raw.get("branch_weights") and derive:
        from biomedcat.config import settings

        strata_path = _strata_path(path)
        if strata_path.is_file():
            derived = weights_mod.derive_for_profile(raw, path, strata_path, settings.kg2c_dir, presence=presence)
            profile.branch_weights = dict(derived.get("branch_weights", {}))
            profile.category_weights = dict(derived.get("category_weights", {}))
            profile.symmetric_predicates = list(derived.get("symmetric_predicates", []))
            profile.derived_path = str(weights_mod.derived_paths(path, presence)[0])
            if not profile.predicates:
                profile.predicates = {k: float(v) for k, v in derived.get("predicates", {}).items()}
            # A console profile has no stratum: its reading scope is the list of weighted branches.
            if not profile.stratum and not profile.entity_scope and "predicates" not in raw:
                profile.entity_scope = [b for b, w in profile.branch_weights.items() if w > 0]
        else:
            logger.warning("%s: no biolink_strata.json found, branch weights ignored", path)
    elif raw.get("branch_weights"):
        profile.branch_weights = {k: float(v) for k, v in raw["branch_weights"].items()}

    strata_path = _strata_path(path)
    if strata_path.is_file():
        # Symmetric predicates are needed by every directed traversal, whatever the profile format.
        if not profile.symmetric_predicates and profile.directed:
            profile.symmetric_predicates = _symmetric_predicates(str(strata_path))
        # A profile tied to a stratum takes its reading scope from the derived strata file unless
        # the profile overrides it explicitly.
        if profile.stratum and not (profile.entity_scope and profile.reading_focus):
            strata = json.loads(strata_path.read_text(encoding="utf-8")).get("strata", {})
            derived_stratum = strata.get(profile.stratum, {})
            if not profile.entity_scope:
                profile.entity_scope = list(derived_stratum.get("entity_scope", []))
            if not profile.reading_focus:
                profile.reading_focus = str(derived_stratum.get("reading_focus", ""))
    return profile


@lru_cache(maxsize=4)
def _symmetric_predicates(strata_path: str) -> list[str]:
    from biomedcat import weights as weights_mod

    try:
        return weights_mod.BiolinkModel.load(strata_path).symmetric_predicates()
    except (OSError, ValueError, KeyError):
        return []


def register_in_duckdb(con, profile: Profile) -> None:
    """Create the lookup tables prof_pred, prof_src, prof_kl, prof_cat, prof_sym so SQL can weight edges.

    Only listed entries are stored; SQL callers must COALESCE missing sources and levels to 1.0
    and treat a missing predicate as 0 (LEFT JOIN + COALESCE(..., 0)). prof_cat holds the class
    scope (category -> weight; a category absent from it is in scope) and prof_sym the symmetric
    predicates, which the directionality penalty leaves alone.
    """
    for table, col, rows in (
        ("prof_pred", "predicate", list(profile.predicates.items())),
        ("prof_src", "source", list(profile.sources.items())),
        ("prof_kl", "level", list(profile.knowledge_levels.items())),
        ("prof_cat", "category", list(profile.category_weights.items())),
    ):
        con.execute(f"CREATE OR REPLACE TABLE {table} ({col} VARCHAR, w DOUBLE)")
        if rows:  # DuckDB's executemany rejects an empty parameter list
            con.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
    con.execute("CREATE OR REPLACE TABLE prof_sym (predicate VARCHAR)")
    if profile.symmetric_predicates:
        con.executemany("INSERT INTO prof_sym VALUES (?)", [(p,) for p in profile.symmetric_predicates])
