"""Derive a profile's class scope and predicate weights from its entity-branch priorities.

A profile no longer lists 165 hand-written predicate weights. The user (or the default profile
file) sets one priority per *entity branch* of the Biolink class hierarchy, on three levels:

    0    out of scope (implicit: an absent branch is 0)
    0.5  secondary interest
    1    primary interest

An entity branch is a direct child of `named thing`, except `biological entity`, whose own
children are the branches (gene, disease or phenotypic feature, biological process or
activity, anatomical entity through organismal entity, ...): a single switch for the whole
of biology would not discriminate anything. Every descendant class and every mixin inherits the
priority of its branch (a mixin takes the maximum over its member classes, since it groups
classes of several branches). This inheritance is the "strata config" of the job: the class
weights are written next to the derived predicate weights for traceability.

From the class weights the predicate weights follow automatically:

1. A predicate is *in scope* when both its domain and its range contain at least one class of
   non-zero weight (a descendant or a mixin member counts; an unconstrained domain or range such
   as `named thing` always counts, with weight 1). Its entity priority e(p) is
   max(w(domain), w(range)), where w(X) is the highest class weight under X.
2. Predicates are grouped by *predicate branch*, the level-2 node of the predicate hierarchy
   (`affects`, `interacts with`, `associated with`, `located in`, ...). Inside a branch the
   priority is distributed over the predicates present in the filtered knowledge graph,
   proportionally to e(p) * exp(alpha * depth(p)), and normalised so that the branch sums to
   its priority W_B = max e(p) over its members (0.5 or 1). Branches are independent: there is
   no normalisation across branches.
3. Representation predicates (gene <-> transcript <-> protein, has part / part of between a
   gene or protein identity and a drug-target identity) are changes of identity, not
   mechanistic steps: they stay at weight 1 and are left out of the branch normalisation.

The derivation is cached by fingerprint (Biolink version, branch weights, alpha, directionality,
knowledge-graph predicate table) in a `derived/` folder next to the profile file, so default
profiles are computed once and a user profile built in the console is computed on the fly and
shipped with the job outputs.

Directionality is a profile parameter, applied by the context-graph stage: with mode "on" the
expansion follows subject -> object edges at full weight and object -> subject edges at
`inverse_factor` (symmetric Biolink predicates such as `interacts with` are not penalised);
with mode "off" every edge is traversed both ways at full weight, as before.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BRANCH_ROOT = "named thing"
SPLIT_BRANCHES = ("biological entity",)      # exposed through their children, never weighted themselves
GENERIC_CLASSES = {"named thing", "entity", "thing with taxon", "physical essence or occurrent", "physical essence", "occurrent"}
ALLOWED_LEVELS = (0.0, 0.5, 1.0)
DEFAULT_ALPHA = 0.7
DEFAULT_INVERSE_FACTOR = 0.5

# Changes of identity of one biological object (gene, transcript, protein, drug target), not
# mechanistic steps: fixed weight 1, outside the branch normalisation. has_part / part_of are
# only representation edges between a gene or protein identity and a drug-target identity,
# which the context-graph stage decides per edge from the CURIE prefixes; as predicates they
# keep their computed weight.
REPRESENTATION_PREDICATES = (
    "biolink:gene_product_of", "biolink:has_gene_product", "biolink:transcribed_to",
    "biolink:transcribed_from", "biolink:translates_to", "biolink:translation_of",
)
# Identity and equivalence predicates: never traversed (the filtered KG2c drops them, and the
# pipeline does no identifier conflation), so never weighted.
EQUIVALENCE_PREDICATES = frozenset({
    "biolink:same_as", "biolink:exact_match", "biolink:close_match", "biolink:broad_match", "biolink:narrow_match",
})
# Roots of the predicate hierarchy (depth < 2): they carry no information and, being alone in
# their "branch", would otherwise receive the full priority. 
ROOT_PREDICATE_DEPTH = 2


# ------------------------------------------------------------------------------------------
# Biolink model access (data/biolink_strata.json, built by scripts/build_biolink_strata.py)
# ------------------------------------------------------------------------------------------

def curie(name: str) -> str:
    """'small molecule' -> 'biolink:small_molecule' (the form used in biolink_strata.json)."""
    return "biolink:" + name.strip().replace(" ", "_")


def camel_curie(name: str) -> str:
    """'small molecule' -> 'biolink:SmallMolecule' (the category form stored in RTX-KG2c nodes)."""
    return "biolink:" + "".join(w[:1].upper() + w[1:] for w in name.split())


def plain(pred: str) -> str:
    """'biolink:biolink_treats' -> 'treats' (KG2c carries a few malformed doubled prefixes)."""
    return pred.replace("biolink:", "").replace("biolink_", "").replace("_", " ").strip()


@dataclass
class BiolinkModel:
    """The parts of the derived strata file needed here, with a few indexes."""

    version: str
    classes: dict[str, dict]                     # name -> {is_a, ancestors, children, mixins, curie, ...}
    predicates: dict[str, dict]                  # name -> {curie, is_a, depth, domain, range, symmetric?, ...}
    mixins: dict[str, list[str]]                 # mixin -> direct member classes
    mixin_members: dict[str, set[str]] = field(default_factory=dict)   # mixin -> classes carrying it, ancestors included
    descendants: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        desc: dict[str, set[str]] = defaultdict(set)
        for c, v in self.classes.items():
            for a in v.get("ancestors") or []:
                desc[a].add(c)
        self.descendants = dict(desc)
        members: dict[str, set[str]] = defaultdict(set)
        for c, v in self.classes.items():
            for k in [c] + list(v.get("ancestors") or []):
                for m in (self.classes.get(k) or {}).get("mixins") or []:
                    members[m].add(c)
        for m, direct in self.mixins.items():
            members[m].update(direct)
        self.mixin_members = dict(members)

    @classmethod
    def load(cls, path: str | Path) -> "BiolinkModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(version=str(raw.get("biolink_version", "")), classes=raw["classes"], predicates=raw["predicates"],
                   mixins={m: list(v) for m, v in raw.get("mixins", {}).items()})

    # -- entity branches ---------------------------------------------------------------------
    def branches(self) -> list[str]:
        """Entity branches in a stable order: children of the root, split roots replaced by their children."""
        out: list[str] = []
        for c in sorted(self.classes.get(BRANCH_ROOT, {}).get("children") or []):
            if c in SPLIT_BRANCHES:
                out.extend(sorted(self.classes.get(c, {}).get("children") or []))
            else:
                out.append(c)
        return out

    def branch_of(self, class_name: str) -> str | None:
        """The entity branch a class belongs to (itself or its closest ancestor that is a branch)."""
        branches = set(self.branches())
        if class_name in branches:
            return class_name
        for a in (self.classes.get(class_name) or {}).get("ancestors") or []:
            if a in branches:
                return a
        return None

    def branch_info(self) -> list[dict]:
        """Branches with their description and size, for the console."""
        out = []
        for b in self.branches():
            v = self.classes.get(b, {})
            group = v.get("is_a") if v.get("is_a") in SPLIT_BRANCHES else None
            out.append({"name": b, "curie": camel_curie(b), "group": group, "description": v.get("description", ""),
                        "n_classes": 1 + len(self.descendants.get(b, ())),
                        "examples": sorted(self.descendants.get(b, ()))[:6]})
        return out

    # -- predicate hierarchy -----------------------------------------------------------------
    def predicate_ancestors(self, name: str) -> list[str]:
        out, cur = [], (self.predicates.get(name) or {}).get("is_a")
        while cur:
            out.append(cur)
            cur = (self.predicates.get(cur) or {}).get("is_a")
        return out

    def predicate_branch(self, name: str) -> str:
        """Level-2 node of the predicate hierarchy (child of `related to at instance/concept level`)."""
        chain = list(reversed(self.predicate_ancestors(name))) + [name]
        return chain[2] if len(chain) > 2 else chain[-1]

    def predicate_depth(self, name: str) -> int:
        d = (self.predicates.get(name) or {}).get("depth")
        return int(d) if d is not None else len(self.predicate_ancestors(name))

    def symmetric_predicates(self) -> list[str]:
        """CURIEs of the predicates flagged symmetric in the model (or their names when the flag is missing)."""
        return sorted(v["curie"] for v in self.predicates.values() if v.get("symmetric"))


# ------------------------------------------------------------------------------------------
# Class weights (the per-job strata config)
# ------------------------------------------------------------------------------------------

def normalise_branch_weights(raw: dict[str, Any], model: BiolinkModel) -> dict[str, float]:
    """Validate user branch weights: known branches only, values snapped to 0 / 0.5 / 1."""
    branches = set(model.branches())
    out: dict[str, float] = {}
    for name, value in (raw or {}).items():
        key = name if name in branches else plain(name)
        if key not in branches:
            logger.warning("branch %r is not an entity branch of Biolink %s: ignored", name, model.version)
            continue
        try:
            w = float(value)
        except (TypeError, ValueError):
            w = 0.0
        w = min(ALLOWED_LEVELS, key=lambda lvl: abs(lvl - w))
        out[key] = w
    return out


def class_weights(branch_weights: dict[str, float], model: BiolinkModel) -> dict[str, float]:
    """Every entity class and mixin -> weight inherited from its branch.

    Classes above the branches (the root and the split roots) take the maximum of the branches
    below them, so a node typed `biolink:BiologicalEntity` stays in scope as long as one
    biological branch is. Mixins take the maximum over their member classes.
    """
    out: dict[str, float] = {}
    for c in model.classes:
        b = model.branch_of(c)
        if b is not None:
            out[c] = float(branch_weights.get(b, 0.0))
        else:  # root or split root
            below = [branch_weights.get(d, 0.0) for d in model.descendants.get(c, ()) if d in branch_weights]
            out[c] = float(max(below)) if below else 0.0
    for m, members in model.mixin_members.items():
        out[m] = max((out.get(c, 0.0) for c in members), default=0.0)
    return out


def scope_weight(name: str | None, cw: dict[str, float], model: BiolinkModel) -> float:
    """Highest class weight under a domain or range name; unconstrained names count as 1."""
    if not name or name in GENERIC_CLASSES:
        return 1.0
    if name in model.classes:
        return max([cw.get(name, 0.0)] + [cw.get(d, 0.0) for d in model.descendants.get(name, ())])
    if name in model.mixin_members or name in model.mixins:
        return max((cw.get(c, 0.0) for c in model.mixin_members.get(name, ())), default=cw.get(name, 0.0))
    return 1.0  # unknown to this Biolink version: do not silently exclude


# ------------------------------------------------------------------------------------------
# Predicate weights
# ------------------------------------------------------------------------------------------

def kg_edge_counts(kg_dir: str | Path | None) -> dict[str, int]:
    """Edges kept per predicate in the filtered KG2c (stats.json, else predicates.parquet, else none)."""
    if not kg_dir:
        return {}
    kg_dir = Path(kg_dir)
    stats = kg_dir / "stats.json"
    if stats.is_file():
        try:
            counts = json.loads(stats.read_text(encoding="utf-8")).get("edges", {}).get("edges_kept_by_predicate", {})
            if counts:
                return {k: int(v) for k, v in counts.items()}
        except ValueError:
            pass
    parquet = kg_dir / "predicates.parquet"
    if parquet.is_file():
        try:
            import duckdb

            sql = str(parquet).replace("\\", "/").replace("'", "''")
            return {r[0]: int(r[1]) for r in duckdb.connect().execute(f"SELECT predicate, n_edges_kept FROM read_parquet('{sql}')").fetchall()}
        except Exception as e:  # duckdb missing or unreadable file: every predicate counts as present
            logger.warning("could not read %s (%s): every in-scope predicate is treated as present", parquet, e)
    return {}


@dataclass
class Derived:
    class_weights: dict[str, float]          # class or mixin name -> weight
    category_weights: dict[str, float]       # KG2c category CURIE (biolink:SmallMolecule) -> weight
    predicates: dict[str, float]             # predicate CURIE -> weight
    branch_priorities: dict[str, float]      # predicate branch -> W_B
    symmetric_predicates: list[str]
    representation_predicates: list[str]
    review: list[dict]                       # one row per predicate examined
    absent_in_scope: list[str]               # in scope, present in KG2c, but filtered out of the Parquet build
    fingerprint: str = ""


def predicate_priority(w_domain: float, w_range: float, rule: str = "max") -> float:
    """Entity priority of a predicate from the weights of its domain and range (0 = out of scope)."""
    if w_domain <= 0 or w_range <= 0:
        return 0.0
    return min(w_domain, w_range) if rule == "min" else max(w_domain, w_range)


def derive(branch_weights: dict[str, float], model: BiolinkModel, alpha: float = DEFAULT_ALPHA,
           edge_counts: dict[str, int] | None = None, min_weight: float = 0.0, scope_rule: str = "max") -> Derived:
    """Compute class and predicate weights from entity-branch priorities (see module docstring).

    `edge_counts` are the edges kept per predicate in the current KG2c build (a predicate without
    kept edges receives no weight, so the branch priority is not diluted over absent predicates:
    absent because this profile's Biolink version is ahead of this KG2c build, or because
    scripts/build_kg2c_parquet.py's structural filters removed the predicate).
    """
    cw = class_weights(branch_weights, model)
    edges = edge_counts or {}
    rows: list[dict] = []
    members: dict[str, list[tuple[str, float, int]]] = defaultdict(list)   # branch -> (curie, e, depth)
    absent: list[str] = []
    for name, info in model.predicates.items():
        pred = info["curie"]
        dom, rng = info.get("domain"), info.get("range")
        wd, wr = scope_weight(dom, cw, model), scope_weight(rng, cw, model)
        e = predicate_priority(wd, wr, scope_rule)
        depth = model.predicate_depth(name)
        branch = model.predicate_branch(name)
        n_edges = edges.get(pred, 0)
        row = {"predicate": pred, "branch": branch, "depth": depth, "domain": dom, "range": rng,
               "w_domain": wd, "w_range": wr, "priority": e, "edges": n_edges,
               "weight": None, "note": ""}
        rows.append(row)
        if pred in REPRESENTATION_PREDICATES:
            row["note"] = "representation predicate: fixed weight 1"
            continue
        if pred in EQUIVALENCE_PREDICATES:
            row["note"] = "equivalence predicate: never traversed"
            continue
        if not info.get("kg2c_only") and depth < ROOT_PREDICATE_DEPTH:
            row["note"] = "root of the predicate hierarchy: never weighted"
            continue
        if e <= 0:
            row["note"] = "out of scope"
            continue
        if edges and n_edges == 0:
            row["note"] = ("in scope but no edge in the current KG2c build (absent from this RTX-KG2c version, "
                           "or removed by a structural filter: equivalence, too generic, or an excluded source)")
            absent.append(pred)
            continue
        members[branch].append((pred, e, depth))

    out: dict[str, float] = {}
    priorities: dict[str, float] = {}
    for branch, ms in members.items():
        w_branch = max(e for _, e, _ in ms)
        raws = {pred: e * math.exp(alpha * depth) for pred, e, depth in ms}
        total = sum(raws.values())
        priorities[branch] = w_branch
        for pred, r in raws.items():
            out[pred] = w_branch * r / total
    for pred in REPRESENTATION_PREDICATES:
        out[pred] = 1.0
    out = {k: round(v, 6) for k, v in out.items() if v >= min_weight}
    for row in rows:
        row["weight"] = out.get(row["predicate"])
        if row["weight"] is None and row["note"] == "" and row["priority"] > 0:
            row["note"] = "below min_weight"

    category = {}
    for c, w in cw.items():
        if c in model.classes:
            category[camel_curie(c)] = w
    return Derived(
        class_weights=cw,
        category_weights=category,
        predicates=dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))),
        branch_priorities=dict(sorted(priorities.items(), key=lambda kv: (-kv[1], kv[0]))),
        symmetric_predicates=model.symmetric_predicates(),
        representation_predicates=list(REPRESENTATION_PREDICATES),
        review=sorted(rows, key=lambda r: (r["branch"], -(r["weight"] or 0), r["predicate"])),
        absent_in_scope=sorted(absent),
    )


# ------------------------------------------------------------------------------------------
# Profile-level entry point with cache and trace files
# ------------------------------------------------------------------------------------------

def directionality_of(raw: dict) -> dict:
    """Normalised directionality block: {"mode": "on"|"off", "inverse_factor": float}."""
    d = raw.get("directionality", "on")
    if isinstance(d, str):
        d = {"mode": d}
    d = dict(d or {})
    mode = str(d.get("mode", "on")).lower()
    mode = "on" if mode in ("on", "true", "1", "directed") else "off"
    try:
        factor = float(d.get("inverse_factor", DEFAULT_INVERSE_FACTOR))
    except (TypeError, ValueError):
        factor = DEFAULT_INVERSE_FACTOR
    return {"mode": mode, "inverse_factor": max(0.0, min(1.0, factor))}


def _fingerprint(model: BiolinkModel, branch_weights: dict[str, float], alpha: float, min_weight: float,
                 directionality: dict, edge_counts: dict[str, int], scope_rule: str) -> str:
    payload = {"biolink": model.version, "branches": sorted(branch_weights.items()), "alpha": alpha, "min_weight": min_weight,
               "directionality": directionality, "representation": REPRESENTATION_PREDICATES, "scope_rule": scope_rule,
               "edges": sorted(edge_counts.items()), "schema": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def derived_paths(profile_path: Path) -> tuple[Path, Path]:
    """Cache files of a profile: `<stem>.derived.json` and `<stem>.review.md`, under `derived/`."""
    d = profile_path.parent / "derived"
    return d / f"{profile_path.stem}.derived.json", d / f"{profile_path.stem}.review.md"


def review_markdown(profile_name: str, derived: Derived, branch_weights: dict[str, float], alpha: float, directionality: dict) -> str:
    lines = [f"# Derived weights: {profile_name}", "",
             f"Entity branch priorities: `{json.dumps(branch_weights, ensure_ascii=False)}`  ",
             f"alpha = {alpha}; directionality = `{json.dumps(directionality)}`  ",
             f"Predicate branch priorities (sum of the branch): `{json.dumps(derived.branch_priorities, ensure_ascii=False)}`", ""]
    if derived.absent_in_scope:
        lines += ["**In scope per the branch priorities, but no edge in the current KG2c build** (absent from this "
                  "RTX-KG2c version, or removed by a structural filter of `scripts/build_kg2c_parquet.py` -- "
                  "equivalence, too generic, or an excluded source): " + ", ".join(derived.absent_in_scope), ""]
    lines += ["| predicate | branch | depth | domain | range | priority | edges kept | weight | note |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in derived.review:
        lines.append(f"| {r['predicate']} | {r['branch']} | {r['depth']} | {r['domain'] or '-'} | {r['range'] or '-'} | "
                     f"{r['priority']} | {r['edges']} | {r['weight'] if r['weight'] is not None else '-'} | {r['note']} |")
    lines += ["", "## Class weights (strata config of the job)", "", "| class | weight |", "|---|---|"]
    for c, w in sorted(derived.class_weights.items(), key=lambda kv: (-kv[1], kv[0])):
        if w > 0:
            lines.append(f"| {c} | {w} |")
    return "\n".join(lines) + "\n"


def derive_for_profile(raw: dict, profile_path: Path, strata_path: Path, kg_dir: str | Path | None,
                       write: bool = True) -> dict:
    """Derived weights of a profile file, from cache when its fingerprint is unchanged.

    Returns the derived block as a plain dict (the content of `<stem>.derived.json`). When `write`
    is true the JSON and the review table are written under `derived/` next to the profile: the
    default profiles are computed once (until Biolink, the graph or the priorities change) and a
    console profile is computed on the fly and shipped with the job outputs.
    """
    model = BiolinkModel.load(strata_path)
    bw = normalise_branch_weights(raw.get("branch_weights") or {}, model)
    weighting = raw.get("weighting") or {}
    alpha = float(weighting.get("alpha", DEFAULT_ALPHA))
    min_weight = float(weighting.get("min_weight", 0.0))
    scope_rule = "min" if str(weighting.get("scope_rule", "max")).lower() == "min" else "max"
    directionality = directionality_of(raw)
    edge_counts = kg_edge_counts(kg_dir)
    fp = _fingerprint(model, bw, alpha, min_weight, directionality, edge_counts, scope_rule)

    json_path, md_path = derived_paths(profile_path)
    if json_path.is_file():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fp:
                return cached
        except ValueError:
            pass

    d = derive(bw, model, alpha=alpha, edge_counts=edge_counts, min_weight=min_weight, scope_rule=scope_rule)
    d.fingerprint = fp
    payload = {
        "profile": raw.get("name", profile_path.stem), "source_profile": profile_path.name, "fingerprint": fp,
        "biolink_version": model.version, "branch_weights": bw, "alpha": alpha, "min_weight": min_weight, "scope_rule": scope_rule,
        "directionality": directionality, "branch_priorities": d.branch_priorities,
        "class_weights": d.class_weights, "category_weights": d.category_weights,
        "predicates": d.predicates, "symmetric_predicates": d.symmetric_predicates,
        "representation_predicates": d.representation_predicates, "absent_in_scope": d.absent_in_scope,
        "n_predicates": len(d.predicates), "n_classes_in_scope": sum(1 for w in d.category_weights.values() if w > 0),
    }
    if write:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            md_path.write_text(review_markdown(payload["profile"], d, bw, alpha, directionality), encoding="utf-8")
            logger.info("derived weights for %s: %d predicates, %d branches, %d classes in scope -> %s",
                        payload["profile"], len(d.predicates), len(d.branch_priorities), payload["n_classes_in_scope"], json_path)
        except OSError as e:
            logger.warning("could not write the derived weights of %s: %s", profile_path, e)
    return payload


def default_branch_weights_for_stratum(stratum: str, strata_raw: dict, model: BiolinkModel,
                                       angle_weight: float = 1.0) -> dict[str, float]:
    """Branch priorities of a default profile from the human-defined strata anchors.

    1 when a class of the branch is anchored in the stratum (directly or by inheritance),
    `angle_weight` (1 by default: default profiles weight their whole stratum at 1) when the
    branch only enters the stratum under a reading angle (a shared class such as disease for
    the biochemical stratum), 0 otherwise. Used once to seed the default profile files; the
    numbers are then reviewed and stored in the profile, not recomputed.
    """
    out: dict[str, float] = {}
    for b in model.branches():
        vias = set()
        for c in [b] + sorted(model.descendants.get(b, ())):
            status = ((strata_raw["classes"].get(c) or {}).get("strata") or {}).get(stratum)
            if status:
                vias.add(status.get("via"))
        out[b] = 1.0 if vias & {"anchor", "inherited"} else (angle_weight if vias else 0.0)
    return out
