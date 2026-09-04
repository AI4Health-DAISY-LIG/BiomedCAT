#!/usr/bin/env python
"""Derive granularity strata and scopes of the Biolink model from biolink-model.yaml.

Input: the Biolink YAML (data/biolink-model.yaml) and the human-defined anchors of each stratum
(data/profiles/strata_config.json). Output: data/biolink_strata.json with

  classes    for every entity class: is_a parent, ancestors (vertical links), depth, mixins,
             children, and its status in each stratum: "core" (anchored in this stratum only),
             "shared" (anchored in several strata, with the reading angle of each), or absent;
  mixins     mixin name -> member classes: the horizontal groupings that put classes of
             different branches at the same level (e.g. "gene or gene product");
  predicates every predicate (slot descending from "related to"): parent, domain, range, and
             the strata in which both its domain and its range are in scope, i.e. the links
             between entity classes that a stratum can traverse;
  strata     per stratum: core classes, shared classes with angles, predicates in scope, and
             the entity kinds to name when reading a document (used by the reading prompt).

A class is in a stratum when it, one of its is_a ancestors, or one of its mixins is an anchor
of that stratum. Mixins are followed at the same level, not upward: a class inherits the
strata of its own mixins and of its ancestors' mixins, but not of unrelated classes sharing
the mixin. Predicates inherit domain/range from their is_a parent when unspecified.

Also writes data/profiles/uniform.json: every predicate at weight 1, no source or level
weight, the no-profile baseline used to demonstrate the profiles' effect.

Usage (from the BiomedCAT root):
  uv run python scripts/build_biolink_strata.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Fallback when no observed-predicate file is given: KG2c 2.10.1 predicates with a malformed prefix.
KG2C_EXTRA_PREDICATES = ["biolink:biolink_in_clinical_trials_for", "biolink:biolink_mentioned_in_trials_for", "biolink:biolink_treats"]


def curie(name: str) -> str:
    return "biolink:" + name.strip().replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", default="data/biolink-model.yaml")
    parser.add_argument("--config", default="data/profiles/strata_config.json")
    parser.add_argument("--out", default="data/biolink_strata.json")
    parser.add_argument("--uniform-profile", default="data/profiles/uniform.json")
    parser.add_argument(
        "--kg2c-predicates", default="data/kg2c/observed_predicates.json",
        help="Predicates observed in the KG2c build in use; those absent from the Biolink YAML are added "
             "as kg2c_only entries, in scope for every stratum, so profiles can follow the graph's version.",
    )
    args = parser.parse_args()

    model = yaml.safe_load(open(args.yaml, encoding="utf-8"))
    config = json.load(open(args.config, encoding="utf-8"))
    classes_raw: dict = model["classes"]
    slots_raw: dict = model["slots"]

    # ---- class tree (vertical links) and mixin groupings --------------------------------
    is_mixin = {c for c, d in classes_raw.items() if (d or {}).get("mixin")}
    parent = {c: (d or {}).get("is_a") for c, d in classes_raw.items()}
    mixins_of = {c: list((d or {}).get("mixins") or []) for c, d in classes_raw.items()}

    def ancestors(c: str) -> list[str]:
        out, cur = [], parent.get(c)
        while cur:
            out.append(cur)
            cur = parent.get(cur)
        return out

    children = defaultdict(list)
    for c, p in parent.items():
        if p:
            children[p].append(c)
    mixin_members = defaultdict(list)
    for c, ms in mixins_of.items():
        for m in ms:
            mixin_members[m].append(c)

    # Entity classes: descendants of "named thing" that are not mixins, plus mixin classes
    # kept as groupings only.
    entity_classes = [c for c in classes_raw if c not in is_mixin and ("named thing" in ancestors(c) or c == "named thing")]

    def effective_mixins(c: str) -> set[str]:
        """Mixins of the class and of its ancestors; mixins' own is_a chain is followed too."""
        out: set[str] = set()
        for k in [c] + ancestors(c):
            for m in mixins_of.get(k, []):
                out.add(m)
                out.update(ancestors(m))
        return out

    strata_cfg = config["strata"]

    def membership(c: str) -> dict[str, str]:
        """Stratum -> 'anchor' | 'angle' | 'inherited' for every stratum the class falls into.

        A class listed in a stratum's shared_angle is in that stratum under that angle, and so
        are its descendants (a disease subclass is read under the disease angle).
        """
        found = {}
        lineage = [c] + ancestors(c)
        mix = effective_mixins(c)
        for s, cfg in strata_cfg.items():
            anchors = set(cfg.get("anchor_classes", []))
            amix = set(cfg.get("anchor_mixins", []))
            angled = set(cfg.get("shared_angle", {}).keys())
            if c in anchors or c in amix:
                found[s] = "anchor"
            elif c in angled or angled.intersection(lineage):
                found[s] = "angle"
            elif anchors.intersection(lineage) or amix.intersection(mix):
                found[s] = "inherited"
        return found

    classes_out = {}
    for c in entity_classes:
        d = classes_raw.get(c) or {}
        mem = membership(c)
        status = {}
        for s in mem:
            angles = strata_cfg[s].get("shared_angle", {})
            # The angle of the class itself, else the angle of its closest angled ancestor.
            angle = angles.get(c) or next((angles[a] for a in ancestors(c) if a in angles), None)
            status[s] = {"status": "shared" if len(mem) > 1 else "core", "via": mem[s], "angle": angle}
        classes_out[c] = {
            "curie": curie(c),
            "is_a": parent.get(c),
            "ancestors": ancestors(c),
            "depth": len(ancestors(c)),
            "mixins": mixins_of.get(c, []),
            "children": sorted(children.get(c, [])),
            "description": (d.get("description") or "")[:300],
            "strata": status,
        }

    # ---- predicates: links between entity classes ----------------------------------------
    slot_parent = {s: (d or {}).get("is_a") for s, d in slots_raw.items()}

    def slot_ancestors(s: str) -> list[str]:
        out, cur = [], slot_parent.get(s)
        while cur:
            out.append(cur)
            cur = slot_parent.get(cur)
        return out

    def inherited(s: str, key: str):
        for k in [s] + slot_ancestors(s):
            v = (slots_raw.get(k) or {}).get(key)
            if v:
                return v
        return None

    def descendants(c: str) -> list[str]:
        out, stack = [], list(children.get(c, []))
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(children.get(k, []))
        return out

    def class_in_stratum(name: str, stratum: str) -> bool:
        """A domain/range is in scope when it, or any of its descendants, is in the stratum.

        Broad domains such as "biological entity" (has phenotype) or "named thing" are in scope
        for every stratum that contains one of their subclasses. A mixin domain counts if any
        member class is in scope or the mixin anchors the stratum.
        """
        if not name or name in ("named thing", "entity"):
            return True  # unconstrained
        if name in classes_out:
            if stratum in classes_out[name]["strata"]:
                return True
            return any(stratum in classes_out[d]["strata"] for d in descendants(name) if d in classes_out)
        if name in is_mixin:
            # Members through nested mixins too ("physical essence or occurrent" > "physical essence" > gene).
            members = mixin_scope.get(name, set())
            return name in set(strata_cfg[stratum].get("anchor_mixins", [])) or any(
                stratum in classes_out[m]["strata"] for m in members
            )
        return False

    mixin_scope: dict[str, set[str]] = defaultdict(set)
    for c in classes_out:
        for m in effective_mixins(c):
            mixin_scope[m].add(c)

    predicates_out = {}
    for s, d in slots_raw.items():
        if s != "related to" and "related to" not in slot_ancestors(s):
            continue
        # Slots flagged as mixins ("treats", "interacts with") are real predicates in the graphs.
        domain, rng = inherited(s, "domain"), inherited(s, "range")
        in_strata = [st for st in strata_cfg if class_in_stratum(domain, st) and class_in_stratum(rng, st)]
        predicates_out[s] = {
            "curie": curie(s),
            "is_a": slot_parent.get(s),
            "depth": len(slot_ancestors(s)),
            "domain": domain,
            "range": rng,
            "strata": in_strata,
            "description": ((d or {}).get("description") or "")[:200],
        }

    # ---- predicates present in the KG2c build but absent from this Biolink version ------
    kg2c_only = []
    if Path(args.kg2c_predicates).is_file():
        observed = json.load(open(args.kg2c_predicates, encoding="utf-8"))
        known = {v["curie"] for v in predicates_out.values()}
        for pred_curie, n_edges in observed.get("predicates", {}).items():
            if pred_curie in known:
                continue
            name = pred_curie.replace("biolink:", "").replace("_", " ")
            predicates_out[name] = {
                "curie": pred_curie, "is_a": None, "depth": None, "domain": None, "range": None,
                "strata": list(strata_cfg), "kg2c_only": True, "n_edges_kg2c": n_edges,
                "description": f"Present in {observed.get('kg_version', 'KG2c')} but not in Biolink {model.get('version')}.",
            }
            kg2c_only.append(pred_curie)

    # ---- per-stratum summary and the entity kinds to name when reading ------------------
    strata_out = {}
    for st, cfg in strata_cfg.items():
        core = sorted(c for c, v in classes_out.items() if st in v["strata"] and v["strata"][st]["status"] == "core")
        shared = {c: v["strata"][st]["angle"] for c, v in classes_out.items() if st in v["strata"] and v["strata"][st]["status"] == "shared"}
        # Entity kinds for the reading prompt: the human-defined anchors plus the angled classes,
        # each angled class carrying its reading angle. Derived descendants are for the graph,
        # not for the prompt, which must stay short.
        anchors = [c for c in cfg.get("anchor_classes", []) if c in classes_out]
        angled = cfg.get("shared_angle", {})
        entity_scope = anchors + [c for c in angled if c not in anchors]
        reading_focus = "; ".join(f"{c}: {a}" for c, a in angled.items())
        strata_out[st] = {
            "description": cfg.get("description", ""),
            "core_classes": core,
            "shared_classes": shared,
            "n_core": len(core),
            "n_shared": len(shared),
            "predicates": sorted(p for p, v in predicates_out.items() if st in v["strata"]),
            "entity_scope": entity_scope,
            "reading_focus": reading_focus,
        }

    out = {
        "biolink_version": model.get("version"),
        "config": str(args.config),
        "kg2c_only_predicates": kg2c_only,
        "strata": strata_out,
        "classes": classes_out,
        "mixins": {m: sorted(v) for m, v in mixin_members.items()},
        "predicates": predicates_out,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    uniform = {
        "name": "uniform (no profile)",
        "description": "Baseline: every Biolink predicate at weight 1, no source or knowledge-level weighting.",
        "predicates": {**{v["curie"]: 1.0 for v in predicates_out.values()}, **{p: 1.0 for p in KG2C_EXTRA_PREDICATES}},
        "sources": {},
        "knowledge_levels": {},
        "prefix_restricted_predicates": {},
        "entity_scope": [],
        "reading_focus": "",
    }
    Path(args.uniform_profile).write_text(json.dumps(uniform, indent=2), encoding="utf-8")

    print(f"Biolink {out['biolink_version']}: {len(classes_out)} entity classes, {len(mixin_members)} mixins, "
          f"{len(predicates_out)} predicates ({len(kg2c_only)} KG2c-only)")
    for st, v in strata_out.items():
        print(f"- {st}: {v['n_core']} core classes, {v['n_shared']} shared, {len(v['predicates'])} predicates in scope")
        print(f"    shared: {', '.join(v['shared_classes'])}")
    print(f"-> {args.out}\n-> {args.uniform_profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
