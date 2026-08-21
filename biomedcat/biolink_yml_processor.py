"""
biolink_yml_processor.py
========================

Process the Biolink Model YAML file and produce a JSON structure that:
  1. Maps every Biolink class name to its definition (description + full spec).
  2. Preserves the class hierarchy via ``is_a`` so callers can navigate
     parents, children, siblings, ancestors, and descendants.
  3. Exposes ``mixins``-based neighbour lookups (mixin parents / children).

Three outputs are produced (written to disk and returned as Python dicts):

  ``biolink_classes_nested.json``
      Nested tree built from the ``is_a`` hierarchy.  Each node::

          {
              "definition":       "<description or null>",
              "aliases":           [...],
              "mixins":            [...],
              "class_uri":         "<URI or null>",
              "biolink_definition": { ... full original YAML dict ... },
              "children":          { <child_name>: { ... }, ... }
          }

  ``biolink_classes_flat.json``
      Flat dictionary keyed by class name.  Each entry::

          {
              "definition":          "<description or null>",
              "aliases":             [...],
              "mixins":              [...],
              "class_uri":           "<URI or null>",
              "biolink_definition":  { ... full original YAML dict ... },
              "parent":              "<parent class or null>",
              "children":            ["ChildA", ...],
              "siblings":            ["SiblingA", ...],
              "ancestors":           ["Root", ..., "direct_parent"],
              "descendants":         ["Desc1", ...],
              "mixin_parents":       ["MixinParent1", ...],
              "mixin_children":      ["ClassUsingThisAsMixin", ...],
              "neighbors":           ["unique", "neighbour", "names", ...]
          }

Usage
-----
::

    from biolink_yml_processor import biolink_yml_processor

    nested, flat = biolink_yml_processor(
        source="https://github.com/biolink/biolink-model/blob/master/biolink-model.yaml",
    )

    # Neighbour lookups (note: Biolink uses lowercase / spaced names)
    gene = flat["gene"]
    print(gene["parent"])          # "biological entity"
    print(gene["children"])        # []
    print(gene["siblings"])        # ["biological process or activity", ...]
    print(gene["ancestors"])       # ["entity", "named thing", "biological entity"]
    print(gene["mixin_parents"])   # ["gene or gene product", "genomic entity", ...]
    print(gene["neighbors"])       # union of all neighbour names
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from biomedcat.config import settings

import yaml


# ---------------------------------------------------------------------------
# URL / path resolution
# ---------------------------------------------------------------------------

_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/"
    r"blob/(?P<ref>[^/]+)/(?P<path>.+)$"
)


def _resolve_source(source: str) -> str:
    """Convert a GitHub ``blob`` URL to a ``raw.githubusercontent.com`` URL.

    Falls through unchanged for local paths or already-raw URLs.
    """
    m = _GITHUB_BLOB_RE.match(source)
    if m:
        return (
            f"https://raw.githubusercontent.com/"
            f"{m['owner']}/{m['repo']}/{m['ref']}/{m['path']}"
        )
    return source


def _load_yaml(source: str) -> Dict[str, Any]:
    """Load YAML from a local path, raw URL, or GitHub blob URL."""
    # local file?
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    resolved = _resolve_source(source)

    # looks like a URL?
    if resolved.startswith(("http://", "https://")):
        print("Biolink knowledge updating: importing...")
        import urllib.request
        with urllib.request.urlopen(resolved) as resp:
            raw = resp.read().decode("utf-8")
        # cache locally for convenience
        local_cache = os.path.basename(resolved.split("?")[0]) or "biolink-model.yaml"
        with open(local_cache, "w", encoding="utf-8") as fh:
            fh.write(raw)
        print(".yml loaded")
        return yaml.safe_load(raw)

    raise FileNotFoundError(
        f"Could not resolve source: {source!r}. "
        f"Pass a local file path, a raw URL, or a GitHub blob URL."
    )


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def biolink_yml_processor(
    source: str = settings.biolink_model_data,
    output_nested: str = "biolink_classes_nested.json",
    output_flat: str = "biolink_classes_flat.json",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse the Biolink Model YAML and emit hierarchical + flat JSON.

    Parameters
    ----------
    source : str
        Path to the local YAML file, a raw GitHub URL, or a GitHub blob URL.
    output_nested : str
        Path for the nested-tree JSON output.
    output_flat : str
        Path for the flat lookup-table JSON output.

    Returns
    -------
    (nested_dict, flat_dict)
    """
    print(f"Biolink model source: {source}")
    data = _load_yaml(source)
    classes_section: Dict[str, Any] = data.get("classes", {})

    # --- first pass: build flat entries -----------------------------------
    flat: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)      # is_a parent → children
    mixin_children_map: Dict[str, List[str]] = defaultdict(list)  # mixin target → users

    for class_name, class_def in classes_section.items():
        if not isinstance(class_def, dict):
            continue

        parent = class_def.get("is_a")
        definition = class_def.get("description")
        aliases = class_def.get("aliases", []) or []
        mixins = class_def.get("mixins", []) or []
        class_uri = class_def.get("class_uri")

        flat[class_name] = {
            "definition": definition,
            "aliases": aliases,
            "mixins": mixins,
            "class_uri": class_uri,
            "biolink_definition": class_def,          # full original spec
            "parent": parent,
            "children": [],
            "siblings": [],
            "ancestors": [],
            "descendants": [],
            "mixin_parents": mixins.copy(),           # initialised from own mixins
            "mixin_children": [],
            "neighbors": [],
        }

        if parent:
            children_map[parent].append(class_name)
        for mx in mixins:
            mixin_children_map[mx].append(class_name)

    # Warn about parents that don't exist (orphan / external references)
    for class_name, entry in flat.items():
        p = entry["parent"]
        if p and p not in classes_section:
            warnings.warn(
                f"Class {class_name!r} has is_a={p!r} which is not in the "
                f"classes section — treated as a root.",
                stacklevel=2,
            )

    # --- fill children / siblings -----------------------------------------
    for class_name, entry in flat.items():
        entry["children"] = sorted(children_map.get(class_name, []))

    for class_name, entry in flat.items():
        parent = entry["parent"]
        if parent and parent in flat:
            entry["siblings"] = sorted(
                c for c in flat[parent]["children"] if c != class_name
            )
        elif parent and parent in children_map:
            # parent referenced but not a class in the file
            entry["siblings"] = sorted(
                c for c in children_map[parent] if c != class_name
            )

    # --- ancestors (root → ... → direct parent) with cycle detection ------
    for class_name, entry in flat.items():
        chain: List[str] = []
        visited: Set[str] = set()
        current = entry["parent"]
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            if current in flat:
                current = flat[current]["parent"]
            else:
                break
        entry["ancestors"] = list(reversed(chain))   # root first

    # --- descendants (transitive) ----------------------------------------
    for class_name, entry in flat.items():
        entry["descendants"] = sorted(_collect_descendants(class_name, children_map))

    # --- mixin children (who uses this class as a mixin?) ----------------
    for class_name, entry in flat.items():
        entry["mixin_children"] = sorted(mixin_children_map.get(class_name, []))

    # --- combined neighbours ----------------------------------------------
    for class_name, entry in flat.items():
        neighbours: Set[str] = set()
        if entry["parent"]:
            neighbours.add(entry["parent"])
        neighbours.update(entry["children"])
        neighbours.update(entry["siblings"])
        neighbours.update(entry["mixin_parents"])
        neighbours.update(entry["mixin_children"])
        neighbours.discard(class_name)
        entry["neighbors"] = sorted(neighbours)

    # --- build nested tree ------------------------------------------------
    roots = [name for name, e in flat.items() if e["parent"] is None]
    # Also include orphans (parent not in classes_section)
    orphans = [
        name for name, e in flat.items()
        if e["parent"] and e["parent"] not in classes_section
    ]
    roots.extend(orphans)
    roots = sorted(set(roots))

    nested: Dict[str, Any] = {}
    for root in roots:
        nested[root] = _build_tree_node(root, flat, children_map)

    # --- write outputs ----------------------------------------------------
    with open(output_nested, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2, ensure_ascii=False)
    with open(output_flat, "w", encoding="utf-8") as fh:
        json.dump(flat, fh, indent=2, ensure_ascii=False)

    return nested, flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_descendants(node: str, children_map: Dict[str, List[str]]) -> List[str]:
    """Return all transitive descendants of *node*."""
    result: List[str] = []
    for child in children_map.get(node, []):
        result.append(child)
        result.extend(_collect_descendants(child, children_map))
    return result


def _build_tree_node(
    name: str,
    flat: Dict[str, Dict[str, Any]],
    children_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Recursively build a nested-tree node from the flat dictionary."""
    entry = flat[name]
    node: Dict[str, Any] = {
        "definition": entry["definition"],
        "aliases": entry["aliases"],
        "mixins": entry["mixins"],
        "class_uri": entry["class_uri"],
        "biolink_definition": entry["biolink_definition"],
    }
    child_names = sorted(children_map.get(name, []))
    if child_names:
        node["children"] = {
            child: _build_tree_node(child, flat, children_map)
            for child in child_names
        }
    return node


# ---------------------------------------------------------------------------
# Neighbour-search utilities (operate on the flat dict)
# ---------------------------------------------------------------------------

def get_parent(flat: Dict[str, Any], class_name: str) -> Optional[str]:
    """Return the direct parent class name, or ``None``."""
    return flat.get(class_name, {}).get("parent")


def get_children(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return direct children of *class_name*."""
    return flat.get(class_name, {}).get("children", [])


def get_siblings(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return sibling classes (same parent, excluding self)."""
    return flat.get(class_name, {}).get("siblings", [])


def get_ancestors(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return ancestors from root down to direct parent."""
    return flat.get(class_name, {}).get("ancestors", [])


def get_descendants(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return all transitive descendants."""
    return flat.get(class_name, {}).get("descendants", [])


def get_mixin_parents(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return the mixins that *class_name* inherits from."""
    return flat.get(class_name, {}).get("mixin_parents", [])


def get_mixin_children(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return classes that use *class_name* as a mixin."""
    return flat.get(class_name, {}).get("mixin_children", [])


def get_neighbors(flat: Dict[str, Any], class_name: str) -> List[str]:
    """Return a sorted list of all neighbouring class names.

    Combines parent, children, siblings, mixin_parents, and mixin_children
    into a single de-duplicated list.
    """
    return flat.get(class_name, {}).get("neighbors", [])


def search_by_keyword(
    flat: Dict[str, Any],
    keyword: str,
    fields: Tuple[str, ...] = ("definition", "aliases"),
) -> List[str]:
    """Case-insensitive search across definition text and aliases.

    Returns a sorted list of class names whose *fields* contain *keyword*.
    """
    keyword_lower = keyword.lower()
    results: List[str] = []
    for class_name, entry in flat.items():
        for field in fields:
            value = entry.get(field)
            if isinstance(value, str):
                if keyword_lower in value.lower():
                    results.append(class_name)
                    break
            elif isinstance(value, list):
                if any(keyword_lower in str(v).lower() for v in value):
                    results.append(class_name)
                    break
    return sorted(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Biolink Model YAML → hierarchical + flat JSON."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="https://github.com/biolink/biolink-model/blob/master/biolink-model.yaml",
        help="Local path, raw URL, or GitHub blob URL for biolink-model.yaml.",
    )
    parser.add_argument(
        "--nested-out",
        default="biolink_classes_nested.json",
        help="Output path for nested JSON.",
        )
    parser.add_argument(
        "--flat-out",
        default="biolink_classes_flat.json",
        help="Output path for flat JSON.",
    )
    args = parser.parse_args()

    nested_result, flat_result = biolink_yml_processor(
        source=args.source,
        output_nested=args.nested_out,
        output_flat=args.flat_out,
    )

    print(f"Classes processed: {len(flat_result)}")
    print(f"Root classes:      {len(nested_result)}")
    print(f"Nested JSON → {args.nested_out}")
    print(f"Flat JSON   → {args.flat_out}")

    # Demo: show hierarchy depth
    max_depth = 0
    deepest = ""
    for name, entry in flat_result.items():
        depth = len(entry["ancestors"])
        if depth > max_depth:
            max_depth = depth
            deepest = name
    print(f"Deepest class:     {deepest} (depth {max_depth})")

    # Demo: mixin neighbours
    gene = flat_result.get("gene")
    if gene:
        print(f"\ngene neighbours: {gene['neighbors'][:10]}")
        print(f"gene mixin_parents: {gene['mixin_parents']}")
        print(f"gene mixin_children: {gene['mixin_children'][:5]}")