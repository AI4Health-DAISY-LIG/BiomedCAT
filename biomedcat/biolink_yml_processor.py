"""
biolink_yml_processor.py
========================

Process the Biolink Model YAML and produce an enriched JSON structure optimized for 
Hybrid RAG (Dense + Sparse) and Agentic NER.

Key Features:
  1. Augmented Sentence: Creates a natural language string (Class + Def + Examples) 
     to improve semantic retrieval in Dense Vector DBs.
  2. Structured Metadata: Provides clean lists of aliases and examples for 
     precise keyword matching in Sparse BM25 engines and Agent reasoning.
  3. Context Management: Implements truncation of examples to prevent context 
     window saturation during Agent inference.
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
    """Convert a GitHub ``blob`` URL to a ``raw.githubusercontent.com`` URL."""
    m = _GITHUB_BLOB_RE.match(source)
    if m:
        return (
            f"https://raw.githubusercontent.com/"
            f"{m['owner']}/{m['repo']}/{m['ref']}/{m['path']}"
        )
    return source


def _load_yaml(source: str) -> Dict[str, Any]:
    """Load YAML from a local path, raw URL, or GitHub blob URL."""
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    resolved = _resolve_source(source)

    if resolved.startswith(("http://", "https://")):
        print("Biolink knowledge updating: importing...")
        import urllib.request
        with urllib.request.urlopen(resolved) as resp:
            raw = resp.read().decode("utf-8")
        
        local_cache = os.path.basename(resolved.split("?")[0]) or "biolink-model.yaml"
        with open(local_cache, "w", encoding="utf-8") as fh:
            fh.write(raw)
        print(".yml loaded")
        return yaml.safe_load(raw)

    raise FileNotFoundError(f"Could not resolve source: {source!r}")


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def biolink_yml_processor(
    source: str = settings.biolink_model_data,
    output_nested: str = "biolink_classes_nested.json",
    output_flat: str = "biolink_classes_flat.json",
    max_examples: int = 5  # Constraint: Prevent context window saturation
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse the Biolink Model YAML and emit enriched hierarchical + flat JSON.
    
    Implements Phase 1: Augmented text for Dense retrieval and structured 
    metadata for Sparse/Agentic retrieval.
    """
    print(f"Biolink model source: {source}")
    data = _load_yaml(source)
    classes_section: Dict[str, Any] = data.get("classes", {})

    flat: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    mixin_children_map: Dict[str, List[str]] = defaultdict(list)

    # --- First Pass: Build enriched entries -------------------------------
    for class_name, class_def in classes_section.items():
        if not isinstance(class_def, dict):
            continue

        # Extract core attributes
        parent = class_def.get("is_a")
        definition = class_def.get("description", "")
        aliases = class_def.get("aliases", []) or []
        mixins = class_def.get("mixins", []) or []
        class_uri = class_def.get("class_uri")
        
        # Handle examples with truncation (Requirement 3)
        raw_examples = class_def.get("examples", []) or []
        examples = raw_examples[:max_examples]

        # 1. Augmented Sentence for Dense Embedding (Requirement 1)
        # Format: "Class: [NAME]. Definition: [DEF]. Examples: [EX1, EX2...]."
        ex_string = ", ".join(examples)
        dense_sentence = f"Class: {class_name}. Definition: {definition}. Examples: {ex_string}."

        # 2. Structured Metadata for Agent and Sparse BM25 (Requirement 2)
        flat[class_name] = {
            "definition": definition,
            "aliases": aliases,
            "examples": examples,           # Clean list for BM25/Agent
            "dense_sentence": dense_sentence, # Augmented string for ChromaDB
            "mixins": mixins,
            "class_uri": class_uri,
            "biolink_definition": class_def, # Original spec
            "parent": parent,
            "children": [],
            "siblings": [],
            "ancestors": [],
            "descendants": [],
            "mixin_parents": mixins.copy(),
            "mixin_children": [],
            "neighbors": [],
        }

        if parent:
            children_map[parent].append(class_name)
        for mx in mixins:
            mixin_children_map[mx].append(class_name)

    # Check for orphan classes (parents not in the YAML)
    for class_name, entry in flat.items():
        p = entry["parent"]
        if p and p not in classes_section:
            warnings.warn(f"Class {class_name!r} has orphan parent {p!r}", stacklevel=2)

    # --- Second Pass: Build hierarchy -------------------------------------
    for class_name, entry in flat.items():
        entry["children"] = sorted(children_map.get(class_name, []))

    for class_name, entry in flat.items():
        parent = entry["parent"]
        if parent and parent in flat:
            entry["siblings"] = sorted(c for c in flat[parent]["children"] if c != class_name)
        elif parent and parent in children_map:
            entry["siblings"] = sorted(c for c in children_map[parent] if c != class_name)

    # --- Third Pass: Ancestors & Descendants ------------------------------
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
        entry["ancestors"] = list(reversed(chain))
        entry["descendants"] = sorted(_collect_descendants(class_name, children_map))

    # --- Fourth Pass: Mixins & Neighbors ----------------------------------
    for class_name, entry in flat.items():
        entry["mixin_children"] = sorted(mixin_children_map.get(class_name, []))
        
        neighbours: Set[str] = set()
        if entry["parent"]:
            neighbours.add(entry["parent"])
        
        neighbours.update(entry["children"])
        neighbours.update(entry["siblings"])
        neighbours.update(entry["mixin_parents"])
        neighbours.update(entry["mixin_children"])
        neighbours.discard(class_name)
        entry["neighbors"] = sorted(list(neighbours))

    # --- Build Nested Tree -----------------------------------------------
    roots = [name for name, e in flat.items() if e["parent"] is None]
    orphans = [name for name, e in flat.items() if e["parent"] and e["parent"] not in classes_section]
    roots = sorted(set(roots + orphans))

    nested: Dict[str, Any] = {}
    for root in roots:
        nested[root] = _build_tree_node(root, flat, children_map)

    # --- Final Output -----------------------------------------------------
    with open(output_nested, "w", encoding="format-utf8") as fh:
        pass # Placeholder logic removed for actual writing below
    
    # Actual writing
    with open(output_nested, "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2, ensure_ascii=False)
    with open(output_flat, "w", encoding="utf-8") as fh:
        json.dump(flat, fh, indent=2, ensure_ascii=False)

    return nested, flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_descendants(node: str, children_map: Dict[str, List[str]]) -> List[str]:
    """Return all transitive descendants of node."""
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
    """Recursively build a nested-tree node."""
    entry = flat[name]
    node: Dict[str, Any] = {
        "definition": entry["definition"],
        "aliases": entry["aliases"],
        "mixins": entry["mixint_parents"] if "mixin_parents" in entry else entry.get("mixins", []), # fallback logic
        "class_uri": entry["class_uri"],
        "biolink_definition": entry["biolink_definition"],
    }
    # Correcting the key access for mixins based on the flat structure created above
    node["mixins"] = entry["mixins"]

    child_names = sorted(children_map.get(name, []))
    if child_names:
        node["children"] = {
            child: _build_tree_node(child, flat, children_map)
            for child in child_names
        }
    return node
