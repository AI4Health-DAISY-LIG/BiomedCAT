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
from typing import Any, Dict, List, Set, Tuple
from biomedcat.config import settings
import urllib.request
from pathlib import Path

import yaml

CACHE_FILE = Path(".biolink_cache.json")

# ---------------------------------------------------------------------------
# URL / path resolution
# ---------------------------------------------------------------------------

_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/"
    r"blob/(?P<ref>[^/]+)/(?P<path>.+)$"
)

def get_remote_etag(url: str) -> str:                                                                                                                                                                     
    """Récupère l'ETag (identifiant unique de version) via une requête HEAD."""                                                                                                                           
    try:                                                                                                                                                                                                  
        # On utilise la méthode HEAD pour ne pas télécharger le fichier, juste les headers                                                                                                                
        req = urllib.request.Request(url, method='HEAD')                                                                                                                                                  
        with urllib.request.urlopen(req, timeout=5) as response:                                                                                                                                          
            return response.getheader('ETag', '')                                                                                                                                                         
    except Exception as e:                                                                                                                                                                                
        print(f"[!] Impossible de récupérer l'ETag : {e}")                                                                                                                                                
        return ""

def run_smart_update(source_url: str, processor_func):                                                                                                                                                    
    """Lance le processeur uniquement si l'ETag a changé."""                                                                                                                                              
    remote_etag = get_remote_etag(source_url)                                                                                                                                                             
    local_cache = {}                                                                                                                                                                                      
                                                                                                                                                                                                          
    if CACHE_FILE.exists():                                                                                                                                                                               
        try:                                                                                                                                                                                              
            with open(CACHE_FILE, "r") as f:                                                                                                                                                              
                local_cache = json.load(f)                                                                                                                                                                
        except json.JSONDecodeError:                                                                                                                                                                      
            local_cache = {}                                                                                                                                                                              
                                                                                                                                                                                                          
    if remote_etag and local_cache.get("etag") == remote_etag:                                                                                                                                            
        print("[*] Le modèle Biolink est déjà à jour (ETag identique).")                                                                                                                                  
        return False # Pas de mise à jour effectuée                                                                                                                                                       
                                                                                                                                                                                                          
    print(f"[*] Changement détecté ou première exécution (Remote: {remote_etag}). Mise à jour...")                                                                                                        
    processor_func(source=source_url)                                                                                                                                                                     
                                                                                                                                                                                                          
    # Sauvegarde du nouvel ETag                                                                                                                                                                           
    with open(CACHE_FILE, "w") as f:                                                                                                                                                                      
        json.dump({"etag": remote_etag}, f)                                                                                                                                                               
    return True


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
        
        local_cache = "data/biolink-model.yaml"
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
    output_nested: str = "data/biolink_classes_nested.json",
    output_flat: str = "data/biolink_classes_flat.json",
    max_examples: int = 5  # Constraint: Prevent context window saturation
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse the Biolink Model YAML and emit enriched hierarchical + flat JSON.
    
    Implements Phase 1: 
      1. Augmented 'doc_text' for Dense retrieval (Chroma).
      2. Structured 'metadata' for Sparse/Agentic retrieval (BM25).
      3. Context management via example truncation.
    """
    print(f"Biolink model source: {source}")
    data = _load_yaml(source)
    classes_section: Dict[str, Any] = data.get("classes", {})

    flat: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    mixin_children_map: Dict[str, List[str]] = defaultdict(list)

    # --- First Pass: Build enriched entries -------------------------------
    counter_ex = 0
    for class_name, class_def in classes_section.items():
        if not isinstance(class_def, dict):
            continue

        # Extract core attributes
        parent = class_def.get("is_a")
        definition = class_def.get("description", "")
        aliases = class_def.get("aliases", []) or []
        mixins = class_def.get("mixins", []) or []
        class_uri = class_def.get("class_uri")
        
        # Requirement 3: Handle examples with truncation to prevent context saturation
        raw_examples = class_def.get("examples", [])                                                                                                                                                       
        examples = []                                                                                                                                                                                      
                                                                                                                                                                                                           
        if isinstance(raw_examples, list):
            if len(raw_examples) != 0:
                counter_ex += 1
                extracted_texts = []
                for num_examples, ex in enumerate(raw_examples): # if list of dicts
                    if num_examples <= max_examples: # limit number of examples
                        if isinstance(ex,str):
                            extracted_texts.append(raw_examples) # HARD TRUNCATION                                                                                                                                                        
                        elif isinstance(ex, dict):
                                text_parts = []
                                for v in ex.values():
                                    if isinstance(v, (str, int)):
                                        text_parts.append(str(v))
                                    elif isinstance(v,dict):
                                        text_parts.append(" ".join([str(v1) for v1 in v.values()]))
                                    else:
                                        print("Could not parse the example.")                                                                                              
                                if text_parts:                                                                                                                                                                         
                                    extracted_texts.append(" ".join(text_parts))                                                                                                                                       
                        else:                                                                                                                                      
                            extracted_texts.append(str(ex))
                
            else:
                extracted_texts = '' 
        elif isinstance(raw_examples, str):
            extracted_texts = raw_examples 
        elif isinstance(raw_examples, dict):                                                                                                                                                                          
            for entry in raw_examples.values():                                                                                                                                                            
                if isinstance(entry, dict):                                                                                                                                                                
                    # On concatène toutes les valeurs textuelles de l'objet (value, description, etc.)                                                                                                     
                    text_parts = [str(v) for v in entry.values() if isinstance(v, (str, int))]                                                                                                             
                    if text_parts:                                                                                                                                                                         
                        extracted_texts.append(" ".join(text_parts))                                                                                                                                       
                else:                                                                                                                                      
                    extracted_texts.append(str(entry)) 
                                                                                                                                                   
        examples = extracted_texts # HARD TRUNCATION

        # Requirement 1: Augmented Sentence for Dense Embedding (Doc_Text)
        ex_string = "; ".join(examples)

        doc_text = f"Class: {class_name}. Definition: {definition}. Examples: {ex_string}."

        # Requirement 2: Structured Metadata for Agent and Sparse BM25
        metadata = {
            "definition": definition,
            "aliases": aliases,
            "examples": examples,
            "mixins": mixins,
            "class_uri": class_uri,
            "biolink_definition": class_def, # Keep original for reference
            "parent": parent,
        }

        # The 'flat' entry separates the searchable text from the structured metadata
        flat[class_name] = {
            "doc_text": doc_text,
            "metadata": metadata
        }

        if parent:
            children_map[parent].append(class_name)
        for mx in mixins:
            mixin_children_map[mx].append(class_name)

    # Check for orphan classes
    for class_name, entry in flat.items():
        p = entry["metadata"]["parent"]
        if p and p not in classes_section:
            warnings.warn(f"Class {class_name!r} has orphan parent {p!r}", stacklevel=2)

    # --- Second Pass: Build hierarchy -------------------------------------
    for class_name, entry in flat.items():                                                                                 
        meta = entry["metadata"]                                                                                           
        parent = meta["parent"]                                                                                            
                                                                                                                           
        # Utilisation de children_map qui est déjà complet et fiable                                                       
        if parent and parent in flat:                                                                                      
            # On récupère les enfants via le map, pas via le metadata du parent                                            
            parent_children = children_map.get(parent, [])                                                                 
            meta["siblings"] = sorted(c for c in parent_children if c != class_name)                                       
                                                                                                                           
            # On met à jour le metadata du parent pour la structure arborescente (Nested Tree)                             
            flat[parent]["metadata"]["children"] = sorted(parent_children)                                                 
        else:                                                                                                              
            meta["siblings"] = [] 

    # --- Third Pass: Ancestors & Descendants ------------------------------
    for class_name, entry in flat.items():
        chain: List[str] = []
        visited: Set[str] = set()
        current = entry["metadata"]["parent"]
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            if current in flat:
                current = flat[current]["metadata"]["parent"]
            else:
                break
        entry["metadata"]["ancestors"] = list(reversed(chain))
        entry["metadata"]["descendants"] = sorted(_collect_descendants(class_name, children_map))

    # --- Fourth Pass: Mixins & Neighbors ----------------------------------
    for class_name, entry in flat.items():
        meta = entry["metadata"]
        meta["mixin_children"] = sorted(mixin_children_map.get(class_name, []))
        
        neighbours: Set[str] = set()
        if meta["parent"]:
            neighbours.add(meta["parent"])
        
        neighbours.update(children_map.get(class_name, []))
        neighbours.update(meta.get("siblings", []))
        neighbours.update(meta.get("mixins", []))
        neighbours.update(meta["mixin_children"])
        neighbours.discard(class_name)
        meta["neighbors"] = sorted(list(neighbours))

    # --- Build Nested Tree -----------------------------------------------
    roots = [name for name, e in flat.items() if e["metadata"]["parent"] is None]
    orphans = [name for name, e in flat.items() if e["metadata"]["parent"] and e["metadata"]["parent"] not in classes_section]
    roots = sorted(set(roots + orphans))

    nested: Dict[str, Any] = {}
    for root in roots:
        nested[root] = _build_tree_node(root, flat, children_map)

    # --- Final Output -----------------------------------------------------
    with open(output_nested, "reg_utf8" if False else "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2, ensure_ascii=False)
    with open(output_flat, "w", encoding="utf-8") as fh:
        json.dump(flat, fh, indent=2, ensure_ascii=False)

    return nested, flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_descendants(node: str, children_map: Dict[str, List[str]]) -> List[str]:
    """Return all transitive descendants of node."""
    result: List[str] = [] # Note: type hint fix needed if strictly following typing
    res: List[str] = []
    for child in children_map.get(node, []):
        res.append(child)
        res.extend(_collect_descendants(child, children_map))
    return res


def _build_tree_node(
    name: str,
    flat: Dict[str, Dict[str, Any]],
    children_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Recursively build a nested-tree node using the metadata structure."""
    entry = flat[name]
    meta = entry["metadata"]
    node: Dict[str, Any] = {
        "definition": meta.get("definition", ""),
        "aliases": meta.get("aliases", []),
        "mixins": meta.get("mixins", []),
        "class_uri": meta.get("class_uri"),
        "biolink_definition": meta.get("biolink_definition"),
    }

    child_names = sorted(children_map.get(name, []))
    if child_names:
        node["children"] = {
            child: _build_tree_node(child, flat, children_map)
            for child in child_names
        }
    return node

if __name__ == "__main__":
    print ("Loading state of the art data model and computing local knowledge base...")
    run_smart_update(settings.biolobink_model_data, biolink_yml_processor)
    print("biolink data model processing... done.")