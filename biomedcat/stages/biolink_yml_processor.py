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
from typing import Any, Dict, List, Set, Tuple, Callable, Optional
from biomedcat.config import settings
import urllib.request
from pathlib import Path

import yaml

CACHE_FILE = Path("data/.biolink_cache.json")

# ---------------------------------------------------------------------------
# URL / path resolution
# ---------------------------------------------------------------------------
# --- Github -------------------------------
_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/"
    r"(?P<owner>[^/]+)/"
    r"(?P<repo>[^/]+)/"
    r"blob/"
    r"(?P<ref>[^/]+)/"
    r"(?P<path>.+)$"
)

def get_remote_blob_sha(url: str) -> str:
    """
    Retrieve the Git blob SHA for a file referenced by a GitHub blob URL.

    The returned SHA identifies the file content, not an HTTP response.
    """
    api_url = github_blob_to_api_url(url)

    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "biolink-model-updater",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"GitHub API request failed with HTTP {error.code}: "
            f"{error.reason}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach GitHub API: {error.reason}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected GitHub API response.")

    blob_sha = payload.get("sha")

    if not blob_sha:
        raise RuntimeError(
            "GitHub API response does not contain a file SHA. "
            "The URL may refer to a directory, or the file may not exist."
        )

    return str(blob_sha)

def github_blob_to_api_url(url: str) -> str:
    """
    Convert a GitHub web/blob URL to a GitHub Contents API URL.

    Example:
        https://github.com/biolink/biolink-model/blob/master/
        biolink-model.yaml

    becomes:
        https://api.github.com/repos/biolink/biolink-model/contents/
        biolink-model.yaml?ref=master
    """
    match = _GITHUB_BLOB_RE.match(url)

    if not match:
        raise ValueError(
            "Expected a GitHub blob URL of the form: "
            "https://github.com/OWNER/REPO/blob/REF/PATH"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    ref = match.group("ref")
    path = match.group("path")

    encoded_path = urllib.parse.quote(path, safe="/")
    encoded_ref = urllib.parse.quote(ref, safe="")

    return (
        f"https://api.github.com/repos/{owner}/{repo}/contents/"
        f"{encoded_path}?ref={encoded_ref}"
    )

def load_cache(cache_file: Path) -> Dict[str, Any]:
    """
    Load the update cache. Invalid or missing caches are treated as empty.
    """
    if not cache_file.exists():
        return {}

    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return payload if isinstance(payload, dict) else {}

    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(
    cache_file: Path,
    source_url: str,
    remote_sha: str,
) -> None:
    """
    Save the source URL and Git blob SHA atomically.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_url": source_url,
        "blob_sha": remote_sha,
    }

    temporary_file = cache_file.with_suffix(
        cache_file.suffix + ".tmp"
    )

    with temporary_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    temporary_file.replace(cache_file)

# def get_remote_etag(url: str) -> str:                                                                                                                                                                     
#     """Récupère l'ETag (identifiant unique de version) via une requête HEAD."""                                                                                                                           
#     try:                                                                                                                                                                                                  
#         # On utilise la méthode HEAD pour ne pas télécharger le fichier, juste les headers                                                                                                                
#         req = urllib.request.Request(url, method='HEAD')                                                                                                                                                  
#         with urllib.request.urlopen(req, timeout=5) as response:                                                                                                                                          
#             return response.getheader('ETag', '')                                                                                                                                                         
#     except Exception as e:                                                                                                                                                                                
#         print(f"[!] Impossible de récupérer l'ETag : {e}")                                                                                                                                                
#         return ""

def _resolve_source(source: str) -> str:
    """Convert a GitHub ``blob`` URL to a ``raw.githubusercontent.com`` URL."""
    m = _GITHUB_BLOB_RE.match(source)
    if m:
        return (
            f"https://raw.githubusercontent.com/"
            f"{m['owner']}/{m['repo']}/{m['ref']}/{m['path']}"
        )
    return source

# --- Files -------------------------------
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
    all_classes: Dict[str, Any] = data.get("classes", {}) or {}

    if not isinstance(all_classes, dict):
        raise ValueError("The YAML `classes` section must be a mapping.")
    
    # --- Restrict biolink to entity classes -------------------------------
    # Restrict the parser to entity classes by excluding `association`
    # and every class inheriting from it.
    classes_section: Dict[str, Dict[str, Any]] = {
        class_name: class_def
        for class_name, class_def in all_classes.items()
        if isinstance(class_def, dict) and _is_entity_class(class_name, all_classes) and not _is_association_class(
            class_name,
            class_def,
            all_classes,
        )
    }

    # --- First Pass: Build enriched entries -------------------------------
    flat: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    mixin_children_map: Dict[str, List[str]] = defaultdict(list)
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
            "aliases": ", ".join(aliases) if aliases else "",
            "examples": "; ".join(examples) if examples else "",
            "mixins": ", ".join(mixins) if mixins else "",
            "class_uri": class_uri,
            #"biolink_definition": class_def, # Keep original for reference - cannot add because dict
            "parent": parent
        }

        # The 'flat' entry separates the searchable text from the structured metadata
        flat[class_name] = {
            "doc_text": doc_text,
            "metadata": metadata
        }

        if parent and parent in classes_section:
            children_map[parent].append(class_name)
        for mixin in mixins:
            if mixin in classes_section:
                mixin_children_map[mixin].append(class_name)

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
        
        neighbors: Set[str] = set()
        if meta["parent"]:
            neighbors.add(meta["parent"])
        
        neighbors.update(children_map.get(class_name, []))
        neighbors.update(meta.get("siblings", []))
        neighbors.update(meta.get("mixins", []))
        neighbors.update(meta["mixin_children"])
        neighbors.discard(class_name)
        meta["neighbors"] = sorted(list(neighbors))

    # --- Build Nested Tree -----------------------------------------------
    roots = [class_name for class_name, entry in flat.items() if not entry["metadata"]["parent"] or entry["metadata"]["parent"] not in flat]
    orphans = [name for name, e in flat.items() if e["metadata"]["parent"] and e["metadata"]["parent"] not in classes_section]
    roots = sorted(set(roots + orphans))

    nested: Dict[str, Any] = {}
    for root in roots:
        nested[root] = _build_tree_node(root, flat, children_map)

    # --- Clean flat for Chroma indexing:
    flat_for_chroma = {}
    keys_to_keep = {"definition","aliases","examples","mixins","class_uri","parent","siblings"}
    
    for class_name, entry in flat.items():  
        new_entry = {                                                                                                                          
            "doc_text": entry["doc_text"],                                                                                                     
            "metadata": {}                                                                                                                     
        }   

        for k in keys_to_keep:                                                                                                                 
            if k in entry["metadata"]:                                                                                                         
                val = entry["metadata"][k]                                                                                                     
                                                                                                                                               
                if isinstance(val, list):                                                                                                      
                    # Conversion de la liste en chaîne (ex: ['a', 'b'] -> "a, b")                                                              
                    new_entry["metadata"][k] = ", ".join(map(str, val)) if val else ""                                                         
                # elif isinstance(val, dict):                                                           
                #     new_entry["metadata"][k] = json.dumps(val)                                                                                 
                elif isinstance(val, (str, int, float, bool)):                                                                                                                          
                    # Valeur simple (str, int, etc.)                                                                                           
                    new_entry["metadata"][k] = val
                elif v is None:
                    new_entry["metadata"][k] = ""
                else:
                    new_entry["metadata"][k] = str(v)                                                                                          
                                                                                                                                               
        flat_for_chroma[class_name] = new_entry

    # --- Final Output -----------------------------------------------------
    with open(output_nested, "reg_utf8" if False else "w", encoding="utf-8") as fh:
        json.dump(nested, fh, indent=2, ensure_ascii=False)
    with open(output_flat, "w", encoding="utf-8") as fh:
        json.dump(flat_for_chroma, fh, indent=2, ensure_ascii=False)

    return nested, flat_for_chroma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_smart_update(
    source_url: str,
    processor_func: Callable[..., Any],
    cache_file: Optional[Path] = None,
    ) -> bool:

    """
    Run the processor only if the GitHub file content changed.

    Returns:
        True  if the processor ran.
        False if the cached Git blob SHA is unchanged.
    """
    cache_file = cache_file or CACHE_FILE

    remote_sha = get_remote_blob_sha(source_url)
    local_cache = load_cache(cache_file)

    cached_sha = local_cache.get("blob_sha")
    cached_source_url = local_cache.get("source_url")

    if (cached_sha == remote_sha and cached_source_url == source_url):
        print("[*] Biolink Model standards is already up to date. ") #(Git blob SHA identique)."
        return False

    if cached_sha is None:
        reason = "première exécution"
    elif cached_source_url != source_url:
        reason = "source modifiée"
    else:
        reason = "contenu modifié"

    print(
        f"[*] Mise à jour nécessaire ({reason}). "
        f"Remote blob SHA: {remote_sha}"
    )

    processor_func(source=source_url)

    save_cache(
        cache_file=cache_file,
        source_url=source_url,
        remote_sha=remote_sha,
    )

    return True

def _is_entity_class(
    class_name: str,
    all_classes: Dict[str, Any],
) -> bool:
    visited: Set[str] = set()
    current = class_name

    while current and current not in visited:
        visited.add(current)

        if current == "entity":
            return True

        current_def = all_classes.get(current, {})
        if not isinstance(current_def, dict):
            return False

        current = current_def.get("is_a")

    return False

def _is_association_class(
    class_name: str,
    class_def: Dict[str, Any],
    all_classes: Dict[str, Any],
) -> bool:
    """
    Return True if a Biolink class is an association class or inherits from one.

    Association classes are identified recursively through `is_a`.
    This excludes:
      - association
      - all descendants of association
      - classes explicitly marked as association classes by `defining_slots`
        or association-related metadata
    """
    visited: Set[str] = set()
    current = class_name

    while current and current not in visited:
        visited.add(current)

        if current == "association":
            return True

        current_def = all_classes.get(current, {})
        if not isinstance(current_def, dict):
            return False

        current = current_def.get("is_a")

    return False

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
    run_smart_update(settings.biolink_model_data, biolink_yml_processor)
    print("biolink data model processing... done.")