import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

from biomedcat.config import settings
from biomedcat.types import Candidate

logger = logging.getLogger(__name__)
TYPE_TO_BIOLINK = {
    "GENE":      "biolink:Gene",
    "DISEASE":   "biolink:Disease",
    "CHEMICAL":  "biolink:ChemicalEntity",
    "CELL_TYPE": "biolink:Cell",
    "ANATOMY":   "biolink:AnatomicalEntity",
}

_SESSION = requests.Session()


def biolink_type_curie(type_name: str) -> str | None:
    """'gross anatomical structure' -> 'biolink:GrossAnatomicalStructure'; None for empty input."""
    words = re.sub(r"[_\-]", " ", (type_name or "").replace("biolink:", "")).split()
    if not words:
        return None
    return "biolink:" + "".join(w[:1].upper() + w[1:] for w in words)

def _fetch_json(url, params):
    """GET a URL and return parsed JSON, or None on any error.

    A failed lookup must not kill the run, but it must be VISIBLE: a silently swallowed
    error looks identical to "no candidates", so it is logged.
    """
    try:
        response = _SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.warning("Lookup failed (%s): %s", url, e)
        return None
    

def renci_lookup(term: str, biolink_type: str | None = None) -> list[Candidate]:
    """Look up a term in the RENCI name resolver; optionally constrain to one biolink type."""
    params = {"string": term, "limit": settings.api_limit}
    if biolink_type:                          # type-constrained pass: ask for this category only
        params["biolink_type"] = biolink_type

    data = _fetch_json(settings.renci_url, params)
    if not data:
        return []

    candidates = []
    for rank, hit in enumerate(data[:settings.api_limit]):
        if not hit.get("curie"):
            continue
        types = hit.get("types", [])
        candidates.append(Candidate(
            curie=hit["curie"],
            label=hit.get("label", ""),
            biolink_type=types[0] if types else None,
            rank=rank,
            source="RENCI",
        ))
    return candidates


def arax_lookup(term: str) -> list[Candidate]:
    """Look up a term in the ARAX entity normalizer; return the canonical id plus graph nodes."""
    data = _fetch_json(settings.arax_url, {"q": term})
    entry = data.get(term) if data else None
    if not entry:
        return []

    canon = entry.get("id", {})
    nodes = entry.get("knowledge_graph", {}).get("nodes", {})

    candidates = []
    if canon.get("identifier"):
        candidates.append(Candidate(
            curie=canon["identifier"],
            label=canon.get("name", ""),
            biolink_type=canon.get("category"),
            rank=0,
            source="ARAX",
        ))
    for nid, node in nodes.items():
        if nid == canon.get("identifier"):
            continue
        if len(candidates) >= settings.api_limit:
            break
        candidates.append(Candidate(
            curie=nid,
            label=node.get("name", ""),
            biolink_type=node.get("category"),
            rank=len(candidates),
            source="ARAX",
        ))
    return candidates


def _merge_and_rank(candidate_lists: list[list[Candidate]]) -> list[Candidate]:
    """Union candidate lists by CURIE, combine sources, and assign one global rank.
    Order: candidates multiple resolvers agree on first, then by best original lookup-rank,
    then CURIE for a deterministic tie-break.
    """
    pool = {}
    for candidates in candidate_lists:
        for cand in candidates:
            if cand.curie in pool:
                existing = pool[cand.curie]
                if cand.source not in existing.source:    
                    existing.source += f"+{cand.source}"
            else:
                pool[cand.curie] = cand

    merged = sorted(
        pool.values(),
        key=lambda c: (0 if "+" in c.source else 1, c.rank, c.curie),
    )
    for i, cand in enumerate(merged):
        cand.rank = i                                       # reassign to the global position
    return merged


_EQUIV_CACHE: dict[str, str | None] | None = None


def canonicalize(curies: list[str]) -> dict[str, str | None]:
    """Map CURIEs to canonical RTX-KG2c ids with the offline equivalents table.

    Reads data/kg2c/equivalents.parquet (built by scripts/build_kg2c_parquet.py) through DuckDB,
    so no network call is needed to align resolver output with the knowledge graph. A CURIE
    that is not in the filtered graph maps to None. Results are cached per process.
    """
    global _EQUIV_CACHE
    if _EQUIV_CACHE is None:
        _EQUIV_CACHE = {}
    missing = [c for c in set(curies) if c not in _EQUIV_CACHE]
    if missing:
        table = Path(settings.kg2c_dir) / "equivalents.parquet"
        if not table.is_file():
            logger.warning("KG2c equivalents table not found at %s: no canonicalization.", table)
            for c in missing:
                _EQUIV_CACHE[c] = None
        else:
            import duckdb

            con = duckdb.connect()
            path = str(table).replace("\\", "/")
            rows = con.execute(
                f"SELECT curie, canonical_id FROM read_parquet('{path}') WHERE curie IN (SELECT UNNEST(?))",
                [missing],
            ).fetchall()
            con.close()
            found = dict(rows)
            for c in missing:
                _EQUIV_CACHE[c] = found.get(c)
        # Fallback for CURIEs absent from the local table: the Translator Node Normalizer gives
        # the clique's preferred id, then the local table says whether that id is in KG2c.
        still_missing = [c for c in missing if _EQUIV_CACHE.get(c) is None]
        if still_missing and not settings.resolvers_offline:
            preferred = _nodenorm_preferred(still_missing)
            pref_ids = sorted({p for p in preferred.values() if p})
            in_kg = {}
            if pref_ids and table.is_file():
                import duckdb

                con = duckdb.connect()
                in_kg = dict(con.execute(
                    f"SELECT curie, canonical_id FROM read_parquet('{path}') WHERE curie IN (SELECT UNNEST(?))", [pref_ids]
                ).fetchall())
                con.close()
            for c in still_missing:
                p = preferred.get(c)
                _EQUIV_CACHE[c] = in_kg.get(p) if p else None
    return {c: _EQUIV_CACHE[c] for c in curies}


def _nodenorm_preferred(curies: list[str]) -> dict[str, str | None]:
    """Preferred clique id per CURIE from the Translator Node Normalizer (conflation OFF)."""
    out: dict[str, str | None] = {}
    for i in range(0, len(curies), 50):
        batch = curies[i:i + 50]
        data = _fetch_json(settings.nodenorm_url, {"curie": batch, "conflate": "false", "drug_chemical_conflate": "false"}) or {}
        for c in batch:
            entry = data.get(c)
            out[c] = entry["id"]["identifier"] if entry else None
    return out


def build_pool(term: str, types: set[str]) -> list[Candidate]:
    """Retrieve and merge the candidate pool for one term.

    Runs the unconstrained RENCI + ARAX lookups plus one type-constrained RENCI pass per
    mapped entity type, then unions everything into one ranked pool.
    """
    candidate_lists = [renci_lookup(term), arax_lookup(term)]

    # One type-constrained NameRes pass per NER type. Types are Biolink class names as the
    # agent returns them ("gross anatomical structure"); the legacy upper-case labels are
    # still accepted.
    biolink_types = set()
    for t in types:
        if not t or t.upper() == "NONE":
            continue
        bt = TYPE_TO_BIOLINK.get(t) or biolink_type_curie(t)
        if bt:
            biolink_types.add(bt)
    for bt in sorted(biolink_types):
        candidate_lists.append(renci_lookup(term, biolink_type=bt))

    pool = _merge_and_rank(candidate_lists)
    # Attach the KG2c canonical id offline; candidates absent from the filtered graph keep None.
    canonical = canonicalize([c.curie for c in pool])
    for cand in pool:
        cand.kg2c_id = canonical.get(cand.curie)
    logger.info("Retrieval %r: %d unique candidate(s), %d in KG2c",
                term, len(pool), sum(1 for c in pool if c.kg2c_id))
    return pool


# Biolink classes that denote people or groups of people: never sent to an external service.
PERSONAL_TYPES = {"case", "individual organism", "cohort", "study population", "population of individual organisms", "agent"}
# Surface forms that look like identifiers or contact data rather than biomedical concepts.
IDENTIFIER_PATTERNS = [
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",       # dates
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",               # e-mails
    r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",             # phone numbers
    r"\b(?:MRN|NHS|SSN|patient|pt|id)\s*[:#]?\s*\d{3,}\b",  # record numbers
]


def _local_lookup(term: str, types: set[str]) -> list[Candidate]:
    """Offline resolution: exact, case-insensitive name match in the filtered KG2c node table."""
    table = Path(settings.kg2c_dir) / "nodes.parquet"
    if not table.is_file():
        return []
    import duckdb

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT id, name, category FROM read_parquet('{str(table).replace(chr(92), '/')}') WHERE lower(name) = lower(?) LIMIT ?",
        [term, settings.api_limit],
    ).fetchall()
    con.close()
    return [Candidate(curie=i, label=n, biolink_type=c, rank=k, source="KG2c-local", kg2c_id=i) for k, (i, n, c) in enumerate(rows)]


def is_shareable(term: str, types: set[str]) -> bool:
    """False when a term must not leave the machine: personal-entity types or identifier-like text."""
    if any(t.lower() in PERSONAL_TYPES for t in types):
        return False
    return not any(re.search(p, term, re.IGNORECASE) for p in IDENTIFIER_PATTERNS)


def resolve_terms(term_types: dict[str, set[str]]) -> dict[str, list[Candidate]]:
    """Resolve every unique term concurrently; return term -> ranked candidate pool.

    term_types maps each term to the set of NER types it appeared with; the types drive the
    type-constrained pass. Terms typed as people or looking like identifiers are resolved
    locally only. With settings.resolvers_offline every term is resolved locally: no text
    leaves the machine (exact-name matching against the KG2c node table).
    """
    offline = settings.resolvers_offline
    logger.info("Resolving %d unique term(s) %s...", len(term_types), "offline (KG2c node names)" if offline else "against RENCI + ARAX")

    pools = {}
    with ThreadPoolExecutor(max_workers=settings.max_concurrent_requests) as pool:
        futures = {}
        for term, types in term_types.items():
            if offline or not is_shareable(term, types):
                if not offline:
                    logger.warning("Term %r kept local (personal type or identifier-like): no external lookup", term)
                pools[term] = _local_lookup(term, types)
                continue
            futures[pool.submit(build_pool, term, types)] = term
        for future in futures:
            term = futures[future]
            pools[term] = future.result()

    total = 0
    for candidates in pools.values():
        total += len(candidates)
    logger.info("Retrieval finished: %d candidate(s) across %d term(s).", total, len(pools))
    return pools
