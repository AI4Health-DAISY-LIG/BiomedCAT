import logging
from concurrent.futures import ThreadPoolExecutor
import requests

from biomedcat.config import settings
from biomedcat.types import Candidate

logger = logging.getLogger(__name__)
# Each entity type maps to a Biolink class that is non-mixin and declares id_prefixes, so a
# resolved identifier can actually carry it. CURIEs verified against the live resolver, since
# an unrecognised biolink_type returns HTTP 200 with an empty list rather than an error.
TYPE_TO_BIOLINK = {
    "GENE":               "biolink:Gene",
    "PROTEIN":            "biolink:Protein",
    "DISEASE":            "biolink:Disease",
    "PHENOTYPIC_FEATURE": "biolink:PhenotypicFeature",
    "CHEMICAL":           "biolink:ChemicalEntity",
    "CELL_TYPE":          "biolink:Cell",
    "CELLULAR_COMPONENT": "biolink:CellularComponent",
    "ANATOMY":            "biolink:GrossAnatomicalStructure",   # concrete subclass, disjoint from the two above
    "BIOLOGICAL_PROCESS": "biolink:BiologicalProcessOrActivity", # umbrella: process and molecular activity are siblings
    "SEQUENCE_VARIANT":   "biolink:SequenceVariant",             # RENCI does not index variants; ARAX covers this type
}

_SESSION = requests.Session()

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


def build_pool(term: str, types: set[str]) -> list[Candidate]:
    """Retrieve and merge the candidate pool for one term.

    Runs the unconstrained RENCI + ARAX lookups plus one type-constrained RENCI pass per
    mapped entity type, then unions everything into one ranked pool.
    """
    candidate_lists = [renci_lookup(term), arax_lookup(term)]

    biolink_types = set()
    for t in types:
        if t in TYPE_TO_BIOLINK:
            biolink_types.add(TYPE_TO_BIOLINK[t])
    for bt in sorted(biolink_types):
        candidate_lists.append(renci_lookup(term, biolink_type=bt))

    pool = _merge_and_rank(candidate_lists)
    logger.info("Retrieval %r: %d unique candidate(s)", term, len(pool))
    return pool


def resolve_terms(term_types: dict[str, set[str]]) -> dict[str, list[Candidate]]:
    """Resolve every unique term concurrently; return term -> ranked candidate pool.

    term_types maps each term to the set of NER types it appeared with; the types drive the
    type-constrained pass. One thread per term (capped by max_concurrent_requests) overlaps
    the network waits.
    """
    logger.info("Resolving %d unique term(s) against RENCI + ARAX...", len(term_types))

    pools = {}
    with ThreadPoolExecutor(max_workers=settings.max_concurrent_requests) as pool:
        futures = {}
        for term, types in term_types.items():
            futures[pool.submit(build_pool, term, types)] = term
        for future in futures:
            term = futures[future]
            pools[term] = future.result()

    total = 0
    for candidates in pools.values():
        total += len(candidates)
    logger.info("Retrieval finished: %d candidate(s) across %d term(s).", total, len(pools))
    return pools
