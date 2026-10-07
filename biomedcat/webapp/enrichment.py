"""Gene Ontology enrichment of the genes of a context graph.

Two backends:

* ``local``: hypergeometric test over the gene -> GO annotations present in RTX-KG2c. Uses
  data/kg2c/gene_go.parquet (every gene-GO edge of the full KG2c, built by
  scripts/build_kg2c_extras.py) when present, otherwise the gene-GO edges of the filtered
  KG (fewer annotations, mostly cellular components). Nothing leaves the machine.
* ``gprofiler``: the g:Profiler g:GOSt web service (University of Tartu), which receives the
  gene identifiers only. Refused when RESOLVERS_OFFLINE is set; the console says what is sent.

Both return the same rows: term id, name, aspect (BP/CC/MF), p-value, FDR, term size, overlap
and the overlapping genes, sorted by FDR.
"""
from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from pathlib import Path

from biomedcat.config import settings

logger = logging.getLogger(__name__)

ASPECTS = {"biolink:BiologicalProcess": "BP", "biolink:CellularComponent": "CC", "biolink:MolecularActivity": "MF",
           "biolink:Pathway": "Pathway"}
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeom(N, K, n), via log-binomials (no scipy dependency)."""
    if k <= 0:
        return 1.0
    upper = min(K, n)
    if k > upper:
        return 0.0
    log_denominator = math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)
    total = 0.0
    for x in range(k, upper + 1):
        if N - K < n - x:
            continue
        log_p = (math.lgamma(K + 1) - math.lgamma(x + 1) - math.lgamma(K - x + 1)
                 + math.lgamma(N - K + 1) - math.lgamma(n - x + 1) - math.lgamma(N - K - n + x + 1) - log_denominator)
        total += math.exp(log_p)
    return min(1.0, total)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    fdr = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        fdr[i] = min(1.0, running)
    return fdr


def _annotation_stamp(kg_dir: str) -> float:
    """Modification time of the annotation source, so a table built later refreshes the cache."""
    kg = Path(kg_dir)
    for name in ("gene_go.parquet", "edges.parquet"):
        if (kg / name).is_file():
            return (kg / name).stat().st_mtime
    return 0.0


def load_annotations(kg_dir: str) -> tuple[dict[str, set[str]], dict[str, dict]]:
    return _load_annotations(kg_dir, _annotation_stamp(kg_dir))


@lru_cache(maxsize=2)
def _load_annotations(kg_dir: str, stamp: float) -> tuple[dict[str, set[str]], dict[str, dict]]:
    """term -> genes, and term -> {name, aspect}; from gene_go.parquet or the filtered edges."""
    import duckdb

    kg = Path(kg_dir)
    con = duckdb.connect()
    if (kg / "gene_go.parquet").is_file():
        rows = con.execute(f"SELECT gene, go FROM read_parquet('{_sql_path(kg / 'gene_go.parquet')}')").fetchall()
        source = "gene_go.parquet (full RTX-KG2c gene-GO annotations)"
    elif (kg / "edges.parquet").is_file():
        rows = con.execute(f"""
            SELECT subject, object FROM read_parquet('{_sql_path(kg / 'edges.parquet')}')
            WHERE subject LIKE 'NCBIGene:%' AND object LIKE 'GO:%'""").fetchall()
        source = "filtered KG2c edges (partial GO annotations)"
    else:
        con.close()
        return {}, {}
    term_genes: dict[str, set[str]] = {}
    for gene, go in rows:
        term_genes.setdefault(go, set()).add(gene)
    names: dict[str, dict] = {}
    if term_genes and (kg / "nodes.parquet").is_file():
        con.execute("CREATE TABLE want (id VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(t,) for t in term_genes])
        for tid, name, cat in con.execute(f"SELECT n.id, n.name, n.category FROM read_parquet('{_sql_path(kg / 'nodes.parquet')}') n JOIN want USING (id)").fetchall():
            names[tid] = {"name": name, "aspect": ASPECTS.get(cat, cat.replace("biolink:", ""))}
    con.close()
    logger.info("GO annotations loaded from %s: %d terms", source, len(term_genes))
    return term_genes, names


def enrich_local(genes: list[str], kg_dir: str | None = None, min_term_size: int = 3, max_term_size: int = 2000,
                 min_overlap: int = 2, fdr_max: float = 0.25, top: int = 60) -> dict:
    t0 = time.perf_counter()
    term_genes, names = load_annotations(str(kg_dir or settings.kg2c_dir))
    universe: set[str] = set()
    for gs in term_genes.values():
        universe |= gs
    query = {g for g in genes if g in universe}
    rows = []
    for term, gs in term_genes.items():
        K = len(gs)
        if K < min_term_size or K > max_term_size:
            continue
        overlap = query & gs
        if len(overlap) < min_overlap:
            continue
        p = hypergeom_sf(len(overlap), len(universe), K, len(query))
        rows.append({"term": term, "name": names.get(term, {}).get("name", term), "aspect": names.get(term, {}).get("aspect", ""),
                     "p_value": p, "term_size": K, "overlap": len(overlap), "genes": sorted(overlap)})
    fdr = benjamini_hochberg([r["p_value"] for r in rows])
    for r, q in zip(rows, fdr):
        r["fdr"] = q
    rows = sorted((r for r in rows if r["fdr"] <= fdr_max), key=lambda r: (r["fdr"], r["p_value"]))[:top]
    for r in rows:
        r["p_value"] = float(f"{r['p_value']:.3g}")
        r["fdr"] = float(f"{r['fdr']:.3g}")
    return {"backend": "local", "n_query": len(genes), "n_query_in_universe": len(query), "universe": len(universe),
            "n_terms_tested": len(fdr), "results": rows, "seconds": round(time.perf_counter() - t0, 2),
            "note": "Hypergeometric test over the gene-GO annotations of RTX-KG2c; Benjamini-Hochberg FDR."}


def enrich_gprofiler(genes: list[str], organism: str = "hsapiens", sources: tuple[str, ...] = ("GO:BP", "GO:MF", "GO:CC"),
                     timeout: int = 60) -> dict:
    """g:GOSt query with Entrez gene ids; refused in offline mode."""
    if settings.resolvers_offline:
        return {"backend": "gprofiler", "error": "offline mode: no identifier is sent to external services", "results": []}
    import requests

    entrez = [g.split(":", 1)[1] for g in genes if g.startswith("NCBIGene:")]
    if not entrez:
        return {"backend": "gprofiler", "n_query": 0, "results": [], "note": "no NCBIGene identifier in the graph"}
    payload = {"organism": organism, "query": entrez, "sources": list(sources), "numeric_ns": "ENTREZGENE_ACC",
               "user_threshold": 0.05, "significance_threshold_method": "g_SCS", "no_evidences": False}
    t0 = time.perf_counter()
    try:
        r = requests.post(GPROFILER_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return {"backend": "gprofiler", "error": f"g:Profiler request failed: {e}", "results": []}
    query_ids = data.get("meta", {}).get("genes_metadata", {}).get("query", {}).get("query_1", {}).get("ensgs", [])
    rows = []
    for x in data.get("result", []):
        # `intersections` lists, per query gene, the evidence codes; a non-empty entry means overlap.
        inter = x.get("intersections") or []
        hit_genes = [f"NCBIGene:{entrez[i]}" for i, ev in enumerate(inter) if ev and i < len(entrez)]
        rows.append({"term": x.get("native"), "name": x.get("name"), "aspect": str(x.get("source", "")).replace("GO:", ""),
                     "p_value": float(f"{x.get('p_value', 1.0):.3g}"), "fdr": float(f"{x.get('p_value', 1.0):.3g}"),
                     "term_size": x.get("term_size"), "overlap": x.get("intersection_size"), "genes": hit_genes})
    rows.sort(key=lambda r: r["p_value"])
    return {"backend": "gprofiler", "n_query": len(entrez), "n_query_in_universe": len(query_ids) or None, "results": rows[:60],
            "seconds": round(time.perf_counter() - t0, 2), "organism": organism,
            "note": "g:Profiler g:GOSt (University of Tartu), g:SCS-corrected p-values; the gene identifiers were sent to the service."}
