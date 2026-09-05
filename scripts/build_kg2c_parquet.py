#!/usr/bin/env python
"""Build a filtered, compact Parquet copy of RTX-KG2c for the BiomedCAT context-graph stage.

The full KG2c graph (about 6.7 M nodes and 27 M edges in version 2.10.1) does not fit the
16 GB RAM budget targeted by BiomedCAT once an igraph object, an LLM server and the RAG
engine coexist. This script streams the KG2c JSONL dumps once and keeps everything that is
*structurally* usable by any preference profile, present or future:

  1. edges whose predicate, and nodes whose category, are not part of the Biolink model
     itself (--strata, derived from biolink-model.yaml by scripts/build_biolink_strata.py):
     RTX-KG2c carries a handful of predicates of its own that are not standard Biolink terms
     (a doubled "biolink_" prefix, e.g. biolink:biolink_treats) -- the graph is aligned to the
     Biolink version of --strata, not to whatever RTX-KG2c happens to emit;
  2. edges that are not identifier-equivalence links (close_match, same_as, ...), because
     KG2c is already canonicalized and those links carry no biology;
  3. edges whose predicate is too generic to ever be informative (related_to and its two
     at-instance/at-concept-level variants: 2.6 M edges for related_to alone, no biology);
  4. edges whose primary knowledge source is not in --exclude-sources (SemMedDB by default:
     literature co-occurrence, the single largest and noisiest source in RTX-KG2);
  5. nodes that still have at least one kept edge (isolated nodes are dropped implicitly).

This build does *not* know about preference profiles, and that is deliberate: which predicates,
knowledge sources and Biolink categories matter is a per-user, per-run choice, applied at query
time by the context-graph stage (biomedcat.profiles, biomedcat.weights) from the profile that is
active for that run. Baking a profile's scope into this Parquet build would tie its content to
whichever profiles happened to exist the last time this script ran: a new profile (a new class
scope built in the console, say) would silently lose the nodes and edges it needs until this
(slow, whole-dump) script is run again. One build serves every profile, current and future.

Outputs (all under --out-dir):
  edges.parquet        subject, object, predicate_code (int16), knowledge_level, agent_type,
                       primary_knowledge_source. No weight column: edge weights are a profile
                       concept, computed at query time (biomedcat.stages.context_graph).
  nodes.parquet        id, name, category, all_categories (list<string>)
  equivalents.parquet  curie -> canonical KG2c id, exploded from `equivalent_curies`.
                       This is a local, offline replacement for the Node Normalizer when
                       BiomedCAT CURIEs (e.g. HGNC:50800 for DUX4) must be mapped to KG2c ids.
  predicates.parquet   predicate_code -> predicate CURIE, edges kept (n_edges_kept).
  stats.json           counts before/after every filter, peak RSS, wall time
  summary.md           human-readable version of stats.json (for the manuscript)

Nothing here is written to git: the repository .gitignore excludes data/.

Usage (from the BiomedCAT root):
  uv run python scripts/build_kg2c_parquet.py \
      --nodes ../mechanism_of_action_learning/data/KG/kg2c-2.10.1-v1.0-nodes.jsonl.gz \
      --edges ../mechanism_of_action_learning/data/KG/kg2c-2.10.1-v1.0-edges.jsonl.gz \
      --out-dir data/kg2c
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from biomedcat.weights import BiolinkModel, camel_curie

logger = logging.getLogger("build_kg2c_parquet")

# Identifier-equivalence predicates: KG2c is canonicalized, these carry no biology.
EQUIVALENCE_PREDICATES = frozenset(
    {
        "biolink:close_match",
        "biolink:same_as",
        "biolink:exact_match",
        "biolink:broad_match",
        "biolink:narrow_match",
    }
)
# Roots of the "related to" branch: too generic to ever be informative (related_to alone is
# 2.6 M edges in KG2c 2.10.1), whatever a profile's scope. biomedcat.weights independently
# refuses to weight any predicate this shallow for a branch_weights profile (depth < 2); this
# is the same decision, applied unconditionally, including to a legacy flat profile that lists
# them explicitly (the uniform baseline does, like it lists the equivalence predicates above).
TOO_GENERAL_PREDICATES = frozenset(
    {
        "biolink:related_to",
        "biolink:related_to_at_instance_level",
        "biolink:related_to_at_concept_level",
    }
)
EDGE_CHUNK_ROWS = 1_000_000
NODE_CHUNK_ROWS = 500_000
LOG_EVERY_LINES = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", required=True, help="KG2c nodes JSONL (.gz accepted).")
    parser.add_argument("--edges", required=True, help="KG2c edges JSONL (.gz accepted).")
    parser.add_argument("--out-dir", default="data/kg2c", help="Output directory (default: data/kg2c).")
    parser.add_argument(
        "--exclude-sources",
        default="infores:semmeddb",
        help="Comma-separated primary_knowledge_source values to drop (default: infores:semmeddb).",
    )
    parser.add_argument(
        "--strata", default="data/biolink_strata.json",
        help="Derived Biolink model (scripts/build_biolink_strata.py), used to keep only predicates and "
             "categories that are actually part of that Biolink version (default: data/biolink_strata.json).",
    )
    parser.add_argument("--kg-version", default="kg2c-2.10.1-v1.0", help="Label stored in stats.json.")
    return parser.parse_args()


def open_text(path: str):
    """Open a plain or gzip JSONL file for line iteration."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def rss_gb(process: psutil.Process) -> float:
    return process.memory_info().rss / 1024**3


def peak_rss_gb(process: psutil.Process) -> float | None:
    """Peak resident set size when the platform reports it (Windows: peak_wset)."""
    info = process.memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 1024**3
    try:
        import resource  # POSIX only

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return peak_kb / 1024**2 if platform.system() == "Linux" else peak_kb / 1024**3
    except ImportError:
        return None


def load_biolink_model_terms(strata_path: str) -> tuple[set[str], set[str], str]:
    """(valid predicate CURIEs, valid category CURIEs, Biolink version) from the derived strata file.

    A predicate is valid when it is a real Biolink slot: a `kg2c_only` entry (present in some
    earlier KG2c build but absent from biolink-model.yaml, e.g. the doubled-prefix
    biolink:biolink_treats -- see scripts/build_biolink_strata.py) does not count. A category is
    valid when it names an entity class *or* a mixin: RTX-KG2c uses some mixins (e.g.
    biolink:GenomicEntity) as a node's primary category, even though biomedcat.weights keeps
    mixins out of the entity-branch scheme used to derive profile weights.
    """
    model = BiolinkModel.load(strata_path)
    valid_predicates = {info["curie"] for info in model.predicates.values() if not info.get("kg2c_only")}
    valid_categories = {camel_curie(name) for name in model.classes} | {camel_curie(name) for name in model.mixins}
    return valid_predicates, valid_categories, model.version


# ----------------------------------------------------------------------------------------
# Pass 1: edges
# ----------------------------------------------------------------------------------------

EDGE_SCHEMA = pa.schema(
    [
        ("subject", pa.string()),
        ("object", pa.string()),
        ("predicate_code", pa.int16()),
        ("knowledge_level", pa.dictionary(pa.int8(), pa.string())),
        ("agent_type", pa.dictionary(pa.int8(), pa.string())),
        ("primary_knowledge_source", pa.dictionary(pa.int16(), pa.string())),
    ]
)


class EdgeWriter:
    """Buffer edge rows and flush them as Parquet row groups of EDGE_CHUNK_ROWS."""

    def __init__(self, path: Path):
        self.writer = pq.ParquetWriter(str(path), EDGE_SCHEMA, compression="zstd")
        self.reset()

    def reset(self) -> None:
        self.subject: list[str] = []
        self.object: list[str] = []
        self.code: list[int] = []
        self.kl: list[str] = []
        self.agent: list[str] = []
        self.source: list[str] = []

    def add(self, s: str, o: str, code: int, kl: str, agent: str, src: str) -> None:
        self.subject.append(s)
        self.object.append(o)
        self.code.append(code)
        self.kl.append(kl)
        self.agent.append(agent)
        self.source.append(src)
        if len(self.subject) >= EDGE_CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.subject:
            return
        table = pa.table(
            {
                "subject": pa.array(self.subject, pa.string()),
                "object": pa.array(self.object, pa.string()),
                "predicate_code": pa.array(self.code, pa.int16()),
                "knowledge_level": pa.array(self.kl, pa.string()).dictionary_encode(),
                "agent_type": pa.array(self.agent, pa.string()).dictionary_encode(),
                "primary_knowledge_source": pa.array(self.source, pa.string()).dictionary_encode(),
            },
            schema=EDGE_SCHEMA,
        )
        self.writer.write_table(table)
        self.reset()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def pass_edges(edges_path: str, out_dir: Path, excluded_sources: set[str], valid_predicates: set[str],
               process: psutil.Process) -> tuple[set[str], dict]:
    """Stream the edge dump once; write every structurally-kept edge; return the kept node ids.

    Predicate codes are assigned on the fly, in first-seen order: unlike the old profile-driven
    build, the set of predicates that will survive filtering is not known ahead of the pass.
    """
    kept_nodes: set[str] = set()
    predicate_codes: dict[str, int] = {}
    n_total = 0
    n_kept = 0
    dropped_reason = Counter()
    pred_total = Counter()
    pred_kept = Counter()
    source_dropped = Counter()
    kl_kept = Counter()

    writer = EdgeWriter(out_dir / "edges.parquet")
    t0 = time.perf_counter()

    with open_text(edges_path) as f:
        for line in f:
            n_total += 1
            if n_total % LOG_EVERY_LINES == 0:
                logger.info(
                    "edges: %d read, %d kept (%.1f%%), rss %.2f GB, %.0f s",
                    n_total, n_kept, 100.0 * n_kept / n_total, rss_gb(process), time.perf_counter() - t0,
                )
            line = line.strip()
            if not line:
                continue
            edge = json.loads(line)
            pred = edge.get("predicate", "")
            pred_total[pred] += 1

            # Filter order matters only for the "reason" accounting; all filters are ANDed.
            if pred in EQUIVALENCE_PREDICATES:
                dropped_reason["equivalence_predicate"] += 1
                continue
            if pred in TOO_GENERAL_PREDICATES:
                dropped_reason["too_general_predicate"] += 1
                continue
            if pred not in valid_predicates:
                dropped_reason["predicate_not_in_biolink_model"] += 1
                continue
            source = edge.get("primary_knowledge_source", "") or ""
            if source in excluded_sources:
                dropped_reason["excluded_source"] += 1
                source_dropped[source] += 1
                continue

            s = edge["subject"]
            o = edge["object"]
            kl = edge.get("knowledge_level", "") or ""
            agent = edge.get("agent_type", "") or ""
            code = predicate_codes.setdefault(pred, len(predicate_codes))
            writer.add(s, o, code, kl, agent, source)
            kept_nodes.add(s)
            kept_nodes.add(o)
            n_kept += 1
            pred_kept[pred] += 1
            kl_kept[kl] += 1

    writer.close()
    elapsed = time.perf_counter() - t0
    logger.info("edges done: %d read, %d kept, %d nodes touched, %d distinct predicates kept, %.0f s",
                n_total, n_kept, len(kept_nodes), len(predicate_codes), elapsed)

    # Predicate code table, also written as Parquet for DuckDB joins. No weight column: a
    # predicate's weight is a profile concept, computed at query time from branch priorities
    # (biomedcat.weights) or from an explicit profile predicates block, never stored here.
    preds_sorted = sorted(predicate_codes, key=lambda p: predicate_codes[p])
    pq.write_table(
        pa.table(
            {
                "predicate_code": pa.array([predicate_codes[p] for p in preds_sorted], pa.int16()),
                "predicate": pa.array(preds_sorted, pa.string()),
                "n_edges_kept": pa.array([pred_kept.get(p, 0) for p in preds_sorted], pa.int64()),
            }
        ),
        str(out_dir / "predicates.parquet"),
    )

    stats = {
        "edges_total": n_total,
        "edges_kept": n_kept,
        "edges_dropped_by_reason": dict(dropped_reason),
        "edges_dropped_by_excluded_source": dict(source_dropped),
        "edges_kept_by_predicate": dict(pred_kept.most_common()),
        "edges_total_by_predicate": dict(pred_total.most_common()),
        "edges_kept_by_knowledge_level": dict(kl_kept.most_common()),
        "edge_pass_seconds": round(elapsed, 1),
    }
    return kept_nodes, stats


# ----------------------------------------------------------------------------------------
# Pass 2: nodes + equivalent CURIEs
# ----------------------------------------------------------------------------------------

NODE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("category", pa.dictionary(pa.int16(), pa.string())),
        ("all_categories", pa.list_(pa.string())),
    ]
)

EQUIV_SCHEMA = pa.schema([("curie", pa.string()), ("canonical_id", pa.string())])


def pass_nodes(nodes_path: str, out_dir: Path, kept_nodes: set[str], valid_categories: set[str],
               process: psutil.Process) -> dict:
    """Stream the node dump once; keep nodes with >= 1 kept edge and a Biolink-valid category;
    explode equivalent CURIEs."""
    node_writer = pq.ParquetWriter(str(out_dir / "nodes.parquet"), NODE_SCHEMA, compression="zstd")
    equiv_writer = pq.ParquetWriter(str(out_dir / "equivalents.parquet"), EQUIV_SCHEMA, compression="zstd")

    ids: list[str] = []
    names: list[str] = []
    cats: list[str] = []
    allcats: list[list[str]] = []
    eq_curie: list[str] = []
    eq_canon: list[str] = []

    n_total = 0
    n_kept = 0
    n_equiv = 0
    cat_total = Counter()
    cat_kept = Counter()
    cat_dropped_off_model = Counter()
    t0 = time.perf_counter()

    def flush_nodes() -> None:
        nonlocal ids, names, cats, allcats
        if not ids:
            return
        node_writer.write_table(
            pa.table(
                {
                    "id": pa.array(ids, pa.string()),
                    "name": pa.array(names, pa.string()),
                    "category": pa.array(cats, pa.string()).dictionary_encode(),
                    "all_categories": pa.array(allcats, pa.list_(pa.string())),
                },
                schema=NODE_SCHEMA,
            )
        )
        ids, names, cats, allcats = [], [], [], []

    def flush_equiv() -> None:
        nonlocal eq_curie, eq_canon
        if not eq_curie:
            return
        equiv_writer.write_table(
            pa.table({"curie": pa.array(eq_curie, pa.string()), "canonical_id": pa.array(eq_canon, pa.string())}, schema=EQUIV_SCHEMA)
        )
        eq_curie, eq_canon = [], []

    with open_text(nodes_path) as f:
        for line in f:
            n_total += 1
            if n_total % LOG_EVERY_LINES == 0:
                logger.info("nodes: %d read, %d kept, rss %.2f GB", n_total, n_kept, rss_gb(process))
            line = line.strip()
            if not line:
                continue
            node = json.loads(line)
            node_id = node.get("id", "")
            category = node.get("category", "") or ""
            cat_total[category] += 1
            if node_id not in kept_nodes:
                continue
            if category not in valid_categories:
                cat_dropped_off_model[category] += 1
                continue

            n_kept += 1
            cat_kept[category] += 1
            ids.append(node_id)
            names.append(str(node.get("name", "") or ""))
            cats.append(category)
            allcats.append([str(c) for c in (node.get("all_categories") or [])])
            if len(ids) >= NODE_CHUNK_ROWS:
                flush_nodes()

            for curie in node.get("equivalent_curies") or []:
                eq_curie.append(str(curie))
                eq_canon.append(node_id)
                n_equiv += 1
            if len(eq_curie) >= EDGE_CHUNK_ROWS:
                flush_equiv()

    flush_nodes()
    flush_equiv()
    node_writer.close()
    equiv_writer.close()
    elapsed = time.perf_counter() - t0
    logger.info("nodes done: %d read, %d kept, %d equivalent CURIEs, %.0f s", n_total, n_kept, n_equiv, elapsed)

    return {
        "nodes_total": n_total,
        "nodes_kept": n_kept,
        "nodes_dropped_category_not_in_biolink_model": dict(cat_dropped_off_model.most_common()),
        "equivalent_curies_written": n_equiv,
        "nodes_kept_by_category": dict(cat_kept.most_common()),
        "nodes_total_by_category": dict(cat_total.most_common()),
        "node_pass_seconds": round(elapsed, 1),
    }


# ----------------------------------------------------------------------------------------
# Post-hoc structural statistics (DuckDB over the Parquet files; no graph object needed)
# ----------------------------------------------------------------------------------------

def degree_stats(out_dir: Path) -> dict:
    """Degree distribution and hub counts, computed out-of-core with DuckDB."""
    import duckdb

    con = duckdb.connect()
    edges = str(out_dir / "edges.parquet").replace("\\", "/")
    deg_cte = f"""
        WITH deg AS (
            SELECT node, SUM(n) AS degree FROM (
                SELECT subject AS node, COUNT(*) AS n FROM read_parquet('{edges}') GROUP BY subject
                UNION ALL
                SELECT object AS node, COUNT(*) AS n FROM read_parquet('{edges}') GROUP BY object
            ) GROUP BY node
        )
    """
    row = con.execute(
        deg_cte
        + """
        SELECT COUNT(*), MAX(degree), AVG(degree), quantile_cont(degree, 0.5), quantile_cont(degree, 0.99),
               SUM(CASE WHEN degree > 500 THEN 1 ELSE 0 END), SUM(CASE WHEN degree > 5000 THEN 1 ELSE 0 END)
        FROM deg
        """
    ).fetchone()
    if not row or not row[0]:
        con.close()
        logger.warning("no kept edges: degree statistics skipped")
        return {"n_nodes_with_edges": 0}
    top = con.execute(deg_cte + " SELECT node, degree FROM deg ORDER BY degree DESC LIMIT 20").fetchall()
    con.close()
    return {
        "n_nodes_with_edges": int(row[0]),
        "max_degree": int(row[1]),
        "mean_degree": round(float(row[2]), 2),
        "median_degree": float(row[3]),
        "p99_degree": float(row[4]),
        "hubs_over_500": int(row[5]),
        "hubs_over_5000": int(row[6]),
        "top20_hubs": [{"node": n, "degree": int(d)} for n, d in top],
    }


def write_summary(stats: dict, out_dir: Path) -> None:
    """Markdown summary with the numbers the manuscript needs (before/after each filter)."""
    e = stats["edges"]
    n = stats["nodes"]
    d = stats.get("degrees", {})
    lines = [
        f"# KG2c filtered build ({stats['kg_version']}, aligned to Biolink {stats.get('biolink_version', '?')})",
        "",
        f"Structural filters only, no profile: {stats.get('n_valid_biolink_predicates', '?')} valid predicate(s) and "
        f"{stats.get('n_valid_biolink_categories', '?')} valid categor(y/ies) per Biolink "
        f"{stats.get('biolink_version', '?')}; {len(stats['equivalence_predicates_dropped'])} equivalence predicate(s); "
        f"{len(stats['too_general_predicates_dropped'])} too-general predicate(s); excluded sources: "
        f"{', '.join(stats['excluded_sources'])}. Profile-specific weights and class scope are applied at query "
        "time (biomedcat.stages.context_graph).",
        "",
        "| Quantity | Before | After | Kept |",
        "|---|---:|---:|---:|",
        f"| Edges | {e['edges_total']:,} | {e['edges_kept']:,} | {100.0 * e['edges_kept'] / e['edges_total']:.1f}% |",
        f"| Nodes | {n['nodes_total']:,} | {n['nodes_kept']:,} | {100.0 * n['nodes_kept'] / n['nodes_total']:.1f}% |",
        "",
        "Edges dropped by reason:",
        "",
    ]
    for reason, count in e["edges_dropped_by_reason"].items():
        lines.append(f"- {reason}: {count:,}")
    lines += ["", "Top kept predicates:", ""]
    for pred, count in list(e["edges_kept_by_predicate"].items())[:15]:
        lines.append(f"- {pred}: {count:,}")
    lines += ["", "Top kept node categories:", ""]
    for cat, count in list(n["nodes_kept_by_category"].items())[:15]:
        lines.append(f"- {cat}: {count:,}")
    if d.get("n_nodes_with_edges"):
        lines += [
            "",
            "Degree distribution (undirected degree over kept edges):",
            "",
            f"- mean {d['mean_degree']}, median {d['median_degree']}, p99 {d['p99_degree']}, max {d['max_degree']:,}",
            f"- hubs with degree > 500: {d['hubs_over_500']:,}; > 5000: {d['hubs_over_5000']:,}",
        ]
    lines += [
        "",
        "Resources:",
        "",
        f"- peak RSS of this build: {stats['peak_rss_gb']} GB",
        f"- wall time: {stats['wall_seconds']} s",
        f"- Parquet sizes (MB): {stats['parquet_sizes_mb']}",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    if not Path(args.strata).is_file():
        logger.error("%s not found: run scripts/build_biolink_strata.py first (it derives this file from "
                     "data/biolink-model.yaml), so the build can be aligned to the Biolink model.", args.strata)
        return 1
    valid_predicates, valid_categories, biolink_version = load_biolink_model_terms(args.strata)
    logger.info("aligned to Biolink %s: %d valid predicate(s), %d valid categor(y/ies)",
                biolink_version, len(valid_predicates), len(valid_categories))

    excluded_sources = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}

    kept_nodes, edge_stats = pass_edges(args.edges, out_dir, excluded_sources, valid_predicates, process)
    node_stats = pass_nodes(args.nodes, out_dir, kept_nodes, valid_categories, process)
    del kept_nodes

    # A stale degree table from a previous build must not survive a rebuild.
    stale = out_dir / "degrees.parquet"
    if stale.is_file():
        stale.unlink()

    logger.info("computing degree statistics with DuckDB...")
    deg_stats = degree_stats(out_dir)

    sizes = {p.name: round(p.stat().st_size / 1024**2, 1) for p in out_dir.glob("*.parquet")}
    stats = {
        "kg_version": args.kg_version,
        "biolink_version": biolink_version,
        "excluded_sources": sorted(excluded_sources),
        "equivalence_predicates_dropped": sorted(EQUIVALENCE_PREDICATES),
        "too_general_predicates_dropped": sorted(TOO_GENERAL_PREDICATES),
        "n_valid_biolink_predicates": len(valid_predicates),
        "n_valid_biolink_categories": len(valid_categories),
        "edges": edge_stats,
        "nodes": node_stats,
        "degrees": deg_stats,
        "parquet_sizes_mb": sizes,
        "peak_rss_gb": round(peak_rss_gb(process) or 0.0, 2),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_summary(stats, out_dir)
    logger.info("done in %.0f s, peak RSS %.2f GB; outputs in %s", stats["wall_seconds"], stats["peak_rss_gb"], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
