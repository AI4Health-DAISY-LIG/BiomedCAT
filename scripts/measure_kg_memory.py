#!/usr/bin/env python
"""Measure the RAM cost of the filtered KG2c graph for the manuscript's resource table.

Two numbers are reported, both as resident set size (RSS) deltas of this process:

  1. igraph in memory: the filtered graph loaded from Parquet into a python-igraph object
     with the attributes the context-graph stage needs (id, name, category on vertices;
     predicate_code and weight on edges). This is the upper bound: the whole filtered
     graph resident at once.
  2. DuckDB out-of-core: a bounded k-hop neighborhood query around a seed set executed
     directly on the Parquet files, without a graph object. This is what the context-graph
     stage actually does; the graph never has to be fully resident.

The seed set defaults to the FSHD case study: DUX4 (NCBIGene:100288687) and FSHD1
(MONDO:0008030). Hubs above --hub-cap are not expanded, the same safeguard the context-graph
stage uses.

Usage (from the BiomedCAT root):
  uv run python scripts/measure_kg_memory.py --kg-dir data/kg2c --hops 2 --hub-cap 500
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil

logger = logging.getLogger("measure_kg_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kg-dir", default="data/kg2c", help="Directory produced by build_kg2c_parquet.py.")
    parser.add_argument("--seeds", default="NCBIGene:100288687,MONDO:0008030", help="Comma-separated seed CURIEs.")
    parser.add_argument("--hops", type=int, default=2, help="Neighborhood radius (default 2).")
    parser.add_argument("--hub-cap", type=int, default=500, help="Do not expand nodes with degree above this.")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Ignore edges below this weight during expansion.")
    parser.add_argument("--skip-igraph", action="store_true", help="Only run the DuckDB measurement.")
    parser.add_argument("--save-pickle", default=None, help="Optional path to save the igraph object.")
    parser.add_argument("--out", default=None, help="Optional JSON path for the measurements (default: <kg-dir>/memory.json).")
    return parser.parse_args()


def rss_gb(process: psutil.Process) -> float:
    return process.memory_info().rss / 1024**3


def peak_rss_gb(process: psutil.Process) -> float | None:
    info = process.memory_info()
    return info.peak_wset / 1024**3 if hasattr(info, "peak_wset") else None


# ----------------------------------------------------------------------------------------
# 1. Full filtered graph resident in igraph
# ----------------------------------------------------------------------------------------

def measure_igraph(kg_dir: Path, process: psutil.Process, save_pickle: str | None) -> dict:
    import igraph as ig
    import pyarrow.parquet as pq

    gc.collect()
    rss_before = rss_gb(process)
    t0 = time.perf_counter()

    nodes = pq.read_table(kg_dir / "nodes.parquet", columns=["id", "name", "category"])
    ids = nodes.column("id").to_pylist()
    index = {node_id: i for i, node_id in enumerate(ids)}

    edges = pq.read_table(kg_dir / "edges.parquet", columns=["subject", "object", "predicate_code", "weight"])
    src = [index[s] for s in edges.column("subject").to_pylist()]
    dst = [index[o] for o in edges.column("object").to_pylist()]

    graph = ig.Graph(n=len(ids), edges=list(zip(src, dst)), directed=True)
    del src, dst
    graph.vs["biolink_id"] = ids
    graph.vs["name"] = nodes.column("name").to_pylist()
    graph.vs["category"] = nodes.column("category").to_pylist()
    graph.es["predicate_code"] = edges.column("predicate_code").to_pylist()
    graph.es["weight"] = edges.column("weight").to_pylist()
    del edges, nodes, index
    gc.collect()

    rss_after = rss_gb(process)
    elapsed = time.perf_counter() - t0
    result = {
        "vcount": graph.vcount(),
        "ecount": graph.ecount(),
        "rss_before_gb": round(rss_before, 2),
        "rss_after_gb": round(rss_after, 2),
        "rss_delta_gb": round(rss_after - rss_before, 2),
        "peak_rss_gb": round(peak_rss_gb(process) or 0.0, 2),
        "load_seconds": round(elapsed, 1),
    }
    logger.info("igraph: %d vertices, %d edges, +%.2f GB RSS, peak %.2f GB, %.0f s",
                result["vcount"], result["ecount"], result["rss_delta_gb"], result["peak_rss_gb"], elapsed)

    if save_pickle:
        graph.write_pickle(save_pickle)
        logger.info("igraph pickle written to %s (%.1f MB)", save_pickle, os.path.getsize(save_pickle) / 1024**2)

    del graph
    gc.collect()
    return result


# ----------------------------------------------------------------------------------------
# 2. Out-of-core bounded neighborhood with DuckDB
# ----------------------------------------------------------------------------------------

def measure_duckdb_neighborhood(
    kg_dir: Path, seeds: list[str], hops: int, hub_cap: int, min_weight: float, process: psutil.Process
) -> dict:
    """Undirected k-hop expansion from the seeds, skipping hubs, entirely inside DuckDB."""
    import duckdb

    gc.collect()
    rss_before = rss_gb(process)
    t0 = time.perf_counter()

    edges = str(kg_dir / "edges.parquet").replace("\\", "/")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW e AS SELECT subject, object, weight FROM read_parquet('{edges}') WHERE weight >= {min_weight}")
    con.execute(
        """
        CREATE TABLE deg AS
        SELECT node, SUM(n) AS degree FROM (
            SELECT subject AS node, COUNT(*) AS n FROM e GROUP BY subject
            UNION ALL
            SELECT object AS node, COUNT(*) AS n FROM e GROUP BY object
        ) GROUP BY node
        """
    )
    con.execute("CREATE TABLE visited (node VARCHAR PRIMARY KEY, hop INTEGER)")
    con.execute("CREATE TABLE frontier (node VARCHAR)")
    con.executemany("INSERT INTO visited VALUES (?, 0)", [(s,) for s in seeds])
    con.executemany("INSERT INTO frontier VALUES (?)", [(s,) for s in seeds])

    missing = [s for s in seeds if con.execute("SELECT COUNT(*) FROM deg WHERE node = ?", [s]).fetchone()[0] == 0]
    if missing:
        logger.warning("seeds absent from the filtered graph: %s", missing)

    per_hop = []
    for hop in range(1, hops + 1):
        con.execute(
            f"""
            CREATE OR REPLACE TABLE next_frontier AS
            WITH expandable AS (
                SELECT f.node FROM frontier f JOIN deg d ON d.node = f.node WHERE d.degree <= {hub_cap}
            ),
            neighbors AS (
                SELECT e.object AS node FROM e JOIN expandable x ON e.subject = x.node
                UNION
                SELECT e.subject AS node FROM e JOIN expandable x ON e.object = x.node
            )
            SELECT DISTINCT n.node FROM neighbors n
            WHERE n.node NOT IN (SELECT node FROM visited)
            """
        )
        n_new = con.execute("SELECT COUNT(*) FROM next_frontier").fetchone()[0]
        con.execute(f"INSERT INTO visited SELECT node, {hop} FROM next_frontier")
        con.execute("DELETE FROM frontier")
        con.execute("INSERT INTO frontier SELECT node FROM next_frontier")
        per_hop.append({"hop": hop, "new_nodes": int(n_new)})
        logger.info("hop %d: +%d nodes (rss %.2f GB)", hop, n_new, rss_gb(process))
        if n_new == 0:
            break

    n_nodes = con.execute("SELECT COUNT(*) FROM visited").fetchone()[0]
    n_edges = con.execute(
        "SELECT COUNT(*) FROM e WHERE subject IN (SELECT node FROM visited) AND object IN (SELECT node FROM visited)"
    ).fetchone()[0]
    con.close()

    rss_after = rss_gb(process)
    elapsed = time.perf_counter() - t0
    result = {
        "seeds": seeds,
        "missing_seeds": missing,
        "hops": hops,
        "hub_cap": hub_cap,
        "min_weight": min_weight,
        "per_hop": per_hop,
        "neighborhood_nodes": int(n_nodes),
        "neighborhood_edges": int(n_edges),
        "rss_before_gb": round(rss_before, 2),
        "rss_after_gb": round(rss_after, 2),
        "rss_delta_gb": round(rss_after - rss_before, 2),
        "peak_rss_gb": round(peak_rss_gb(process) or 0.0, 2),
        "query_seconds": round(elapsed, 1),
    }
    logger.info("duckdb %d-hop: %d nodes, %d edges, +%.2f GB RSS, %.1f s",
                hops, n_nodes, n_edges, result["rss_delta_gb"], elapsed)
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    kg_dir = Path(args.kg_dir)
    process = psutil.Process(os.getpid())
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    results = {"kg_dir": str(kg_dir), "baseline_rss_gb": round(rss_gb(process), 2)}
    # DuckDB first: it must be measured from a clean process, not after igraph freed memory.
    results["duckdb_neighborhood"] = measure_duckdb_neighborhood(
        kg_dir, seeds, args.hops, args.hub_cap, args.min_weight, process
    )
    if not args.skip_igraph:
        results["igraph_full_graph"] = measure_igraph(kg_dir, process, args.save_pickle)

    out = Path(args.out) if args.out else kg_dir / "memory.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("measurements written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
