#!/usr/bin/env python
"""Build the optional RTX-KG2c "extras" tables used by the console viewer and the exports.

The filtered KG (scripts/build_kg2c_parquet.py) keeps only what the context-graph stage needs.
This script streams the raw KG2c JSONL files once more and writes, next to it:

    node_details.parquet      id, description, iri, synonyms (all_names), publications
                              for every node of the filtered KG
    edge_publications.parquet subject, predicate, object, publications for the kept edges
                              that carry publications (both endpoints in the filtered KG and a
                              kept predicate)
    gene_go.parquet           gene, go, predicate for every NCBIGene -> GO edge of the FULL KG2c,
                              the annotation universe of the local GO enrichment

Memory stays below 1 GB: only the id set of the filtered KG and the kept predicates are held.

Usage:
  uv run python scripts/build_kg2c_extras.py \
      --nodes ../mechanism_of_action_learning/data/KG/kg2c-2.10.1-v1.0-nodes.jsonl.gz \
      --edges ../mechanism_of_action_learning/data/KG/kg2c-2.10.1-v1.0-edges.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("build_kg2c_extras")


def open_jsonl(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")
    logger.info("-> %s (%d rows)", path, table.num_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--kg2c-dir", default="data/kg2c")
    parser.add_argument("--max-publications", type=int, default=100, help="Cap of publications stored per node or edge.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kg = Path(args.kg2c_dir)

    ids = set(pq.read_table(kg / "nodes.parquet", columns=["id"]).column("id").to_pylist())
    predicates = set(pq.read_table(kg / "predicates.parquet", columns=["predicate"]).column("predicate").to_pylist())
    logger.info("filtered KG: %d nodes, %d predicates", len(ids), len(predicates))

    # ---- nodes ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    node_rows: list[dict] = []
    n_read = 0
    with open_jsonl(args.nodes) as f:
        for line in f:
            n_read += 1
            if n_read % 1_000_000 == 0:
                logger.info("nodes: %d read, %d kept, %.0f s", n_read, len(node_rows), time.perf_counter() - t0)
            try:
                d = json.loads(line)
            except ValueError:
                continue
            nid = d.get("id")
            if nid not in ids:
                continue
            names = [str(n) for n in (d.get("all_names") or []) if n and n != d.get("name")]
            # A few KG2c descriptions and IRIs are not strings (numbers, lists): normalize.
            node_rows.append({"id": nid, "description": str(d.get("description") or "")[:2000], "iri": str(d.get("iri") or ""),
                              "synonyms": names[:30], "publications": [str(p) for p in (d.get("publications") or [])][: args.max_publications]})
    node_schema = pa.schema([("id", pa.string()), ("description", pa.string()), ("iri", pa.string()),
                             ("synonyms", pa.list_(pa.string())), ("publications", pa.list_(pa.string()))])
    write_parquet(node_rows, node_schema, kg / "node_details.parquet")
    del node_rows

    # ---- edges ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    pub_rows: list[dict] = []
    go_rows: list[dict] = []
    n_read = 0
    with open_jsonl(args.edges) as f:
        for line in f:
            n_read += 1
            if n_read % 2_000_000 == 0:
                logger.info("edges: %d read, %d with publications, %d gene-GO, %.0f s", n_read, len(pub_rows), len(go_rows), time.perf_counter() - t0)
            try:
                d = json.loads(line)
            except ValueError:
                continue
            s, o, p = d.get("subject", ""), d.get("object", ""), d.get("predicate", "")
            if s.startswith("NCBIGene:") and o.startswith("GO:"):
                go_rows.append({"gene": s, "go": o, "predicate": p})
            pubs = d.get("publications")
            if pubs and p in predicates and s in ids and o in ids:
                pub_rows.append({"subject": s, "predicate": p, "object": o, "publications": [str(x) for x in pubs][: args.max_publications]})
    write_parquet(pub_rows, pa.schema([("subject", pa.string()), ("predicate", pa.string()), ("object", pa.string()),
                                       ("publications", pa.list_(pa.string()))]), kg / "edge_publications.parquet")
    write_parquet(go_rows, pa.schema([("gene", pa.string()), ("go", pa.string()), ("predicate", pa.string())]), kg / "gene_go.parquet")
    logger.info("done in %.0f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
