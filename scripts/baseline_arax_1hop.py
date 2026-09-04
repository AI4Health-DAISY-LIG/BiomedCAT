#!/usr/bin/env python
"""Baseline: the one-hop neighborhood of the seeds returned by the ARAX Translator service.

This is what a researcher gets today from the Translator ecosystem with the same seeds and no
local tool: one TRAPI query per seed (seed -> any node, any predicate), results merged. The
same presence metrics as the context-graph stage are reported so the two can be compared:
size, category composition, nodes of interest reached, vocabulary expansion, and time.

The ARAX server applies its own result limits and its own KG2 version (reported in the
response when available); both are stored with the output.

Usage (from the BiomedCAT root):
  uv run python scripts/baseline_arax_1hop.py --result-json output/FSHD_BiomedCAT.json \
      --out data/context_graph/FSHD_arax_1hop --find CHEBI:131167,NCBIGene:1432
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import requests

logger = logging.getLogger("baseline_arax_1hop")


def seeds_from_result(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted({e["kg2c_id"] for e in payload.get("entities", []) if e.get("kg2c_id")})


def one_hop(url: str, seed: str, timeout: int) -> dict:
    query = {
        "message": {
            "query_graph": {
                "nodes": {"n0": {"ids": [seed]}, "n1": {"categories": ["biolink:NamedThing"]}},
                "edges": {"e01": {"subject": "n0", "object": "n1"}},
            }
        },
        "submitter": "BiomedCAT baseline",
    }
    r = requests.post(url, json=query, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-json", help="BiomedCAT output JSON; seeds are its entities' kg2c_id values.")
    parser.add_argument("--seeds", help="Comma-separated KG2c ids instead of --result-json.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="https://kg2cploverdb.transltr.io/query", help="TRAPI query endpoint; default: the RTX-KG2c knowledge provider (Plover). ARAX: https://arax.ncats.io/api/arax/v1.4/query")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--find", default="", help="Comma-separated ids whose presence is reported.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seeds = seeds_from_result(Path(args.result_json)) if args.result_json else [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("no seeds")
    find_ids = [s.strip() for s in args.find.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    per_seed = []
    kg_versions = Counter()
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds, start=1):
        t1 = time.perf_counter()
        try:
            resp = one_hop(args.url, seed, args.timeout)
        except requests.RequestException as e:
            logger.warning("[%d/%d] %s failed: %s", i, len(seeds), seed, str(e)[:120])
            per_seed.append({"seed": seed, "nodes": 0, "edges": 0, "seconds": round(time.perf_counter() - t1, 1), "error": str(e)[:120]})
            continue
        kg = resp.get("message", {}).get("knowledge_graph", {}) or {}
        n_nodes, n_edges = len(kg.get("nodes", {})), len(kg.get("edges", {}))
        for nid, node in (kg.get("nodes") or {}).items():
            nodes.setdefault(nid, {"id": nid, "name": node.get("name", ""), "category": (node.get("categories") or [""])[0], "seeds": set()})
            nodes[nid]["seeds"].add(seed)
        for eid, edge in (kg.get("edges") or {}).items():
            edges.setdefault(eid, {"subject": edge.get("subject"), "predicate": edge.get("predicate"), "object": edge.get("object"),
                                   "source": next((a.get("value") for a in (edge.get("sources") or []) if isinstance(a, dict) and a.get("resource_role") == "primary_knowledge_source"), "")})
        for log in resp.get("logs", []) or []:
            msg = str(log.get("message", ""))
            if "KG2" in msg and "version" in msg.lower():
                kg_versions[msg[:120]] += 1
        per_seed.append({"seed": seed, "nodes": n_nodes, "edges": n_edges, "seconds": round(time.perf_counter() - t1, 1)})
        logger.info("[%d/%d] %s: %d nodes, %d edges, %.0f s", i, len(seeds), seed, n_nodes, n_edges, time.perf_counter() - t1)

    with open(out_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "category", "coverage", "seeds", "is_seed"])
        for n in nodes.values():
            w.writerow([n["id"], n["name"], n["category"], len(n["seeds"]), ";".join(sorted(n["seeds"])), int(n["id"] in seeds)])
    with open(out_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "primary_knowledge_source"])
        for e in edges.values():
            w.writerow([e["subject"], e["predicate"], e["object"], e["source"]])

    categories = Counter(n["category"] for n in nodes.values() if n["id"] not in seeds)
    summary = {
        "service": args.url,
        "n_seeds": len(seeds),
        "n_seeds_answered": sum(1 for p in per_seed if not p.get("error")),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "vocabulary_expansion": round(len(nodes) / len(seeds), 2),
        "categories": dict(categories.most_common()),
        "shared_nodes": sum(1 for n in nodes.values() if len(n["seeds"]) > 1 and n["id"] not in seeds),
        "found": {fid: (fid in nodes) for fid in find_ids},
        "kg_version_messages": dict(kg_versions),
        "per_seed": per_seed,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("ARAX 1-hop: %d nodes, %d edges from %d seeds in %.0f s -> %s", len(nodes), len(edges), len(seeds), summary["seconds"], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
