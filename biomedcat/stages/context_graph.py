"""Stage 4 (context graph): bounded, user-defined neighborhood of RTX-KG2c around the slide entities.

The seeds are the RTX-KG2c ids of the entities linked by the normalization stage. The stage
expands them hop by hop over the filtered KG2c Parquet files (scripts/build_kg2c_parquet.py),
entirely inside DuckDB, so the graph is never fully resident in memory. Every retained node
carries its provenance: which seeds reach it, hence which slides, and at what distance.

Scoring (explainable, no random walk):

    path_weight(s -> v)  = product of predicate weights along the path
                           / product over intermediate nodes u of log2(2 + degree(u))
    score(v)             = sum over seeds s of best path_weight(s -> v)
                           / log2(2 + degree(v)) ** alpha

The predicate weights come from the user preference profile. Intermediate nodes are penalized
by their degree, so a path that goes through a generic hub ("cancer", actin, a housekeeping
protein) contributes little: the same evidence reached through a specific node counts more.
The target node itself is penalized more mildly (alpha, default 0.5), so that a well-connected
but relevant node is not ranked below an obscure singleton. Nodes above `hub_cap` are reached
but never expanded.

Optional pharmacology restriction: edges that touch a chemical node are kept only when their
primary knowledge source is a pharmacology database (ChEMBL, DGIdb, DrugCentral, DrugBank).
This removes metabolite and cofactor edges (ATP, water, copper...) that otherwise crowd the
compound ranking.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from biomedcat.config import settings

logger = logging.getLogger(__name__)

CHEMICAL_CATEGORIES = (
    "biolink:Drug",
    "biolink:SmallMolecule",
    "biolink:ChemicalEntity",
    "biolink:MolecularMixture",
    "biolink:ChemicalMixture",
)

PHARMACOLOGY_SOURCES = (
    "infores:chembl",
    "infores:dgidb",
    "infores:drugcentral",
    "infores:drugbank",
)


@dataclass
class Seed:
    """One context-graph seed: a KG2c id with the slides and mentions it came from."""

    kg2c_id: str
    label: str = ""
    pages: list[int] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class ContextGraphParams:
    hops: int = 2
    hub_cap: int = 500
    min_edge_weight: float = 0.0
    top_k_nodes: int = 2000
    pharmacology_only: bool = False
    exclude_prediction_edges: bool = False
    # Exponent of the target node's own degree penalty (0 = none, 1 = full log-degree).
    alpha: float = 0.5


@dataclass
class ContextGraphResult:
    out_dir: str
    n_seeds: int
    n_seeds_in_graph: int
    n_nodes: int
    n_edges: int
    per_hop: list[dict]
    top_compounds: list[dict]
    params: dict
    seconds: float
    # Ranks of requested nodes (evaluation hook), keyed by KG2c id.
    ranks: dict = field(default_factory=dict)


def seeds_from_normalized(entities: Iterable) -> list[Seed]:
    """Group normalized entities (with kg2c_id) into seeds, keeping slide and mention provenance."""
    seeds: dict[str, Seed] = {}
    for e in entities:
        kg2c_id = getattr(e, "kg2c_id", None)
        if not kg2c_id:
            continue
        seed = seeds.setdefault(kg2c_id, Seed(kg2c_id=kg2c_id, label=getattr(e, "label", "") or ""))
        if getattr(e, "page", None) is not None and e.page not in seed.pages:
            seed.pages.append(e.page)
        if e.text not in seed.mentions:
            seed.mentions.append(e.text)
    return list(seeds.values())


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _ensure_degrees(con, kg_dir: Path) -> None:
    """Materialize the undirected degree of every node once, next to the Parquet files."""
    degrees = kg_dir / "degrees.parquet"
    if not degrees.is_file():
        logger.info("computing node degrees into %s", degrees)
        con.execute(
            f"""
            COPY (
                SELECT node, SUM(n)::BIGINT AS degree FROM (
                    SELECT subject AS node, COUNT(*) AS n FROM read_parquet('{_sql_path(kg_dir / "edges.parquet")}') GROUP BY subject
                    UNION ALL
                    SELECT object AS node, COUNT(*) AS n FROM read_parquet('{_sql_path(kg_dir / "edges.parquet")}') GROUP BY object
                ) GROUP BY node
            ) TO '{_sql_path(degrees)}' (FORMAT PARQUET)
            """
        )
    con.execute(f"CREATE TABLE deg AS SELECT * FROM read_parquet('{_sql_path(degrees)}')")


def _create_edge_view(con, kg_dir: Path, params: ContextGraphParams) -> None:
    """Expose the traversable edges as `e` (directed) and `und` (both directions)."""
    edges = _sql_path(kg_dir / "edges.parquet")
    nodes = _sql_path(kg_dir / "nodes.parquet")
    preds = _sql_path(kg_dir / "predicates.parquet")
    chem = ", ".join(f"'{c}'" for c in CHEMICAL_CATEGORIES)
    pharm = ", ".join(f"'{s}'" for s in PHARMACOLOGY_SOURCES)

    conditions = [f"e.weight >= {params.min_edge_weight}"]
    if params.exclude_prediction_edges:
        conditions.append("e.knowledge_level <> 'prediction'")
    if params.pharmacology_only:
        conditions.append(
            f"(NOT (a.category IN ({chem}) OR b.category IN ({chem})) OR e.primary_knowledge_source IN ({pharm}))"
        )
    where = " AND ".join(conditions)

    con.execute(
        f"""
        CREATE TABLE e AS
        SELECT e.subject, e.object, p.predicate, e.weight, e.primary_knowledge_source AS source,
               e.knowledge_level
        FROM read_parquet('{edges}') e
        JOIN read_parquet('{preds}') p USING (predicate_code)
        JOIN read_parquet('{nodes}') a ON a.id = e.subject
        JOIN read_parquet('{nodes}') b ON b.id = e.object
        WHERE {where}
        """
    )
    con.execute("CREATE VIEW und AS SELECT subject AS a, object AS b, weight FROM e UNION ALL SELECT object AS a, subject AS b, weight FROM e")


def run_context_graph(
    seeds: list[Seed],
    out_dir: str | Path,
    params: ContextGraphParams | None = None,
    kg_dir: str | Path | None = None,
    find_ids: Iterable[str] | None = None,
) -> ContextGraphResult:
    """Expand the seeds, score the reached nodes, and write the context graph to `out_dir`.

    `find_ids` are KG2c ids whose ranks are reported in the result (evaluation of known entities).
    """
    import duckdb

    params = params or ContextGraphParams()
    kg_dir = Path(kg_dir or settings.kg2c_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    con = duckdb.connect()
    _ensure_degrees(con, kg_dir)
    _create_edge_view(con, kg_dir, params)
    nodes_path = _sql_path(kg_dir / "nodes.parquet")
    con.execute(f"CREATE TABLE nodes AS SELECT id, name, category FROM read_parquet('{nodes_path}')")

    # Seeds: only those present in the filtered graph can be expanded.
    con.execute("CREATE TABLE seeds (seed VARCHAR, seed_weight DOUBLE)")
    con.executemany("INSERT INTO seeds VALUES (?, ?)", [(s.kg2c_id, s.weight) for s in seeds])
    present = {r[0] for r in con.execute("SELECT seed FROM seeds WHERE seed IN (SELECT id FROM nodes)").fetchall()}
    missing = [s.kg2c_id for s in seeds if s.kg2c_id not in present]
    if missing:
        logger.warning("%d seed(s) absent from the filtered KG2c: %s", len(missing), missing[:10])

    # reach(seed, node, hop, path_w): best path weight from each seed to each node, hop by hop.
    con.execute("CREATE TABLE reach (seed VARCHAR, node VARCHAR, hop INTEGER, path_w DOUBLE)")
    con.execute("INSERT INTO reach SELECT seed, seed, 0, seed_weight FROM seeds WHERE seed IN (SELECT id FROM nodes)")
    per_hop = []
    for hop in range(1, params.hops + 1):
        # A frontier node at hop >= 1 is an intermediate node of the paths it extends: its
        # degree penalty is applied here. Seeds (hop 0) are not penalized.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE frontier AS
            SELECT r.seed, r.node,
                   CASE WHEN r.hop = 0 THEN r.path_w ELSE r.path_w / LOG2(2 + d.degree) END AS path_w
            FROM reach r
            JOIN deg d ON d.node = r.node
            WHERE r.hop = {hop - 1} AND (r.hop = 0 OR d.degree <= {params.hub_cap})
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE step AS
            SELECT f.seed, u.b AS node, MAX(f.path_w * u.weight) AS path_w
            FROM frontier f JOIN und u ON u.a = f.node
            GROUP BY f.seed, u.b
            """
        )
        con.execute(
            f"""
            INSERT INTO reach
            SELECT s.seed, s.node, {hop}, s.path_w FROM step s
            WHERE NOT EXISTS (SELECT 1 FROM reach r WHERE r.seed = s.seed AND r.node = s.node)
            """
        )
        n_new = con.execute(f"SELECT COUNT(DISTINCT node) FROM reach WHERE hop = {hop}").fetchone()[0]
        per_hop.append({"hop": hop, "nodes_reached": int(n_new)})
        logger.info("hop %d: %d node(s) reached", hop, n_new)

    # Node scores: coverage-weighted path weight, penalized by the node's own degree.
    con.execute(
        f"""
        CREATE TABLE scored AS
        SELECT r.node, n.name, n.category,
               MIN(r.hop) AS hop,
               COUNT(DISTINCT r.seed) AS coverage,
               SUM(r.path_w) / POWER(LOG2(2 + COALESCE(d.degree, 0)), {params.alpha}) AS score,
               COALESCE(d.degree, 0) AS degree,
               STRING_AGG(DISTINCT r.seed, ';') AS seeds
        FROM reach r
        JOIN nodes n ON n.id = r.node
        LEFT JOIN deg d ON d.node = r.node
        WHERE r.hop > 0 AND r.node NOT IN (SELECT seed FROM seeds)
        GROUP BY r.node, n.name, n.category, d.degree
        """
    )
    con.execute(
        f"""
        CREATE TABLE kept AS
        SELECT * FROM scored ORDER BY score DESC, coverage DESC, hop ASC LIMIT {params.top_k_nodes}
        """
    )
    con.execute("CREATE TABLE kept_ids AS SELECT node AS id FROM kept UNION SELECT seed FROM seeds WHERE seed IN (SELECT id FROM nodes)")

    # Provenance: seed id -> slides and mentions, joined into the node table.
    seed_pages = {s.kg2c_id: s for s in seeds}

    def pages_for(seed_list: str) -> str:
        pages: set[int] = set()
        for sid in seed_list.split(";"):
            pages.update(seed_pages.get(sid, Seed(sid)).pages)
        return ";".join(str(p) for p in sorted(pages))

    # ---- exports ---------------------------------------------------------------------
    node_rows = con.execute(
        """
        SELECT k.node, k.name, k.category, k.hop, k.coverage, k.score, k.degree, k.seeds FROM kept k
        ORDER BY k.score DESC
        """
    ).fetchall()
    with open(out_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "category", "hop", "coverage", "score", "degree", "seeds", "slides", "is_seed"])
        for s in seeds:
            if s.kg2c_id in present:
                w.writerow([s.kg2c_id, s.label, "", 0, 0, "", "", s.kg2c_id, ";".join(map(str, s.pages)), 1])
        for row in node_rows:
            w.writerow([*row[:8], pages_for(row[7]), 0])

    edge_rows = con.execute(
        """
        SELECT e.subject, e.predicate, e.object, e.weight, e.source, e.knowledge_level
        FROM e WHERE e.subject IN (SELECT id FROM kept_ids) AND e.object IN (SELECT id FROM kept_ids)
        """
    ).fetchall()
    with open(out_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "weight", "primary_knowledge_source", "knowledge_level"])
        w.writerows(edge_rows)

    chem = ", ".join(f"'{c}'" for c in CHEMICAL_CATEGORIES)
    compound_rows = con.execute(
        f"""
        SELECT node, name, hop, coverage, score, degree, seeds FROM scored
        WHERE category IN ({chem}) ORDER BY score DESC LIMIT 50
        """
    ).fetchall()
    top_compounds = [
        {"id": r[0], "name": r[1], "hop": r[2], "coverage": r[3], "score": round(r[4], 6), "degree": r[5], "seeds": r[6]}
        for r in compound_rows
    ]
    with open(out_dir / "compounds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "hop", "coverage", "score", "degree", "seeds"])
        w.writeheader()
        w.writerows(top_compounds)

    _write_graphml(out_dir / "context_graph.graphml", node_rows, edge_rows, seeds, present)

    # Ranks of nodes of interest (evaluation hook): overall rank and rank within their category.
    ranks = {}
    if find_ids:
        placeholders = ", ".join("?" for _ in find_ids)
        for row in con.execute(
            f"""
            WITH ranked AS (
                SELECT node, name, category, hop, coverage, score, degree,
                       RANK() OVER (ORDER BY score DESC) AS rank_all,
                       RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rank_in_category,
                       COUNT(*) OVER () AS n_all,
                       COUNT(*) OVER (PARTITION BY category) AS n_in_category
                FROM scored
            )
            SELECT * FROM ranked WHERE node IN ({placeholders})
            """,
            list(find_ids),
        ).fetchall():
            ranks[row[0]] = {
                "name": row[1], "category": row[2], "hop": row[3], "coverage": row[4],
                "score": round(row[5], 6), "degree": row[6],
                "rank_all": f"{row[7]}/{row[9]}", "rank_in_category": f"{row[8]}/{row[10]}",
            }
        for node_id in find_ids:
            ranks.setdefault(node_id, {"name": None, "reached": False})

    result = ContextGraphResult(
        out_dir=str(out_dir),
        n_seeds=len(seeds),
        n_seeds_in_graph=len(present),
        n_nodes=len(node_rows) + len(present),
        n_edges=len(edge_rows),
        per_hop=per_hop,
        top_compounds=top_compounds[:20],
        params=asdict(params),
        seconds=round(time.perf_counter() - t0, 1),
        ranks=ranks,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    con.close()
    logger.info("context graph: %d nodes, %d edges, %.1f s -> %s", result.n_nodes, result.n_edges, result.seconds, out_dir)
    return result


def _write_graphml(path: Path, node_rows, edge_rows, seeds: list[Seed], present: set[str]) -> None:
    """Write the context graph for Cytoscape or Gephi; skipped when python-igraph is missing."""
    try:
        import igraph as ig
    except ImportError:
        logger.warning("python-igraph not installed: GraphML export skipped")
        return
    ids = [s.kg2c_id for s in seeds if s.kg2c_id in present] + [r[0] for r in node_rows]
    index = {node_id: i for i, node_id in enumerate(ids)}
    g = ig.Graph(n=len(ids), directed=True)
    g.vs["id"] = ids
    g.vs["name"] = [s.label for s in seeds if s.kg2c_id in present] + [r[1] for r in node_rows]
    g.vs["category"] = ["seed"] * len(present) + [r[2] for r in node_rows]
    g.vs["hop"] = [0] * len(present) + [int(r[3]) for r in node_rows]
    g.vs["score"] = [0.0] * len(present) + [float(r[5]) for r in node_rows]
    g.vs["is_seed"] = [True] * len(present) + [False] * len(node_rows)
    edges = [(index[s], index[o]) for s, _, o, *_ in edge_rows if s in index and o in index]
    g.add_edges(edges)
    g.es["predicate"] = [p for s, p, o, *_ in edge_rows if s in index and o in index]
    g.es["weight"] = [float(w) for s, p, o, w, *_ in edge_rows if s in index and o in index]
    g.write_graphml(str(path))


# ------------------------------------------------------------------------------------------
# Command line: run on explicit seeds or on a BiomedCAT result JSON
# ------------------------------------------------------------------------------------------

def _seeds_from_result_json(path: Path) -> list[Seed]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds: dict[str, Seed] = {}
    for ent in payload.get("entities", []):
        kg2c_id = ent.get("kg2c_id")
        if not kg2c_id:
            continue
        seed = seeds.setdefault(kg2c_id, Seed(kg2c_id=kg2c_id, label=ent.get("label") or ent.get("text", "")))
        if ent.get("page") is not None and ent["page"] not in seed.pages:
            seed.pages.append(ent["page"])
        if ent.get("text") not in seed.mentions:
            seed.mentions.append(ent.get("text", ""))
    return list(seeds.values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a user-defined context graph around KG2c seeds.")
    parser.add_argument("--seeds", help="Comma-separated KG2c ids (e.g. NCBIGene:100288687,MONDO:0008030).")
    parser.add_argument("--result-json", help="BiomedCAT output JSON; seeds are its entities' kg2c_id values.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--hub-cap", type=int, default=500)
    parser.add_argument("--min-edge-weight", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--pharmacology-only", action="store_true", help="Keep chemical edges from pharmacology sources only.")
    parser.add_argument("--no-predictions", action="store_true", help="Drop edges with knowledge_level = prediction.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Exponent of the target node degree penalty.")
    parser.add_argument("--find", default=None, help="Comma-separated KG2c ids whose ranks are reported.")
    parser.add_argument("--kg-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.result_json:
        seed_list = _seeds_from_result_json(Path(args.result_json))
    elif args.seeds:
        seed_list = [Seed(kg2c_id=s.strip(), label=s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        parser.error("--seeds or --result-json is required")

    res = run_context_graph(
        seed_list,
        args.out,
        ContextGraphParams(
            hops=args.hops,
            hub_cap=args.hub_cap,
            min_edge_weight=args.min_edge_weight,
            top_k_nodes=args.top_k,
            pharmacology_only=args.pharmacology_only,
            exclude_prediction_edges=args.no_predictions,
            alpha=args.alpha,
        ),
        kg_dir=args.kg_dir,
        find_ids=[s.strip() for s in args.find.split(",")] if args.find else None,
    )
    print(json.dumps({k: v for k, v in asdict(res).items() if k != "top_compounds"}, indent=2))
    print("\nTop compounds:")
    for c in res.top_compounds:
        print(f"  {c['score']:.4f}  hop {c['hop']}  cov {c['coverage']}  deg {c['degree']:>5}  {c['name'][:70]}")
