"""Stage 4 (context graph): bounded, user-defined neighborhood of RTX-KG2c around the slide entities.

The seeds are the RTX-KG2c ids of the entities linked by the normalization stage. The stage
expands them hop by hop over the filtered KG2c Parquet files (scripts/build_kg2c_parquet.py),
entirely inside DuckDB, so the graph is never fully resident in memory. Every retained node
carries its provenance: which seeds reach it, hence which slides, and at what distance.

Edge weights come from the user preference profile (biomedcat.profiles): predicate weight x
knowledge-source weight x knowledge-level weight. Edges whose predicate is absent from the
profile are not traversed. Two profiles on the same seeds therefore give two different
context graphs.

Pruning (this is not a ranking of hypotheses, which is the subject of MEDiQ; it only bounds
the graph so that it can be displayed and shared):

    path_weight(s -> v)  = product of edge weights along the path
                           / product over intermediate nodes u of log2(2 + degree(u))
    score(v)             = sum over seeds s of best path_weight(s -> v)
                           / log2(2 + degree(v)) ** alpha

Intermediate nodes are penalized by their degree so that a path through a generic hub
("cancer", actin, a housekeeping protein) contributes little. The graph is then capped per
Biolink category, so every layer of the category layout stays populated. Nodes above
`hub_cap` are reached but never expanded.

Reported metrics are presence metrics: size, category composition, vocabulary expansion
(nodes per seed), diameter and number of connected components of the kept graph, number of
seed pairs connected inside it, and the position of nodes of interest when asked.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from biomedcat.config import settings
from biomedcat.profiles import Profile, load_profile, register_in_duckdb

logger = logging.getLogger(__name__)


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
    profile: str = ""                     # profile JSON path; default: settings.predicate_profile
    hops: int = 2
    hub_cap: int = 500
    min_edge_weight: float = 0.0          # edges below this (after profile weighting) are ignored
    per_category_cap: int = 300           # kept nodes per Biolink category
    alpha: float = 0.5                    # exponent of the target node's own degree penalty
    exclude_prediction_edges: bool = False
    max_connectivity_passes: int = 3      # prune islands + refill caps, at most this many times


@dataclass
class ContextGraphResult:
    out_dir: str
    profile: str
    n_seeds: int
    n_seeds_in_graph: int
    n_nodes: int
    n_edges: int
    n_reached: int
    per_hop: list[dict]
    categories: dict
    vocabulary_expansion: float           # kept nodes / seeds present
    diameter: int | None
    n_components: int
    seed_pairs_connected: str             # "connected/total"
    connectivity: dict                    # passes, converged, nodes_removed
    params: dict
    seconds: float
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
    edges = _sql_path(kg_dir / "edges.parquet")
    if not degrees.is_file():
        logger.info("computing node degrees into %s", degrees)
        con.execute(
            f"""
            COPY (
                SELECT node, SUM(n)::BIGINT AS degree FROM (
                    SELECT subject AS node, COUNT(*) AS n FROM read_parquet('{edges}') GROUP BY subject
                    UNION ALL
                    SELECT object AS node, COUNT(*) AS n FROM read_parquet('{edges}') GROUP BY object
                ) GROUP BY node
            ) TO '{_sql_path(degrees)}' (FORMAT PARQUET)
            """
        )
    con.execute(f"CREATE TABLE deg AS SELECT * FROM read_parquet('{_sql_path(degrees)}')")


def _create_edge_table(con, kg_dir: Path, profile: Profile, params: ContextGraphParams) -> None:
    """Expose the traversable edges as `e` (directed) and `und` (both directions), profile-weighted."""
    edges = _sql_path(kg_dir / "edges.parquet")
    preds = _sql_path(kg_dir / "predicates.parquet")
    register_in_duckdb(con, profile)

    # Inside the outer WHERE, `w` is the predicate weight and `weight` the full edge weight.
    conditions = ["w > 0", f"weight >= {params.min_edge_weight}"]
    if params.exclude_prediction_edges:
        conditions.append("knowledge_level <> 'prediction'")
    where = " AND ".join(conditions)

    con.execute(
        f"""
        CREATE TABLE e AS
        SELECT * FROM (
            SELECT e.subject, e.object, p.predicate,
                   pp.w * COALESCE(ps.w, 1.0) * COALESCE(pk.w, 1.0) AS weight,
                   e.primary_knowledge_source AS source, e.knowledge_level,
                   pp.w
            FROM read_parquet('{edges}') e
            JOIN read_parquet('{preds}') p USING (predicate_code)
            LEFT JOIN prof_pred pp ON pp.predicate = p.predicate
            LEFT JOIN prof_src ps ON ps.source = e.primary_knowledge_source
            LEFT JOIN prof_kl pk ON pk.level = e.knowledge_level
        ) WHERE {where}
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
    """Expand the seeds, prune, write the context graph to `out_dir`, return presence metrics.

    `find_ids` are KG2c ids whose position is reported in the result (presence of known entities).
    """
    import duckdb

    params = params or ContextGraphParams()
    if not seeds:
        raise ValueError("No seeds: the normalization stage linked no entity to RTX-KG2c, nothing to expand.")
    profile = load_profile(params.profile or settings.predicate_profile)
    kg_dir = Path(kg_dir or settings.kg2c_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    con = duckdb.connect()
    _ensure_degrees(con, kg_dir)
    _create_edge_table(con, kg_dir, profile, params)
    con.execute(f"CREATE TABLE nodes AS SELECT id, name, category FROM read_parquet('{_sql_path(kg_dir / 'nodes.parquet')}')")

    # Seeds: only those present in the filtered graph can be expanded.
    con.execute("CREATE TABLE seeds (seed VARCHAR, seed_weight DOUBLE)")
    con.executemany("INSERT INTO seeds VALUES (?, ?)", [(s.kg2c_id, s.weight) for s in seeds])
    present = [s.kg2c_id for s in seeds if con.execute("SELECT COUNT(*) FROM nodes WHERE id = ?", [s.kg2c_id]).fetchone()[0]]
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
            FROM reach r JOIN deg d ON d.node = r.node
            WHERE r.hop = {hop - 1} AND (r.hop = 0 OR d.degree <= {params.hub_cap})
            """
        )
        con.execute(
            """
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

    # Pruning score, then a cap per Biolink category.
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
    n_reached = con.execute("SELECT COUNT(*) FROM scored").fetchone()[0]
    con.execute(
        f"""
        CREATE TABLE kept AS
        SELECT * EXCLUDE (rk) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC, coverage DESC, hop ASC) AS rk
            FROM scored
        ) WHERE rk <= {params.per_category_cap}
        """
    )
    connectivity = _connectivity_passes(con, params)

    # ---- exports ---------------------------------------------------------------------
    seed_by_id = {s.kg2c_id: s for s in seeds}

    def pages_for(seed_list: str) -> str:
        pages: set[int] = set()
        for sid in seed_list.split(";"):
            pages.update(seed_by_id[sid].pages if sid in seed_by_id else [])
        return ";".join(str(p) for p in sorted(pages))

    node_rows = con.execute(
        "SELECT node, name, category, hop, coverage, score, degree, seeds FROM kept ORDER BY category, score DESC"
    ).fetchall()
    with open(out_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "category", "hop", "coverage", "shared", "score", "degree", "seeds", "slides", "is_seed"])
        for sid in present:
            s = seed_by_id[sid]
            w.writerow([sid, s.label, "seed", 0, 0, "", "", "", sid, ";".join(map(str, s.pages)), 1])
        for row in node_rows:
            shared = "shared" if row[4] > 1 else "specific"
            w.writerow([row[0], row[1], row[2], row[3], row[4], shared, round(row[5], 6), row[6], row[7], pages_for(row[7]), 0])

    edge_rows = con.execute(
        """
        SELECT e.subject, e.predicate, e.object, e.weight, e.source, e.knowledge_level
        FROM e WHERE e.subject IN (SELECT id FROM kept_ids) AND e.object IN (SELECT id FROM kept_ids)
        """
    ).fetchall()
    with open(out_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "weight", "primary_knowledge_source", "knowledge_level"])
        for r in edge_rows:
            w.writerow([r[0], r[1], r[2], round(r[3], 6), r[4], r[5]])

    categories = {
        cat: int(n) for cat, n in con.execute("SELECT category, COUNT(*) FROM kept GROUP BY category ORDER BY 2 DESC").fetchall()
    }

    # Position of nodes of interest (presence check), overall and within their category.
    ranks: dict = {}
    if find_ids:
        find_ids = list(find_ids)
        placeholders = ", ".join("?" for _ in find_ids)
        for row in con.execute(
            f"""
            WITH ranked AS (
                SELECT node, name, category, hop, coverage, degree,
                       RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rank_in_category,
                       COUNT(*) OVER (PARTITION BY category) AS n_in_category,
                       node IN (SELECT id FROM kept_ids) AS kept
                FROM scored
            )
            SELECT * FROM ranked WHERE node IN ({placeholders})
            """,
            find_ids,
        ).fetchall():
            ranks[row[0]] = {
                "name": row[1], "category": row[2], "hop": row[3], "coverage": row[4], "degree": row[5],
                "rank_in_category": f"{row[6]}/{row[7]}", "kept": bool(row[8]), "reached": True,
            }
        for node_id in find_ids:
            ranks.setdefault(node_id, {"reached": False, "kept": False})
    con.close()

    graph_metrics = _graph_metrics(out_dir / "context_graph.graphml", node_rows, edge_rows, seeds, present)

    result = ContextGraphResult(
        out_dir=str(out_dir),
        profile=profile.name,
        n_seeds=len(seeds),
        n_seeds_in_graph=len(present),
        n_nodes=len(node_rows) + len(present),
        n_edges=len(edge_rows),
        n_reached=int(n_reached),
        per_hop=per_hop,
        categories=categories,
        vocabulary_expansion=round((len(node_rows) + len(present)) / max(len(present), 1), 2),
        diameter=graph_metrics["diameter"],
        n_components=graph_metrics["n_components"],
        seed_pairs_connected=graph_metrics["seed_pairs_connected"],
        connectivity=connectivity,
        params=asdict(params),
        seconds=round(time.perf_counter() - t0, 1),
        ranks=ranks,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    logger.info("context graph [%s]: %d nodes, %d edges, diameter %s, %d component(s), %.1f s -> %s",
                profile.name, result.n_nodes, result.n_edges, result.diameter, result.n_components, result.seconds, out_dir)
    return result


def _connectivity_passes(con, params: ContextGraphParams) -> dict:
    """Drop kept nodes that no longer connect to a seed, refill the category caps, repeat.

    The per-category cap can cut the intermediate node through which another kept node was
    reached, leaving islands. Each pass computes the nodes reachable from the seeds through
    kept edges only (a breadth-first search inside DuckDB), removes the others, and refills
    each category with the next best candidates. Refilled nodes may themselves be unconnected,
    hence the loop, bounded by `max_connectivity_passes`; when the bound is hit the result is
    flagged so the user knows the graph may still contain islands.
    """
    con.execute("CREATE OR REPLACE TABLE excluded (node VARCHAR)")
    passes = 0
    removed_total = 0
    converged = False
    while passes < params.max_connectivity_passes:
        passes += 1
        con.execute("CREATE OR REPLACE TABLE kept_ids AS SELECT node AS id FROM kept UNION SELECT seed AS id FROM seeds WHERE seed IN (SELECT id FROM nodes)")
        # BFS from the seeds over kept edges only.
        con.execute("CREATE OR REPLACE TABLE conn AS SELECT seed AS node FROM seeds WHERE seed IN (SELECT id FROM kept_ids)")
        while True:
            n_before = con.execute("SELECT COUNT(*) FROM conn").fetchone()[0]
            con.execute(
                """
                INSERT INTO conn
                SELECT DISTINCT u.b FROM und u
                JOIN conn c ON u.a = c.node
                WHERE u.b IN (SELECT id FROM kept_ids) AND u.b NOT IN (SELECT node FROM conn)
                """
            )
            if con.execute("SELECT COUNT(*) FROM conn").fetchone()[0] == n_before:
                break
        removed = con.execute("SELECT COUNT(*) FROM kept WHERE node NOT IN (SELECT node FROM conn)").fetchone()[0]
        if removed == 0:
            converged = True
            break
        removed_total += removed
        con.execute("INSERT INTO excluded SELECT node FROM kept WHERE node NOT IN (SELECT node FROM conn)")
        con.execute("DELETE FROM kept WHERE node NOT IN (SELECT node FROM conn)")
        # Refill each category up to the cap with the next best candidates never excluded.
        con.execute(
            f"""
            INSERT INTO kept
            SELECT * EXCLUDE (rk, have) FROM (
                SELECT s.*, k.have,
                       ROW_NUMBER() OVER (PARTITION BY s.category ORDER BY s.score DESC, s.coverage DESC, s.hop ASC) AS rk
                FROM scored s
                LEFT JOIN (SELECT category, COUNT(*) AS have FROM kept GROUP BY category) k USING (category)
                WHERE s.node NOT IN (SELECT node FROM kept) AND s.node NOT IN (SELECT node FROM excluded)
            ) WHERE rk <= {params.per_category_cap} - COALESCE(have, 0)
            """
        )
    con.execute("CREATE OR REPLACE TABLE kept_ids AS SELECT node AS id FROM kept UNION SELECT seed AS id FROM seeds WHERE seed IN (SELECT id FROM nodes)")
    if not converged:
        logger.warning("connectivity pruning stopped after %d pass(es) without converging: the graph may contain islands", passes)
    return {"passes": passes, "converged": converged, "nodes_removed": int(removed_total)}


def _graph_metrics(graphml_path: Path, node_rows, edge_rows, seeds: list[Seed], present: list[str]) -> dict:
    """Build the kept graph in igraph, write GraphML for Cytoscape/Gephi, and compute structure metrics."""
    empty = {"diameter": None, "n_components": 0, "seed_pairs_connected": "0/0"}
    try:
        import igraph as ig
    except ImportError:
        logger.warning("python-igraph not installed: GraphML export and graph metrics skipped")
        return empty

    seed_by_id = {s.kg2c_id: s for s in seeds}
    ids = list(present) + [r[0] for r in node_rows]
    index = {node_id: i for i, node_id in enumerate(ids)}
    g = ig.Graph(n=len(ids), directed=True)
    g.vs["id"] = ids
    g.vs["name"] = [seed_by_id[s].label for s in present] + [r[1] for r in node_rows]
    g.vs["category"] = ["seed"] * len(present) + [r[2] for r in node_rows]
    g.vs["hop"] = [0] * len(present) + [int(r[3]) for r in node_rows]
    g.vs["coverage"] = [0] * len(present) + [int(r[4]) for r in node_rows]
    g.vs["shared"] = ["seed"] * len(present) + [("shared" if r[4] > 1 else "specific") for r in node_rows]
    g.vs["is_seed"] = [True] * len(present) + [False] * len(node_rows)
    g.vs["slides"] = [";".join(map(str, seed_by_id[s].pages)) for s in present] + [""] * len(node_rows)
    kept_edges = [(s, p, o, w) for s, p, o, w, *_ in edge_rows if s in index and o in index]
    g.add_edges([(index[s], index[o]) for s, _, o, _ in kept_edges])
    g.es["predicate"] = [p for _, p, _, _ in kept_edges]
    g.es["weight"] = [float(w) for _, _, _, w in kept_edges]
    g.write_graphml(str(graphml_path))

    if g.vcount() == 0:
        return empty
    und = g.as_undirected(mode="collapse")
    components = und.connected_components()
    membership = components.membership
    seed_idx = [index[s] for s in present]
    total_pairs = len(seed_idx) * (len(seed_idx) - 1) // 2
    connected_pairs = sum(
        1 for i in range(len(seed_idx)) for j in range(i + 1, len(seed_idx)) if membership[seed_idx[i]] == membership[seed_idx[j]]
    )
    giant = und.induced_subgraph(max(components, key=len))
    return {
        "diameter": int(giant.diameter(directed=False)),
        "n_components": len(components),
        "seed_pairs_connected": f"{connected_pairs}/{total_pairs}",
    }


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
    parser.add_argument("--profile", default="", help="Profile JSON (default: settings.predicate_profile).")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--hub-cap", type=int, default=500)
    parser.add_argument("--min-edge-weight", type=float, default=0.0)
    parser.add_argument("--per-category-cap", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-predictions", action="store_true", help="Drop edges with knowledge_level = prediction.")
    parser.add_argument("--max-connectivity-passes", type=int, default=3)
    parser.add_argument("--find", default=None, help="Comma-separated KG2c ids whose presence is reported.")
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
            profile=args.profile,
            hops=args.hops,
            hub_cap=args.hub_cap,
            min_edge_weight=args.min_edge_weight,
            per_category_cap=args.per_category_cap,
            alpha=args.alpha,
            exclude_prediction_edges=args.no_predictions,
            max_connectivity_passes=args.max_connectivity_passes,
        ),
        kg_dir=args.kg_dir,
        find_ids=[s.strip() for s in args.find.split(",")] if args.find else None,
    )
    print(json.dumps(asdict(res), indent=2))
