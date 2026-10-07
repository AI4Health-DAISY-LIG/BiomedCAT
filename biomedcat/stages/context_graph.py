"""Stage 4 (context graph): bounded, user-defined neighborhood of RTX-KG2c around the slide entities.

The seeds are the RTX-KG2c ids of the entities linked by the normalization stage. The stage
expands them hop by hop over the filtered KG2c Parquet files (scripts/build_kg2c_parquet.py),
entirely inside DuckDB, so the graph is never fully resident in memory. Every retained node
carries its provenance: which seeds reach it, hence which slides, and at what distance.

Edge weights come from the user preference profile (biomedcat.profiles): predicate weight x
knowledge-source weight x knowledge-level weight. Edges whose predicate is absent from the
profile are not traversed. The profile also defines a class scope (entity branches with a
non-zero priority, biomedcat.weights): a node whose Biolink category is out of scope is
neither expanded nor kept, unless it is a seed. A predicate can also be prefix-restricted to
certain CURIE prefixes on both endpoints (e.g. `subclass_of` kept only between disease/phenotype
ontology terms): an edge whose predicate is restricted and whose subject or object does not
match any listed prefix is not traversed, whatever its weight. Two profiles on the same seeds
therefore give two different context graphs.

None of this is decided when the Parquet files are built (scripts/build_kg2c_parquet.py keeps
every structurally valid edge, for every profile at once): weight, scope and prefix restriction
are all resolved here, at query time, from whichever profile is active for this run.

Directionality is a profile parameter. With mode "on" an edge is followed from subject to
object at its full weight and from object to subject at `inverse_factor` times its weight
(symmetric Biolink predicates such as `interacts with` are never penalised); with mode "off"
the traversal is undirected. Representation edges (gene <-> transcript <-> protein, gene or
protein <-> drug target) carry predicate weight 1 whatever the profile: crossing identities is
not a mechanistic step.

Pruning (this is not a ranking of hypotheses, which is the subject of MEDiQ; it only bounds
the graph so that it can be displayed and shared):

    path_weight(s -> v)  = product of edge weights along the path
                           / product over intermediate nodes u of log2(2 + degree(u))
    score(v)             = sum over seeds s of best path_weight(s -> v)
                           / log2(2 + degree(v)) ** alpha

Intermediate nodes are penalized by their degree so that a path through a generic hub
("cancer", actin, a housekeeping protein) contributes little. Before the per-category cap, a
node scoring below `score_floor_percentile` of its own hop's score distribution is dropped from
contention (`scored` still reports it, for presence checks: only `kept` and the connectivity
refill see the floor). Score scales differently at each hop (an extra edge weight and degree
penalty per hop), so the floor is computed per hop rather than once over the whole reached pool:
a fixed percentile then removes a comparable share of weak evidence at hop 1 and hop 2, instead of
a number tuned to whichever hop's scale dominates. It mainly matters for categories that never
fill their cap, where today every reached node is kept regardless of how little evidence supports
it -- e.g. the Transcript identities pulled in "for free" by the representation closure of a gene
with many annotated isoforms, most reached from a single seed at a near-zero score (checked on the
FSHD case, 10-11 Sept 2026: 50% is comfortably below the ~90% point at which genuinely informative
nodes, such as the DUX4 target-activation mechanism, start being cut). The graph is then capped
per Biolink category, so every layer of the category layout stays populated. Nodes above
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


# Edges that change the *representation* of a target rather than describe a mechanism. RTX-KG2c
# stores one biological target under several identities (gene NCBIGene, transcript, protein
# UniProtKB, drug target CHEMBL.TARGET), and the mechanistic edge often lands on a different
# identity from the one the seeds reach: losmapimod interacts with CHEMBL.TARGET:CHEMBL260
# (p38 alpha, weight 0.4) while the graph holds NCBIGene:1432 (MAPK14). Crossing between those
# identities is not a mechanistic step, so it must not consume the hop budget; the nodes stay
# distinct (this is not identity conflation) and the edge weight still applies.
REPRESENTATION_PREDICATES = (
    "biolink:gene_product_of", "biolink:has_gene_product", "biolink:transcribed_to",
    "biolink:transcribed_from", "biolink:translates_to", "biolink:translation_of",
    "biolink:has_part", "biolink:part_of",
)
# `has_part` / `part_of` are generic (400k edges), so they are hop-neutral only between a gene or
# protein identity and a drug-target identity, which is the case that matters here.
REPRESENTATION_PREFIX_PAIRS = {
    "biolink:has_part": (("NCBIGene:", "UniProtKB:", "ENSEMBL:", "PR:"), ("CHEMBL.TARGET:",)),
    "biolink:part_of": (("CHEMBL.TARGET:",), ("NCBIGene:", "UniProtKB:", "ENSEMBL:", "PR:")),
}
# A change of representation goes from one identifier of a biological object to another identifier
# of the same object. KG2c also carries `gene_product_of` / `has_gene_product` edges whose endpoint
# is a UMLS or NCIT *concept* ("Genes, Regulator", 1,599 edges; "Regulatory Protein"): those are
# classifications, not identities, and letting the hop-neutral closure step through them pulls
# every member of the concept to the seed's hop at full weight (found on the FSHD graphs, 10 Sept
# 2026: obscure genes outranking MAPK14 inside the Gene category). Both endpoints of every
# representation edge must therefore carry one of
# these identifier namespaces. This is a fixed rule of the stage, not a profile parameter.
REPRESENTATION_IDENTITY_PREFIXES = ("NCBIGene:", "ENSEMBL:", "UniProtKB:", "PR:", "CHEMBL.TARGET:", "HGNC:", "RefSeq:", "NCBITranscript:", "MIRBASE:")


@dataclass
class ContextGraphParams:
    profile: str = ""                     # profile JSON path; default: settings.predicate_profile
    hops: int = 2
    representation_hop_neutral: bool = True   # cross gene/protein/target identities without a hop
    max_representation_steps: int = 2         # closure depth of those free steps
    directionality: str = ""              # "" = as the profile says; "on" / "off" to override
    inverse_factor: float | None = None   # override of the profile's reverse-direction factor (mode on)
    hub_cap: int = 500
    min_edge_weight: float = 0.0          # edges below this (after profile weighting) are ignored
    per_category_cap: int = 300           # kept nodes per Biolink category
    score_floor_percentile: float = 0.5   # drop nodes below this percentile of score, per hop, before the cap (0 = off)
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
    weighting: dict = field(default_factory=dict)   # effective directionality, scope, derived-weights file
    score_floor: dict = field(default_factory=dict)   # percentile, per-hop thresholds, nodes dropped before the cap


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


def _representation_clause(pred_col: str = "predicate", subj_col: str = "subject", obj_col: str = "object") -> str:
    """SQL condition selecting the representation edges (see REPRESENTATION_PREDICATES): both
    endpoints in an identifier namespace, plus the prefix pairs of has_part / part_of."""
    clauses = []
    ident = " OR ".join(f"{{c}} LIKE '{p}%'" for p in REPRESENTATION_IDENTITY_PREFIXES)
    for pred in REPRESENTATION_PREDICATES:
        pair = REPRESENTATION_PREFIX_PAIRS.get(pred)
        if pair is None:
            clauses.append(f"({pred_col} = '{pred}' AND ({ident.format(c=subj_col)}) AND ({ident.format(c=obj_col)}))")
        else:
            left = " OR ".join(f"{subj_col} LIKE '{p}%'" for p in pair[0])
            right = " OR ".join(f"{obj_col} LIKE '{p}%'" for p in pair[1])
            clauses.append(f"({pred_col} = '{pred}' AND ({left}) AND ({right}))")
    return " OR ".join(clauses)


def _effective_directionality(profile: Profile, params: ContextGraphParams) -> tuple[str, float]:
    """(mode, reverse factor) after the command-line overrides; factor 1.0 when undirected."""
    mode = params.directionality or ("on" if profile.directed else "off")
    if mode == "off":
        return "off", 1.0
    factor = params.inverse_factor if params.inverse_factor is not None else profile.inverse_factor
    return "on", max(0.0, min(1.0, float(factor)))


def _prefix_restriction_clause(profile: Profile, pred_col: str = "predicate", subj_col: str = "subject",
                               obj_col: str = "object") -> str | None:
    """SQL condition enforcing Profile.endpoints_allowed() for every prefix-restricted predicate.

    A prefix-restricted predicate (e.g. subclass_of limited to MONDO:/HP:) is kept only when both
    endpoints start with one of the profile's listed prefixes for that predicate; every other
    predicate is unaffected. This is a profile choice, applied here at query time, never at the
    Parquet build (which knows nothing about profiles). None when the profile restricts nothing
    (the common case), so the caller adds no condition at all.
    """
    restrictions = {p: prefixes for p, prefixes in profile.prefix_restricted_predicates.items() if prefixes}
    if not restrictions:
        return None
    clauses = []
    for pred, prefixes in restrictions.items():
        pred_lit = pred.replace("'", "''")

        def prefix_or(col: str, prefixes: list[str] = prefixes) -> str:
            return " OR ".join(f"{col} LIKE '{p.replace(chr(39), chr(39) * 2)}%'" for p in prefixes)

        clauses.append(f"({pred_col} != '{pred_lit}' OR (({prefix_or(subj_col)}) AND ({prefix_or(obj_col)})))")
    return " AND ".join(clauses)


def _create_edge_table(con, kg_dir: Path, profile: Profile, params: ContextGraphParams) -> None:
    """Expose the traversable edges as `e` (directed) and `und` (both directions), profile-weighted.

    Requires the `nodes` and `seeds` tables. Inside `e`, `w` is the predicate weight and `weight`
    the full edge weight. Representation edges carry predicate weight 1 whatever the profile.
    An edge whose endpoint category is out of the profile's class scope is dropped, unless that
    endpoint is a seed (the user's own entities are always expandable). An edge whose predicate is
    prefix-restricted and whose endpoints do not match is dropped unconditionally.
    """
    edges = _sql_path(kg_dir / "edges.parquet")
    preds = _sql_path(kg_dir / "predicates.parquet")
    register_in_duckdb(con, profile)
    repr_clause = _representation_clause()
    repr_clause_raw = _representation_clause("p.predicate", "e.subject", "e.object")

    conditions = ["w > 0", f"weight >= {params.min_edge_weight}"]
    if params.exclude_prediction_edges:
        conditions.append("knowledge_level <> 'prediction'")
    prefix_clause = _prefix_restriction_clause(profile)
    if prefix_clause:
        conditions.append(prefix_clause)
    scope_join, scope_where = "", ""
    if profile.category_weights:
        scope_join = """
            LEFT JOIN nodes ns ON ns.id = e.subject
            LEFT JOIN prof_cat cs ON cs.category = ns.category
            LEFT JOIN nodes no ON no.id = e.object
            LEFT JOIN prof_cat co ON co.category = no.category"""
        conditions.append("(subject_in_scope OR subject IN (SELECT seed FROM seeds))")
        conditions.append("(object_in_scope OR object IN (SELECT seed FROM seeds))")
        scope_where = ", COALESCE(cs.w, 1.0) > 0 AS subject_in_scope, COALESCE(co.w, 1.0) > 0 AS object_in_scope"
    else:
        scope_where = ", TRUE AS subject_in_scope, TRUE AS object_in_scope"
    where = " AND ".join(conditions)

    con.execute(
        f"""
        CREATE TABLE e AS
        SELECT subject, object, predicate, weight, source, knowledge_level, w FROM (
            SELECT e.subject, e.object, p.predicate,
                   CASE WHEN {repr_clause_raw} THEN 1.0 ELSE COALESCE(pp.w, 0.0) END AS w,
                   CASE WHEN {repr_clause_raw} THEN 1.0 ELSE COALESCE(pp.w, 0.0) END
                       * COALESCE(ps.w, 1.0) * COALESCE(pk.w, 1.0) AS weight,
                   e.primary_knowledge_source AS source, e.knowledge_level{scope_where}
            FROM read_parquet('{edges}') e
            JOIN read_parquet('{preds}') p USING (predicate_code)
            LEFT JOIN prof_pred pp ON pp.predicate = p.predicate
            LEFT JOIN prof_src ps ON ps.source = e.primary_knowledge_source
            LEFT JOIN prof_kl pk ON pk.level = e.knowledge_level{scope_join}
        ) WHERE {where}
        """
    )

    # Traversal view: forward edges at full weight; reverse edges at the profile's reverse
    # factor when the traversal is directed (symmetric predicates excepted), at full weight
    # when it is undirected. A zero factor removes the reverse direction altogether.
    mode, factor = _effective_directionality(profile, params)
    reverse_w = "weight" if mode == "off" else f"weight * CASE WHEN predicate IN (SELECT predicate FROM prof_sym) THEN 1.0 ELSE {factor} END"
    con.execute(
        f"""
        CREATE VIEW und AS
        SELECT subject AS a, object AS b, weight FROM e
        UNION ALL
        SELECT a, b, weight FROM (SELECT object AS a, subject AS b, {reverse_w} AS weight FROM e) WHERE weight > 0
        """
    )

    # Representation edges, in both directions and without penalty, for the hop-neutral closure.
    con.execute(f"""
        CREATE VIEW und_repr AS
        SELECT a, b, weight FROM (
            SELECT subject AS a, object AS b, weight FROM e WHERE {repr_clause}
            UNION ALL
            SELECT object AS a, subject AS b, weight FROM e WHERE {repr_clause}
        )""")


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
    con.execute(f"CREATE TABLE nodes AS SELECT id, name, category FROM read_parquet('{_sql_path(kg_dir / 'nodes.parquet')}')")

    # Seeds: only those present in the filtered graph can be expanded.
    con.execute("CREATE TABLE seeds (seed VARCHAR, seed_weight DOUBLE)")
    con.executemany("INSERT INTO seeds VALUES (?, ?)", [(s.kg2c_id, s.weight) for s in seeds])
    _create_edge_table(con, kg_dir, profile, params)
    mode, factor = _effective_directionality(profile, params)
    n_edges_traversable = con.execute("SELECT COUNT(*) FROM e").fetchone()[0]
    logger.info("profile %s: %d traversable edge(s), directionality %s (reverse factor %.2f), %d categories out of scope",
                profile.name, n_edges_traversable, mode, factor, sum(1 for w in profile.category_weights.values() if w <= 0))
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
        n_free = 0
        if params.representation_hop_neutral:
            n_free = _representation_closure(con, hop, params)
        n_new = con.execute(f"SELECT COUNT(DISTINCT node) FROM reach WHERE hop = {hop}").fetchone()[0]
        per_hop.append({"hop": hop, "nodes_reached": int(n_new), "via_representation_edges": int(n_free)})
        logger.info("hop %d: %d node(s) reached (%d through hop-neutral representation edges)", hop, n_new, n_free)

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

    # Retention floor: per hop (score scales differently at each hop), the nodes below this
    # percentile of their own hop's score never compete for a category slot. `scored` is
    # untouched -- find_ids presence checks still see a floored-out node as reached, just not kept.
    pct = max(0.0, min(1.0, params.score_floor_percentile))
    con.execute(f"CREATE TABLE score_floor AS SELECT hop, QUANTILE_CONT(score, {pct}) AS thr FROM scored GROUP BY hop")
    floor_thresholds = {str(h): float(t) for h, t in con.execute("SELECT hop, thr FROM score_floor ORDER BY hop").fetchall()}
    n_below_floor = con.execute("SELECT COUNT(*) FROM scored s JOIN score_floor f ON f.hop = s.hop WHERE s.score < f.thr").fetchone()[0]
    logger.info("score floor at the %.0f%% percentile per hop: %s, %d/%d reached node(s) below it",
                pct * 100, floor_thresholds, n_below_floor, n_reached)

    con.execute(
        f"""
        CREATE TABLE kept AS
        SELECT * EXCLUDE (rk, thr) FROM (
            SELECT s.*, f.thr,
                   ROW_NUMBER() OVER (PARTITION BY s.category ORDER BY s.score DESC, s.coverage DESC, s.hop ASC) AS rk
            FROM scored s JOIN score_floor f ON f.hop = s.hop
            WHERE s.score >= f.thr
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
        weighting={"directionality": mode, "inverse_factor": factor, "n_traversable_edges": int(n_edges_traversable),
                   "n_predicates": len(profile.predicates), "n_categories_out_of_scope": sum(1 for w in profile.category_weights.values() if w <= 0),
                   "derived_weights": profile.derived_path},
        score_floor={"percentile": pct, "thresholds": floor_thresholds, "nodes_dropped": int(n_below_floor)},
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    logger.info("context graph [%s]: %d nodes, %d edges, diameter %s, %d component(s), %.1f s -> %s",
                profile.name, result.n_nodes, result.n_edges, result.diameter, result.n_components, result.seconds, out_dir)
    return result


def _representation_closure(con, hop: int, params: ContextGraphParams) -> int:
    """Extend the nodes just reached along representation edges, without incrementing the hop.

    A gene, its protein and its drug-target identity are the same biological object under three
    identifiers; stepping between them is a change of representation, not a mechanistic step, so it
    keeps the current hop. The closure is bounded by `max_representation_steps` and never lowers a
    node's hop: a node already reached at an earlier hop keeps it.
    """
    added = 0
    for _ in range(max(0, params.max_representation_steps)):
        n = con.execute(
            f"""
            INSERT INTO reach
            SELECT s.seed, s.node, {hop}, s.path_w FROM (
                SELECT r.seed, u.b AS node, MAX(r.path_w * u.weight) AS path_w
                FROM reach r JOIN und_repr u ON u.a = r.node
                WHERE r.hop = {hop}
                GROUP BY r.seed, u.b
            ) s
            WHERE NOT EXISTS (SELECT 1 FROM reach r2 WHERE r2.seed = s.seed AND r2.node = s.node)
            """
        ).fetchone()
        n = int(n[0]) if n else 0
        added += n
        if n == 0:
            break
    return added


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
        # Refill each category up to the cap with the next best candidates never excluded,
        # restricted to nodes adjacent to the connected set: a refilled node is then connected
        # by construction, so the next pass removes nothing and the loop converges. Candidates
        # below the score floor of their own hop (see run_context_graph) are not refilled either.
        con.execute(
            f"""
            INSERT INTO kept
            SELECT * EXCLUDE (rk, have, thr) FROM (
                SELECT s.*, f.thr, k.have,
                       ROW_NUMBER() OVER (PARTITION BY s.category ORDER BY s.score DESC, s.coverage DESC, s.hop ASC) AS rk
                FROM scored s
                JOIN score_floor f ON f.hop = s.hop
                LEFT JOIN (SELECT category, COUNT(*) AS have FROM kept GROUP BY category) k USING (category)
                WHERE s.score >= f.thr
                  AND s.node NOT IN (SELECT node FROM kept) AND s.node NOT IN (SELECT node FROM excluded)
                  AND s.node IN (SELECT DISTINCT u.b FROM und u JOIN conn c ON u.a = c.node)
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
    parser.add_argument("--score-floor-percentile", type=float, default=0.5,
                        help="Drop nodes below this percentile of score, computed per hop, before the cap (0 = off).")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-predictions", action="store_true", help="Drop edges with knowledge_level = prediction.")
    parser.add_argument("--max-connectivity-passes", type=int, default=3)
    parser.add_argument("--no-representation-hop-neutral", action="store_true",
                        help="Make gene/protein/drug-target representation edges consume a hop like any other edge.")
    parser.add_argument("--directionality", choices=["on", "off"], default="",
                        help="Override the profile: on = directed with a penalty on the reverse direction, off = undirected.")
    parser.add_argument("--inverse-factor", type=float, default=None,
                        help="Override the profile's reverse-direction factor (0 = never go against an edge, 1 = no penalty).")
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
            score_floor_percentile=args.score_floor_percentile,
            alpha=args.alpha,
            exclude_prediction_edges=args.no_predictions,
            max_connectivity_passes=args.max_connectivity_passes,
            representation_hop_neutral=not args.no_representation_hop_neutral,
            directionality=args.directionality,
            inverse_factor=args.inverse_factor,
        ),
        kg_dir=args.kg_dir,
        find_ids=[s.strip() for s in args.find.split(",")] if args.find else None,
    )
    print(json.dumps(asdict(res), indent=2))
