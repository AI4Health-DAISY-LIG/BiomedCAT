"""Console tests: job store and cost model, graph export and filtering, enrichment statistics,
and the HTTP API with the worker disabled. No model, no network, no KG2c needed."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pytest

os.environ["BIOMEDCAT_NO_WORKER"] = "1"

from biomedcat.webapp import jobs as jobs_mod
from biomedcat.webapp import graphs as graphs_mod
from biomedcat.webapp import enrichment


# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return jobs_mod.JobStore(root=tmp_path / "jobs", dataset_root=tmp_path / "dataset")


@pytest.fixture
def graph_dir(tmp_path):
    d = tmp_path / "graph"
    d.mkdir()
    with open(d / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "category", "hop", "coverage", "shared", "score", "degree", "seeds", "slides", "is_seed"])
        w.writerow(["NCBIGene:1", "DUX4", "seed", 0, 0, "", "", "", "NCBIGene:1", "1;2", 1])
        w.writerow(["CHEBI:2", "losmapimod", "biolink:SmallMolecule", 1, 1, "specific", 0.9, 12, "NCBIGene:1", "1;2", 0])
        w.writerow(["MONDO:3", "FSHD", "biolink:Disease", 1, 1, "specific", 0.5, 300, "NCBIGene:1", "1;2", 0])
        w.writerow(["GO:4", "myogenesis", "biolink:BiologicalProcess", 2, 1, "specific", 0.1, 40, "NCBIGene:1", "1;2", 0])
    with open(d / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "weight", "primary_knowledge_source", "knowledge_level"])
        w.writerow(["CHEBI:2", "biolink:affects", "NCBIGene:1", 0.8, "infores:drugbank", "knowledge_assertion"])
        w.writerow(["NCBIGene:1", "biolink:gene_associated_with_condition", "MONDO:3", 0.6, "infores:diseases", "prediction"])
        w.writerow(["NCBIGene:1", "biolink:participates_in", "GO:4", 0.3, "infores:go", "knowledge_assertion"])
        w.writerow(["NCBIGene:1", "biolink:related_to", "MISSING:9", 0.3, "infores:x", "knowledge_assertion"])
    (d / "summary.json").write_text(json.dumps({"profile": "biochemical actions", "n_seeds_in_graph": 1, "diameter": 2, "n_components": 1,
                                                "seed_pairs_connected": "0/0", "vocabulary_expansion": 4.0, "params": {"hops": 2}}), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------------
# jobs
# ------------------------------------------------------------------------------------------

def test_estimate_grows_with_pages_and_profiles():
    one = jobs_mod.estimate_seconds(1, 1, 600)
    assert jobs_mod.estimate_seconds(4, 1, 600) > one
    assert jobs_mod.estimate_seconds(1, 3, 600) > one


def test_store_create_queue_order_and_persistence(store):
    profiles = [p["id"] for p in jobs_mod.list_profiles()]
    assert profiles, "profiles are required for the console"
    a = store.create("first", [{"file": "x.pdf", "stem": "x", "pages": 2}], profiles[:1])
    b = store.create("second", [{"file": "y.png", "stem": "y", "pages": 1}], profiles[:1])
    assert a.status == "queued" and a.estimate_s > 0
    assert store.next_queued().id == a.id
    assert (store.job_dir(a.id) / "profile_merged.json").is_file()
    store.update(a.id, status="done", elapsed_s=1500.0)
    assert store.next_queued().id == b.id
    reloaded = store.get(a.id)
    assert reloaded.status == "done" and reloaded.elapsed_s == 1500.0
    assert [j.id for j in store.list()] == [b.id, a.id]
    store.delete(b.id)
    assert store.next_queued() is None


def test_record_timing_feeds_the_cost_model(store):
    profiles = [p["id"] for p in jobs_mod.list_profiles()]
    job = store.create("j", [{"file": "x.pdf", "stem": "x", "pages": 2}], profiles[:1])
    job.elapsed_s = 2 * 400 + jobs_mod.SEC_STARTUP + 50
    job.progress["history"] = [{"stage": "graph:p", "seconds": 50}]
    store.record_timing(job)
    assert abs(store.sec_per_page() - 400) < 1


def test_merge_profiles_unions_scope(tmp_path):
    from biomedcat.profiles import load_profile

    profiles = [p["id"] for p in jobs_mod.list_profiles()]
    out = jobs_mod.merge_profiles(profiles, tmp_path / "m.json")
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["members"] == profiles
    first = load_profile(jobs_mod.PROFILES_DIR / f"{profiles[0]}.json")
    assert set(merged["predicates"]) >= set(first.predicates) and set(merged["entity_scope"]) >= set(first.entity_scope)


def test_listed_profiles_carry_branch_weights_and_directionality():
    listed = {p["id"]: p for p in jobs_mod.list_profiles()}
    assert "biochemical_actions" in listed and "uniform" in listed
    bio = listed["biochemical_actions"]
    assert bio["branch_weights"].get("gene") == 1.0 and bio["directionality"]["mode"] == "on" and bio["n_predicates"] > 0
    assert listed["uniform"]["directionality"]["mode"] == "off"


def test_entity_branches_for_the_builder():
    data = jobs_mod.entity_branches()
    names = {b["name"] for b in data["branches"]}
    assert {"gene", "chemical entity", "disease or phenotypic feature", "organismal entity"} <= names
    assert "biological entity" not in names and "named thing" not in names
    gene = next(b for b in data["branches"] if b["name"] == "gene")
    assert gene["group"] == "biological entity" and gene["n_nodes_kg2c"] > 0 and data["levels"] == [0, 0.5, 1]


def test_store_create_with_custom_profile(store):
    from biomedcat.profiles import load_profile

    spec = {"name": "My targets", "branch_weights": {"gene": 1, "chemical entity": 0.5, "not a branch": 1, "exon": 0},
            "reading_focus": "gene:  regulated genes\n" + "x" * 500,
            "directionality": {"mode": "on", "inverse_factor": 0.2}}
    job = store.create("j", [{"file": "x.pdf", "stem": "x", "pages": 1}], ["biochemical_actions"], custom_profiles=[spec])
    assert job.profiles == ["biochemical_actions", "custom-my-targets"] and "custom-my-targets" in job.custom_profiles
    path = jobs_mod.profile_path("custom-my-targets", store.job_dir(job.id))
    assert path == store.job_dir(job.id) / "profiles" / "custom-my-targets.json" and path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["branch_weights"] == {"gene": 1.0, "chemical entity": 0.5} and saved["directionality"] == {"mode": "on", "inverse_factor": 0.2}
    profile = load_profile(path)
    assert profile.predicates and profile.category_weight("biolink:SmallMolecule") == 0.5 and profile.category_weight("biolink:Disease") == 0.0
    assert profile.entity_scope == ["gene", "chemical entity"] and profile.inverse_factor == 0.2
    # The free-text reading focus is whitespace-collapsed and capped, and reaches the reading prompt.
    assert profile.reading_focus.startswith("gene: regulated genes x") and len(profile.reading_focus) == jobs_mod.MAX_READING_FOCUS
    assert profile.reading_focus in __import__("biomedcat.prompts", fromlist=["ocr_description"]).ocr_description(profile.entity_scope, profile.reading_focus)
    # Derived weights are written inside the job for the trace; the reading profile unions both.
    assert (store.job_dir(job.id) / "profiles" / "derived" / "custom-my-targets.derived.json").is_file()
    merged = json.loads((store.job_dir(job.id) / "profile_merged.json").read_text(encoding="utf-8"))
    assert "gene" in merged["entity_scope"] and merged["members"] == job.profiles
    # Reloading the store keeps the custom profiles, and a bad custom profile is refused.
    assert store.get(job.id).custom_profiles["custom-my-targets"]["name"] == "My targets"
    with pytest.raises(ValueError):
        store.create("k", [{"file": "x.pdf", "stem": "x", "pages": 1}], [], custom_profiles=[{"name": "empty", "branch_weights": {"gene": 0}}])


# ------------------------------------------------------------------------------------------
# graphs
# ------------------------------------------------------------------------------------------

def test_cluster_mapping_uses_the_biolink_hierarchy():
    assert graphs_mod.cluster_for("biolink:Gene")[0] == "genes"
    assert graphs_mod.cluster_for("biolink:SmallMolecule")[0] == "chemicals"
    assert graphs_mod.cluster_for("biolink:PhenotypicFeature")[0] == "diseases"
    assert graphs_mod.cluster_for("biolink:CellularComponent")[0] == "anatomy"
    assert graphs_mod.cluster_for("biolink:Pathway")[0] == "processes"
    assert graphs_mod.cluster_for("biolink:OrganismTaxon")[0] == "organisms"
    assert graphs_mod.cluster_for("biolink:Procedure")[0] == "clinical"
    assert graphs_mod.class_name("biolink:GrossAnatomicalStructure") == "gross anatomical structure"


def test_build_and_filter_graph_json(graph_dir):
    g = graphs_mod.build_graph_json(graph_dir, "biochemical_actions", use_kg=False)
    nodes = [e["data"] for e in g["elements"] if "source" not in e["data"] and not e["data"].get("isCluster")]
    edges = [e["data"] for e in g["elements"] if "source" in e["data"]]
    assert len(nodes) == 4 and len(edges) == 3, "the edge to a missing node is dropped"
    seed = next(n for n in nodes if n["isSeed"])
    assert seed["parent"].startswith("cluster:") and seed["slides"] == ["1", "2"]
    assert {c["id"] for c in g["clusters"]} >= {"chemicals", "diseases", "processes"}
    assert all("color" in n for n in nodes) and len(g["legend"]) >= 3
    small = graphs_mod.filter_graph(g, max_nodes=2)
    kept = [e["data"] for e in small["elements"] if "source" not in e["data"] and not e["data"].get("isCluster")]
    assert len(kept) == 2 and any(n["isSeed"] for n in kept) and any(n["id"] == "CHEBI:2" for n in kept)
    (graph_dir / "graph.json").write_text(json.dumps(g), encoding="utf-8")
    graphs_mod.write_details_tables(graph_dir)
    assert (graph_dir / "node_details.csv").is_file() and (graph_dir / "edge_details.csv").is_file()


def test_zip_contains_trace_readme(tmp_path, graph_dir):
    job_dir = tmp_path / "jobs" / "j1"
    (job_dir / "graphs" / "doc" / "p").mkdir(parents=True)
    (job_dir / "graphs" / "doc" / "p" / "nodes.csv").write_text((graph_dir / "nodes.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (job_dir / "job.json").write_text("{}", encoding="utf-8")
    zip_path = graphs_mod.build_zip(job_dir, {"id": "j1", "name": "n", "documents": [], "profiles": ["p"], "progress": {"history": []}})
    import zipfile

    names = zipfile.ZipFile(zip_path).namelist()
    assert "j1/README_trace.md" in names and "j1/graphs/doc/p/nodes.csv" in names and not any(n.endswith(".zip") for n in names)


# ------------------------------------------------------------------------------------------
# enrichment statistics
# ------------------------------------------------------------------------------------------

def test_hypergeometric_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    for k, N, K, n in ((3, 1000, 50, 20), (1, 100, 10, 5), (5, 500, 5, 5), (0, 10, 3, 2)):
        assert abs(enrichment.hypergeom_sf(k, N, K, n) - scipy.hypergeom.sf(k - 1, N, K, n)) < 1e-9


def test_benjamini_hochberg_is_monotone_and_bounded():
    fdr = enrichment.benjamini_hochberg([0.01, 0.04, 0.03, 0.2, 0.5])
    assert all(0 <= q <= 1 for q in fdr)
    assert abs(fdr[0] - 0.05) < 1e-9 and fdr[4] == 0.5


# ------------------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from biomedcat.webapp import server

    server.store = jobs_mod.JobStore(root=tmp_path / "jobs", dataset_root=tmp_path / "dataset")
    server.worker = None
    monkeypatch.setattr(server, "_count_pages", lambda p: 3 if p.suffix == ".pdf" else 1)
    return TestClient(server.app)


def test_api_job_lifecycle(client):
    assert client.get("/").status_code == 200 and client.get("/viewer").status_code == 200
    profiles = client.get("/api/profiles").json()
    assert profiles and all("id" in p for p in profiles)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    files = [("files", ("deck/slide1.png", io.BytesIO(png), "image/png")), ("files", ("deck/notes.txt", io.BytesIO(b"x"), "text/plain")),
             ("files", ("deck/talk.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf"))]
    data = {"paths": json.dumps(["deck/slide1.png", "deck/notes.txt", "deck/talk.pdf"]), "name": "", "profiles": json.dumps([profiles[0]["id"]]),
            "options": json.dumps({"review": True, "offline": False, "gate": "union"})}
    r = client.post("/api/jobs", files=files, data=data)
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["name"] == "deck" and job["status"] == "queued" and len(job["documents"]) == 2, "the .txt is ignored"
    assert sum(d["pages"] for d in job["documents"]) == 4 and job["options"]["review"] is True
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]
    assert client.get(f"/api/jobs/{job['id']}").json()["estimate_s"] > 0
    assert client.post("/api/jobs/x/cancel").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["ok"] is True
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/requeue").json()["ok"] is True
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"
    assert client.get(f"/api/jobs/{job['id']}/graph?doc=slide1&profile=p").status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}").json()["ok"] is True
    assert client.get("/api/jobs").json() == []


def test_api_accepts_custom_profiles_and_lists_branches(client):
    assert {b["name"] for b in client.get("/api/branches").json()["branches"]} >= {"gene", "chemical entity"}
    custom = [{"name": "targets", "branch_weights": {"gene": 1, "polypeptide": 1, "chemical entity": 0.5}, "directionality": "off"}]
    r = client.post("/api/jobs", files=[("files", ("a.png", io.BytesIO(b"\x89PNG" + b"0" * 16), "image/png"))],
                    data={"paths": "[]", "profiles": "[]", "options": "{}", "name": "n", "custom_profiles": json.dumps(custom)})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["profiles"] == ["custom-targets"] and job["custom_profiles"]["custom-targets"]["directionality"]["mode"] == "off"
    r = client.post("/api/jobs", files=[("files", ("a.png", io.BytesIO(b"\x89PNG" + b"0" * 16), "image/png"))],
                    data={"paths": "[]", "profiles": "[]", "options": "{}", "name": "n", "custom_profiles": json.dumps([{"name": "x", "branch_weights": {}}])})
    assert r.status_code == 400


def test_api_rejects_unsupported_uploads(client):
    profiles = client.get("/api/profiles").json()
    r = client.post("/api/jobs", files=[("files", ("a.txt", io.BytesIO(b"x"), "text/plain"))],
                    data={"paths": "[]", "profiles": json.dumps([profiles[0]["id"]]), "options": "{}", "name": "n"})
    assert r.status_code == 400
    r = client.post("/api/jobs", files=[("files", ("a.png", io.BytesIO(b"x"), "image/png"))], data={"paths": "[]", "profiles": "[]", "options": "{}", "name": "n"})
    assert r.status_code == 400


# ------------------------------------------------------------------------------------------
# narrative summary
# ------------------------------------------------------------------------------------------

def test_graph_summary_bolds_every_node_and_converts_to_html(graph_dir):
    from biomedcat.webapp import summary as summary_mod

    g = graphs_mod.build_graph_json(graph_dir, "biochemical_actions", use_kg=False)
    (graph_dir / "graph.json").write_text(json.dumps(g), encoding="utf-8")
    md_path, html_path = summary_mod.write_graph_summary(graph_dir, {"id": "j1"}, doc="deck")
    md = md_path.read_text(encoding="utf-8")
    for name in ("DUX4", "losmapimod", "FSHD", "myogenesis"):
        assert f"**{name}**" in md
    assert "## Seeds" in md and "## Neighbourhood by group" in md and "Direct connections between seeds" not in md
    html = html_path.read_text(encoding="utf-8")
    assert "<b>DUX4</b>" in html and "<table>" in html and "<h2>" in html and "<script" not in html
