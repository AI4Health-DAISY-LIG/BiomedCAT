"""Entity-branch weight derivation (biomedcat.weights) and the profile loader. No model, no
network, no DuckDB: only data/biolink_strata.json and data/kg2c/stats.json are read."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from biomedcat import weights as W
from biomedcat.config import ROOT_PATH

STRATA = Path(ROOT_PATH) / "data" / "biolink_strata.json"
KG_DIR = Path(ROOT_PATH) / "data" / "kg2c"


@pytest.fixture(scope="module")
def model() -> W.BiolinkModel:
    return W.BiolinkModel.load(STRATA)


def test_branches_split_biological_entity(model):
    branches = model.branches()
    assert "gene" in branches and "chemical entity" in branches and "disease or phenotypic feature" in branches
    assert "biological entity" not in branches and "named thing" not in branches
    assert model.branch_of("small molecule") == "chemical entity"
    assert model.branch_of("protein") == "polypeptide"
    assert model.branch_of("disease") == "disease or phenotypic feature"
    assert model.branch_of("named thing") is None
    assert W.camel_curie("small molecule") == "biolink:SmallMolecule" and W.camel_curie("RNA product") == "biolink:RNAProduct"


def test_branch_weights_are_snapped_and_validated(model):
    bw = W.normalise_branch_weights({"gene": 0.7, "chemical entity": "1", "not a class": 1, "biolink:disease_or_phenotypic_feature": 0.4}, model)
    assert bw == {"gene": 0.5, "chemical entity": 1.0, "disease or phenotypic feature": 0.5}


def test_class_weights_inherit_from_the_branch(model):
    cw = W.class_weights({"chemical entity": 1.0, "gene": 0.5}, model)
    assert cw["small molecule"] == 1.0 and cw["drug"] == 1.0
    assert cw["gene"] == 0.5 and cw["disease"] == 0.0 and cw["protein"] == 0.0
    # Roots above the branches take the maximum below them; mixins the maximum of their members.
    assert cw["named thing"] == 1.0 and cw["biological entity"] == 0.5
    assert cw["gene or gene product"] == 0.5 and cw["chemical entity or gene or gene product"] == 1.0


def test_predicate_scope_and_priority(model):
    cw = W.class_weights({"gene": 1.0, "chemical entity": 0.5}, model)
    # has phenotype: biological entity -> phenotypic feature; no phenotype branch weighted.
    assert W.scope_weight("phenotypic feature", cw, model) == 0.0
    assert W.scope_weight("biological entity", cw, model) == 1.0
    assert W.scope_weight("named thing", cw, model) == 1.0
    assert W.scope_weight("chemical or drug or treatment", cw, model) == 0.5
    assert W.predicate_priority(1.0, 0.0) == 0.0
    assert W.predicate_priority(1.0, 0.5) == 1.0 and W.predicate_priority(1.0, 0.5, "min") == 0.5


def test_derive_normalises_each_branch_to_its_priority(model):
    bw = {"gene": 1.0, "chemical entity": 1.0, "biological process or activity": 0.5}
    edges = {"biolink:interacts_with": 100, "biolink:physically_interacts_with": 50, "biolink:directly_physically_interacts_with": 10,
             "biolink:affects": 100, "biolink:regulates": 20, "biolink:has_phenotype": 30, "biolink:gene_product_of": 5,
             "biolink:same_as": 7}
    d = W.derive(bw, model, alpha=0.7, edge_counts=edges)
    # interacts with: three predicates present, the branch sums to 1 and rewards depth.
    inter = {p: w for p, w in d.predicates.items() if "interacts_with" in p}
    assert abs(sum(inter.values()) - 1.0) < 1e-6
    assert inter["biolink:directly_physically_interacts_with"] > inter["biolink:physically_interacts_with"] > inter["biolink:interacts_with"]
    ratio = inter["biolink:physically_interacts_with"] / inter["biolink:interacts_with"]
    assert abs(ratio - math.exp(0.7)) < 1e-3  # weights are rounded to 6 decimals
    # has phenotype: no disease or phenotype branch weighted -> out of scope.
    assert "biolink:has_phenotype" not in d.predicates
    # Representation predicates stay at 1, equivalence predicates never appear.
    assert d.predicates["biolink:gene_product_of"] == 1.0 and "biolink:same_as" not in d.predicates
    assert "biolink:interacts_with" in d.symmetric_predicates and "biolink:affects" not in d.symmetric_predicates
    # A predicate without kept edges gets no weight and does not dilute its branch.
    assert "biolink:binds" not in d.predicates
    assert d.category_weights["biolink:SmallMolecule"] == 1.0 and d.category_weights["biolink:Disease"] == 0.0


def test_min_scope_rule_lowers_predicates_towards_secondary_branches(model):
    bw = {"gene": 1.0, "disease or phenotypic feature": 0.5}
    edges = {"biolink:gene_associated_with_condition": 10, "biolink:associated_with": 10}
    d_max = W.derive(bw, model, edge_counts=edges, scope_rule="max")
    d_min = W.derive(bw, model, edge_counts=edges, scope_rule="min")
    assert d_max.branch_priorities["associated with"] == 1.0
    assert d_min.branch_priorities["associated with"] == 1.0  # associated with is unconstrained (named thing)
    row_max = next(r for r in d_max.review if r["predicate"] == "biolink:gene_associated_with_condition")
    row_min = next(r for r in d_min.review if r["predicate"] == "biolink:gene_associated_with_condition")
    assert row_max["priority"] == 1.0 and row_min["priority"] == 0.5


def test_directionality_block():
    assert W.directionality_of({}) == {"mode": "on", "inverse_factor": 0.5}
    assert W.directionality_of({"directionality": "off"}) == {"mode": "off", "inverse_factor": 0.5}
    assert W.directionality_of({"directionality": {"mode": "on", "inverse_factor": 2}})["inverse_factor"] == 1.0


def test_derive_for_profile_writes_and_reuses_cache(tmp_path):
    profile = {"name": "custom", "branch_weights": {"gene": 1, "chemical entity": 0.5}, "directionality": "on"}
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    first = W.derive_for_profile(profile, path, STRATA, KG_DIR)
    json_path, md_path = W.derived_paths(path)
    assert json_path.is_file() and md_path.is_file() and first["n_predicates"] > 0
    assert first["class_weights"]["small molecule"] == 0.5
    stamp = json_path.stat().st_mtime_ns
    second = W.derive_for_profile(profile, path, STRATA, KG_DIR)
    assert second["fingerprint"] == first["fingerprint"] and json_path.stat().st_mtime_ns == stamp, "unchanged inputs: cache reused"
    profile["branch_weights"]["gene"] = 0.5
    third = W.derive_for_profile(profile, path, STRATA, KG_DIR)
    assert third["fingerprint"] != first["fingerprint"]


def test_default_profiles_load_with_derived_weights():
    from biomedcat.profiles import load_profile

    bio = load_profile(Path(ROOT_PATH) / "data" / "profiles" / "biochemical_actions.json")
    cli = load_profile(Path(ROOT_PATH) / "data" / "profiles" / "clinical_mechanisms.json")
    uni = load_profile(Path(ROOT_PATH) / "data" / "profiles" / "uniform.json")
    for p in (bio, cli):
        assert p.predicates and p.branch_weights and p.category_weights and p.directed and p.inverse_factor == 0.5
        assert p.symmetric_predicates and p.entity_scope and p.derived_path.endswith(".derived.json")
        assert p.category_weight("biolink:Publication") == 0.0 and p.category_weight("biolink:UnknownThing") == 1.0
    assert bio.category_weight("biolink:Gene") == 1.0 and cli.category_weight("biolink:Disease") == 1.0
    assert bio.category_weight("biolink:OrganismTaxon") == 0.0
    # The baseline keeps its explicit flat weights and an undirected traversal.
    assert not uni.directed and uni.inverse_factor == 1.0 and all(w == 1.0 for w in uni.predicates.values())
    assert all(w == 1.0 for w in uni.category_weights.values())


def test_legacy_flat_profile_still_loads(tmp_path):
    from biomedcat.profiles import load_profile

    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"biolink:affects": 0.5}), encoding="utf-8")
    p = load_profile(path)
    assert p.predicates == {"biolink:affects": 0.5} and not p.directed and not p.category_weights
