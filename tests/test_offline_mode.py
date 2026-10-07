"""Offline mode: no term may leave the machine and no network call may block a run.

Two guarantees, both without Ollama or network:
  - the existence gate drops its name-resolver leg when RESOLVERS_OFFLINE is on (union -> llm,
    nameres -> off), so no term is sent to the Name Resolver;
  - the Biolink model update check falls back to the cached YAML when offline or when GitHub is
    unreachable, and accepts a local YAML path without contacting GitHub.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from biomedcat.stages import biolink_yml_processor as byp
from biomedcat.stages import ner_agent as na
from tests.test_ner_agent_decide import FakeRag


# --- existence gate -------------------------------------------------------------------------
@pytest.fixture
def gate_agent(monkeypatch):
    agent = na.NERAgentPipeline(FakeRag(), "fake-model", "fake-guard")
    calls = {"es": [], "llm": []}

    def es(term):
        calls["es"].append(term)
        return False  # the resolver would reject everything

    def llm(terms, batch=40):
        calls["llm"].append(list(terms))
        return ["biozenase"]

    monkeypatch.setattr(agent, "_nameres_has_candidate", es)
    monkeypatch.setattr(agent, "_llm_unknown_terms", llm)
    return agent, calls


@pytest.mark.parametrize("mode, expected_mode_calls", [("union", "llm"), ("nameres", None), ("es", None)])  # "es" = legacy alias
def test_offline_gate_never_calls_the_resolver(gate_agent, monkeypatch, mode, expected_mode_calls):
    agent, calls = gate_agent
    monkeypatch.setattr(na.settings, "resolvers_offline", True)
    monkeypatch.setattr(na.settings, "existence_gate", mode)
    keep = agent.existence_gate(["DUX4", "Biozenase"])
    assert calls["es"] == []
    if expected_mode_calls == "llm":
        assert calls["llm"] == [["Biozenase", "DUX4"]]  # terms are sorted before the batched call
        assert keep == {"dux4": True, "biozenase": False}
    else:
        assert calls["llm"] == []
        assert keep == {"dux4": True, "biozenase": True}


def test_online_gate_still_uses_the_resolver(gate_agent, monkeypatch):
    agent, calls = gate_agent
    monkeypatch.setattr(na.settings, "resolvers_offline", False)
    monkeypatch.setattr(na.settings, "existence_gate", "nameres")
    keep = agent.existence_gate(["DUX4"])
    assert calls["es"] == ["DUX4"]
    assert keep == {"dux4": False}


# --- Biolink model update check -------------------------------------------------------------
GITHUB = "https://github.com/biolink/biolink-model/blob/v4.4.4/biolink-model.yaml"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Cached YAML, class tables and cache file redirected to a temporary directory."""
    yaml = tmp_path / "biolink-model.yaml"
    flat, nested = tmp_path / "flat.json", tmp_path / "nested.json"
    monkeypatch.setattr(byp, "LOCAL_YAML", yaml)
    monkeypatch.setattr(byp, "OUTPUT_FLAT", str(flat))
    monkeypatch.setattr(byp, "OUTPUT_NESTED", str(nested))
    monkeypatch.setattr(byp.settings, "resolvers_offline", False)
    ran = []

    def processor(source):
        ran.append(source)
        flat.write_text("{}"); nested.write_text("{}")

    def unreachable(url):
        raise RuntimeError("Could not reach GitHub API: simulated")

    monkeypatch.setattr(byp, "get_remote_blob_sha", unreachable)
    return dict(yaml=yaml, flat=flat, nested=nested, cache=tmp_path / "cache.json", ran=ran, processor=processor)


def test_local_yaml_path_is_processed_once_without_github(sandbox, tmp_path):
    local = tmp_path / "my-biolink.yaml"
    local.write_text("classes: {}")
    assert byp.run_smart_update(str(local), sandbox["processor"], cache_file=sandbox["cache"]) is True
    assert byp.run_smart_update(str(local), sandbox["processor"], cache_file=sandbox["cache"]) is False
    assert sandbox["ran"] == [str(local)]


def test_offline_with_tables_present_uses_them(sandbox, monkeypatch):
    monkeypatch.setattr(byp.settings, "resolvers_offline", True)
    sandbox["flat"].write_text("{}"); sandbox["nested"].write_text("{}")
    assert byp.run_smart_update(GITHUB, sandbox["processor"], cache_file=sandbox["cache"]) is False
    assert sandbox["ran"] == []


def test_offline_without_tables_rebuilds_from_cached_yaml(sandbox, monkeypatch):
    monkeypatch.setattr(byp.settings, "resolvers_offline", True)
    sandbox["yaml"].write_text("classes: {}")
    assert byp.run_smart_update(GITHUB, sandbox["processor"], cache_file=sandbox["cache"]) is True
    assert sandbox["ran"] == [str(sandbox["yaml"])]


def test_unreachable_github_falls_back_to_cached_yaml(sandbox):
    sandbox["yaml"].write_text("classes: {}")
    assert byp.run_smart_update(GITHUB, sandbox["processor"], cache_file=sandbox["cache"]) is True
    assert sandbox["ran"] == [str(sandbox["yaml"])]


def test_no_network_and_no_cache_is_an_explicit_error(sandbox):
    with pytest.raises(RuntimeError, match="no cached Biolink model"):
        byp.run_smart_update(GITHUB, sandbox["processor"], cache_file=sandbox["cache"])
