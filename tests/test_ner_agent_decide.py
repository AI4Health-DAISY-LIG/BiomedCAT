"""Unit tests of the 'search then decide' agent mode (AGENT_MODE=decide), without Ollama.

A fake retriever exposes a handful of real Biolink class names (they must be real: the verdict
is validated against the Biolink vocabulary) with a small is_a structure; the model call is
replaced by a stub that records how many times it was called.
"""
from __future__ import annotations

import os

import pytest

from biomedcat.stages import ner_agent as na


class FakeRag:
    """Six real Biolink classes: disease and phenotypic feature under 'disease or phenotypic
    feature'; gene, protein and protein isoform (protein isoform is a child of protein)."""

    def __init__(self):
        def entry(parent, children, abstract=False):
            return {"metadata": {"parent": parent, "children": children, "definition": f"def of the class", "abstract": abstract}}
        self.flat_data = {
            "disease or phenotypic feature": entry("biological entity", ["disease", "phenotypic feature"]),
            "disease": entry("disease or phenotypic feature", []),
            "phenotypic feature": entry("disease or phenotypic feature", []),
            "gene": entry("biological entity", []),
            "protein": entry("biological entity", ["protein isoform"]),
            "protein isoform": entry("protein", []),
            "biological entity": entry("named thing", ["disease or phenotypic feature", "gene", "protein"], abstract=True),
        }
        self.exemplars = None
        self.bm25 = None

    def _indexable(self, meta):
        return not meta.get("abstract") and not meta.get("deprecated")

    def search(self, query, top_k=5):
        return ["disease", "gene", "protein", "phenotypic feature"][:top_k]


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "decide")
    monkeypatch.setenv("AGENT_DECIDE_TOP_K", "2")
    monkeypatch.setenv("AGENT_DECIDE_NEIGHBOURS_OF", "1")
    monkeypatch.setenv("AGENT_DECIDE_CAP", "4")
    return na.NERAgentPipeline(FakeRag(), "fake-model", "fake-sanitizer")


def test_neighbours_are_strict_is_a_and_assignable(agent):
    n = agent._neighbours("disease")
    assert "phenotypic feature" in n                 # sibling
    assert "disease or phenotypic feature" in n      # parent
    assert "biological entity" not in n              # abstract, never a verdict
    assert "disease" not in n
    n = agent._neighbours("protein")
    assert n[0] == "protein isoform"                 # children first
    assert "gene" in n and "disease or phenotypic feature" in n   # siblings under the (abstract) parent are assignable
    assert "biological entity" not in n and "protein" not in n


def test_candidates_dedupe_and_cap(agent):
    cands = agent._decide_candidates("anything")
    # top-2 (disease, gene) + neighbours of the first (phenotypic feature, disease or phenotypic feature), capped at 4
    assert cands == ["disease", "gene", "phenotypic feature", "disease or phenotypic feature"]
    assert len(cands) == len(set(cands)) <= 4


def test_prompt_has_context_only_in_document_mode(agent):
    cands = agent._decide_candidates("x")
    bench = agent._decide_prompt("FSHD", "FSHD", cands)          # benchmark: sentence == term
    doc = agent._decide_prompt("FSHD", "FSHD is caused by DUX4.", cands)
    assert "Sentence:" not in bench and "identifiable people" not in bench
    assert "Sentence: FSHD is caused by DUX4." in doc and "identifiable people" in doc
    assert bench.rstrip().endswith("FINAL_VERDICT: <class name>")


def test_decide_accepts_candidate_rejects_outsider_and_none(agent, monkeypatch):
    calls = []

    def fake_generate(model, messages, max_new_tokens, temperature=0.0, think=None, raw=False):
        calls.append((messages[0]["content"], think, raw))
        return fake_generate.reply

    monkeypatch.setattr(na, "generate", fake_generate)
    fake_generate.reply = "FINAL_VERDICT: gene"
    assert agent._run_agentic_loop("DUX4", "DUX4") == "gene"
    fake_generate.reply = "FINAL_VERDICT: small molecule"        # a real class, but not in the list
    assert agent._run_agentic_loop("term2", "term2") is None
    fake_generate.reply = "I would say it is a phenotypic feature.\nFINAL_VERDICT: phenotype"   # recovered by citation
    assert agent._run_agentic_loop("term3", "term3") == "phenotypic feature"
    fake_generate.reply = "FINAL_VERDICT: none"
    assert agent._run_agentic_loop("Dumonceaux", "Dumonceaux") == "NONE"
    # one model call per term, hidden reasoning requested, and no tool loop
    assert len(calls) == 4 and all(think is True and raw is True for _, think, raw in calls)
    assert all("ACTION:" not in prompt for prompt, _, _ in calls)


def test_react_mode_untouched(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "react")
    a = na.NERAgentPipeline(FakeRag(), "fake-model", "fake-sanitizer")
    assert a.mode == "react"


def test_short_definition_ends_on_sentence_or_word():
    f = na.NERAgentPipeline._short_definition
    assert f("A short one.") == "A short one."
    assert f("  folds\n whitespace  ") == "folds whitespace"
    long_first = "A" * 300 + ". Second sentence."
    assert f(long_first).endswith("…") and len(f(long_first)) <= 251 and " " not in f(long_first)[-1]
    two = "First sentence about a gene. " + "x" * 300
    assert f(two) == "First sentence about a gene."
    words = " ".join(["word"] * 100)
    out = f(words)
    assert out.endswith("…") and len(out) <= 251 and not out[:-1].endswith("wor")
    assert f("x" * 40, limit=10) == "x" * 10 + "…"
