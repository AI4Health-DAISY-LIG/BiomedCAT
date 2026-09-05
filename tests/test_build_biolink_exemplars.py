"""Tests for scripts/build_biolink_exemplars.py.

Covers the pure helpers (usability filter, form classification) and the sampling function
against small synthetic KG2c-shaped frames, so the KG2c mapping and dedup logic are checked
without needing the real 36 MB nodes.parquet.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_biolink_exemplars import _classify_form, _is_usable, _sample_class  # noqa: E402


class TestIsUsable:
    def test_rejects_empty(self):
        assert not _is_usable("")
        assert not _is_usable(None)

    def test_rejects_sentence_like_text(self):
        assert not _is_usable("a b c d e f g")  # 7 words, over MAX_WORDS

    def test_rejects_overly_long_single_token(self):
        assert not _is_usable("x" * 61)

    def test_accepts_typical_names(self):
        assert _is_usable("diabetes")
        assert _is_usable("chronic obstructive pulmonary disease")
        assert _is_usable("DM")


class TestClassifyForm:
    def test_canonical_flag_wins(self):
        assert _classify_form("anything", is_canonical=True) == "canonical"

    def test_short_uppercase_is_abbreviation(self):
        assert _classify_form("TB", is_canonical=False) == "abbreviation"
        assert _classify_form("RANKL", is_canonical=False) == "abbreviation"

    def test_three_plus_words_is_compound(self):
        assert _classify_form("lung squamous cell carcinoma", is_canonical=False) == "compound"

    def test_short_lowercase_phrase_is_variant(self):
        assert _classify_form("Alzheimers", is_canonical=False) == "variant"


class TestSampleClass:
    """Exercise _sample_class against a tiny synthetic KG2c-shaped pair of frames."""

    def _frames(self):
        nodes = pd.DataFrame([
            {"id": "MONDO:1", "name": "diabetes", "category": "biolink:Disease"},
            {"id": "MONDO:2", "name": "diabetes", "category": "biolink:Disease"},  # duplicate name, different node
            {"id": "MONDO:3", "name": "cancer", "category": "biolink:Disease"},
            {"id": "MONDO:4", "name": "a sentence-like description that is far too long to keep as an exemplar", "category": "biolink:Disease"},
            {"id": "NCBIGene:1", "name": "BRCA1", "category": "biolink:Gene"},
        ])
        details = pd.DataFrame([
            {"id": "MONDO:1", "synonyms": ["DM", "Diabetes Mellitus"]},
            {"id": "MONDO:3", "synonyms": ["malignant neoplasm"]},
        ])
        return nodes, details

    def test_dedupes_repeated_canonical_names(self):
        nodes, details = self._frames()
        rows = _sample_class("disease", {"depth": 4, "parent": "disease or phenotypic feature"},
                              nodes, details, target=10, max_synonyms_per_node=2, rng=random.Random(0))
        names = [r["exemplar"] for r in rows if r["form"] == "canonical"]
        assert names.count("diabetes") == 1  # the two MONDO:1/MONDO:2 duplicates collapse to one row

    def test_drops_sentence_like_names(self):
        nodes, details = self._frames()
        rows = _sample_class("disease", {"depth": 4, "parent": "disease or phenotypic feature"},
                              nodes, details, target=10, max_synonyms_per_node=2, rng=random.Random(0))
        assert all("sentence-like" not in r["exemplar"] for r in rows)

    def test_fills_from_synonyms_when_target_exceeds_node_count(self):
        nodes, details = self._frames()
        rows = _sample_class("disease", {"depth": 4, "parent": "disease or phenotypic feature"},
                              nodes, details, target=10, max_synonyms_per_node=2, rng=random.Random(0))
        forms = {r["form"] for r in rows}
        assert "canonical" in forms
        assert forms & {"abbreviation", "compound", "variant"}  # at least one synonym-derived row

    def test_unmapped_category_returns_empty(self):
        nodes, details = self._frames()
        rows = _sample_class("procedure", {"depth": 2, "parent": "activity"},
                              nodes, details, target=10, max_synonyms_per_node=2, rng=random.Random(0))
        assert rows == []

    def test_respects_target_cap(self):
        nodes, details = self._frames()
        rows = _sample_class("disease", {"depth": 4, "parent": "disease or phenotypic feature"},
                              nodes, details, target=2, max_synonyms_per_node=2, rng=random.Random(0))
        assert len(rows) == 2

    def test_no_node_id_leaks_into_output_rows(self):
        nodes, details = self._frames()
        rows = _sample_class("disease", {"depth": 4, "parent": "disease or phenotypic feature"},
                              nodes, details, target=10, max_synonyms_per_node=2, rng=random.Random(0))
        assert all("_node_id" not in r for r in rows)
        for r in rows:
            assert set(r) == {"class", "exemplar", "form", "depth", "parent"}
