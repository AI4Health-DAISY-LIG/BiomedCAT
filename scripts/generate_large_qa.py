#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to generate the three large QA splits (seeds 77, 78, 79).

Each split is saved under ``data/large_splits/seed_<seed>/qa_dataset.parquet``.
The script re‑uses the ``StratifiedQAGenerator`` defined in
``scripts/generate_qa_data.py``.
"""

import json
import logging
import random
from pathlib import Path

import yaml

from biomedcat.config import Settings
from biomedcat.stages.rag_engine import build_rag
from scripts.generate_qa_data import StratifiedQAGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

CONFIG_PATH = Path("scripts/ner_test-suite_configuration.yml")
NESTED_DATA_PATH = Path("data/biolink_classes_nested.json")

def main():
    # Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    settings = Settings()
    rag = build_rag(settings)

    # Load nested ontology
    with open(NESTED_DATA_PATH, "r", encoding="utf-8") as f:
        nested = json.load(f)

    # Generate three deterministic splits
    for seed in (77, 78, 79):
        logging.info(f"\n=== Génération du split seed={seed} ===")
        gen = StratifiedQAGenerator(
            rag_engine=rag,
            nested_data=nested,
            config=cfg
        )
        # Override the internal RNG seed
        gen.random = random.Random(seed)
        gen.run()
        # Save the split
        gen.save_split(gen.df, f"seed_{seed}")

if __name__ == "__main__":
    main()
