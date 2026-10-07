#!/usr/bin/env python
"""Existence-gate ablation on the gold test suite: true and false rejections per mode.

The gate runs post hoc on the raw terms of the suite (31 expert positives, 31 adversarial
inventions), in each mode: "es" (no candidate in the Elasticsearch name resolver), "llm" (the
existence model does not recognise the term), "union" (either). A rejected adversarial term is a
true rejection; a rejected positive is a false rejection. No agent call is involved, so this is
a property of the gate alone.

Usage:
  uv run python scripts/eval_gate_modes.py --gold data/gold/fshd_slides_gold.json --out data/eval/gate_modes.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/gold/fshd_slides_gold.json")
    parser.add_argument("--out", default="data/eval/gate_modes.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from biomedcat.config import settings
    from biomedcat.stages.rag_engine import build_rag
    from biomedcat.stages.ner_agent import NERAgentPipeline

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    positives = [g["term"] for g in gold["positives"]]
    adversarial = [g["term"] for g in gold["adversarial"]]
    agent = NERAgentPipeline(build_rag(), settings.classification_model_id, settings.sanitization_model_id)

    report = {"model": settings.existence_model_id, "es_url": settings.nameres_es_url, "n_positives": len(positives),
              "n_adversarial": len(adversarial), "modes": {}}
    for mode in ("es", "llm", "union"):
        settings.existence_gate = mode
        keep = agent.existence_gate(positives + adversarial)
        false_rej = [t for t in positives if not keep.get(t.strip().lower(), True)]
        true_rej = [t for t in adversarial if not keep.get(t.strip().lower(), True)]
        report["modes"][mode] = {"true_rejections": len(true_rej), "false_rejections": len(false_rej),
                                 "adversarial_rejected": true_rej, "positives_rejected": false_rej}
        logging.info("%s: %d/%d adversarial rejected, %d/%d positives rejected", mode, len(true_rej), len(adversarial), len(false_rej), len(positives))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
