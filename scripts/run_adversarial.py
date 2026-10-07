#!/usr/bin/env python
"""Run the NER agent and the linker on the gold terms in isolation (positives and adversarial).

Each term is presented to the agent as its own one-term sentence, the way the original test
harness did, so the measurement isolates the typing decision from the extraction step. The
adversarial terms (nonsense words, absurd compounds such as "Drug Receptorase") must come out
with no type and no CURIE; the rate at which they do not is the false-positive rate reported
by scripts/eval_gold.py.

This script calls the local LLM through Ollama for every term (about 1-3 agent steps per
term) and the name resolvers for typed terms: budget roughly one hour for 62 terms.

Usage (from the BiomedCAT root):
  uv run python scripts/run_adversarial.py --gold data/gold/fshd_slides_gold.json --out output/adversarial_run.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("run_adversarial")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/gold/fshd_slides_gold.json")
    parser.add_argument("--out", default="output/adversarial_run.json")
    parser.add_argument("--skip-linking", action="store_true", help="Only run the typing agent.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N terms of each group (smoke test).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from biomedcat.config import settings
    from biomedcat.stages.rag_engine import build_rag
    from biomedcat.stages.ner_agent import NERAgentPipeline
    from biomedcat.stages.norm import run_norm
    from biomedcat.types import Entity

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    agent = NERAgentPipeline(
        rag_unseen=build_rag(),
        classification_model_id=settings.classification_model_id,
        sanitization_model_id=settings.sanitization_model_id,
    )

    results: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for group in ("positives", "adversarial"):
        items = gold[group][: args.limit] if args.limit else gold[group]
        out_items = []
        for item in items:
            term = item["term"]
            logger.info("[%s] %r", group, term)
            entities = agent.extract([term])
            typed = [{"text": e.text, "type": e.type} for e in entities]
            record = {"id": item["id"], "term": term, "types": typed}
            if not args.skip_linking and entities:
                linked = run_norm([Entity(text=e.text, type=e.type, segment=term) for e in entities])
                record["curies"] = [{"text": r.text, "curie": r.curie, "kg2c_id": r.kg2c_id, "label": r.label} for r in linked if r.curie]
            out_items.append(record)
        results[group] = out_items

    payload = {
        "gold": args.gold,
        "models": {"ner": settings.classification_model_id, "sanitizer": settings.sanitization_model_id},
        "elapsed_s": round(time.perf_counter() - t0, 1),
        **results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    typed_adv = sum(1 for r in results["adversarial"] if r["types"])
    logger.info("done in %.0f s: %d/%d adversarial terms typed -> %s", payload["elapsed_s"], typed_adv, len(results["adversarial"]), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
