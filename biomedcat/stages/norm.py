"""Stage 3 (normalization): link each typed entity to a knowledge-base CURIE, or NIL.

Retrieval (retrieval.resolve_terms) supplies each entity's candidate pool; this stage runs
the 4-bit Llama as a judge that picks the type-consistent candidate, or abstains. Judging is
per-entity (batching it OOMs on the 8 GB card) and greedy (reproducible).
"""
import re
import logging

from biomedcat.runtime import build_llm, generate, free_gpu   # importing runtime bootstraps CUDA/determinism
from biomedcat.types import Entity, NormalizedEntity
from biomedcat import prompts
from biomedcat.retrieval import resolve_terms
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Choice(BaseModel):
    reasoning: str
    choice: int


choice_parser = PydanticOutputParser(pydantic_object=Choice)


def _parse_choice(reply: str) -> int | None:
    """Pull the integer choice out of the judge reply.

    Tries the structured parser first, then falls back to the first anchored 'choice' field,
    then the last integer. Returns None ONLY on a genuine parse failure (no integer at enough).
    """
    try:
        return choice_parser.parse(reply).choice
    except OutputParserException:
        anchored = re.search(r'"?choice"?\s*[:=]\s*(-?\d+)', reply, re.IGNORECASE)
        if anchored:
            return int(anchored.group(1))
        ints = re.findall(r"-?\d+", reply)
        return int(ints[-1]) if ints else None


def _build_menu(ranked: list) -> str:
    """Number the candidates for the．judge; option 0 is always 'none'.

    Each line shows the candidate's type in [brackets] so the judge can enforce type-consistency.
    """
    lines = ["0. none of the candidates"]
    for i, cand in enumerate(ranked, start=1):
        lines.append(f"{i}. {cand.label}  [{cand.biolink_type or '?'}]  ({cand.curie})")
    return "\n".join(lines)


def run_judge(entity: Entity, pool: list, model, tokenizer) -> str | None:
    """Select the best CURIE for one entity from its candidate pool, or None (N/A)."""
    if not pool:
        logger.info("  %r (%s): no candidates -> NIL", entity.text, entity.type)
        return None

    ranked = sorted(pool, key=lambda c: c.rank)          # global rank order: best first
    menu = _build_menu(ranked)
    messages = prompts.judge_messages(entity, menu, choice_parser.get_format_instructions())

    reply = generate(model, tokenizer, messages, max_new_tokens=384)
    choice_idx = _parse_choice(reply)

    if choice_idx is None:
        logger.warning("Unparseable judge reply for %r -> NIL: %.120s", entity.text, reply)
        return None
    if choice_idx <= 0 or choice_idx > len(ranked):      # 0, negative, or too large -> NIL
        logger.info("  %r -> NIL (judge chose %s)", entity.text, choice_idx)
        return None

    chosen = ranked[choice_idx - 1]
    logger.info("  %r -> %s (%s) [choice %d/%d]",
                entity.text, chosen.curie, chosen.label, choice_idx, len(ranked))
    return chosen.curie


def run_norm(entities: list[Entity], model_id: str | None = None) -> list[NormalizedEntity]:
    """Map each typed entity to a CURIE (or None).

    Retrieval runs first (network), before the judge model loads (GPU). The model is freed in
    a finally block (scale-to-zero), and per entity, to guard the 8 GB card against fragmentation.
    """
    if not entities:
        logger.info("No entities to normalize.")
        return []

    # Map each unique term to the set of types it appeared with (drives the type-constrained pass).
    term_types = {}
    for e in entities:
        term_types.setdefault(e.text, set()).add(e.type)
    logger.info("Normalizing %d mention(s) across %d unique term(s).", len(entities), len(term_types))

    # Phase 1: retrieval (network only) -- before the model touches the GPU.
    term_pools = resolve_terms(term_types)

    # Phase 2: judge (GPU).
    tokenizer, model = None, None
    results = []
    try:
        tokenizer, model = build_llm(model_id=model_id)
        for i, e in enumerate(entities, start=1):
            logger.info("[%d/%d] judging %r (%s)", i, len(entities), e.text, e.type)
            pool = term_pools.get(e.text, [])
            curie = run_judge(e, pool, model, tokenizer)
            results.append(NormalizedEntity(text=e.text, type=e.type, segment=e.segment, curie=curie))
            free_gpu()                                   # per-entity: fragmentation insurance on 8 GB
    finally:
        del model, tokenizer                             # scale-to-zero
        free_gpu()

    linked = 0
    for r in results:
        if r.curie:
            linked += 1
    logger.info("Normalization complete: %d/%d linked.", linked, len(results))
    return results
