"""Stage 3 (normalization): link each typed entity to a knowledge-base CURIE, or NIL.

Retrieval (retrieval.resolve_terms) supplies each entity's candidate pool from the RENCI name
resolver and the ARAX entity service, already aligned to RTX-KG2c ids through the offline
equivalents table. This stage then runs the local LLM (through Ollama) as a judge that picks
the type-consistent candidate, or abstains. Judging is per entity and greedy (temperature 0),
so runs are reproducible.
"""
import json
import re
import logging

from pydantic import BaseModel, ValidationError

from biomedcat.runtime import generate
from biomedcat.types import Entity, NormalizedEntity
from biomedcat import prompts
from biomedcat.retrieval import resolve_terms
from biomedcat.config import settings

logger = logging.getLogger(__name__)


class Choice(BaseModel):
    reasoning: str = ""
    choice: int


FORMAT_INSTRUCTIONS = (
    'Answer with a single JSON object and nothing else, in the form '
    '{"reasoning": "<one sentence>", "choice": <integer>}.'
)


def _parse_choice(reply: str) -> int | None:
    """Pull the integer choice out of the judge reply.

    Tries the JSON object first, then an anchored 'choice' field, then the last integer in the
    reply. Returns None only on a genuine parse failure (no integer at all).
    """
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            return Choice.model_validate_json(match.group(0)).choice
        except (ValidationError, json.JSONDecodeError):
            pass
    anchored = re.search(r'"?choice"?\s*[:=]\s*(-?\d+)', reply, re.IGNORECASE)
    if anchored:
        return int(anchored.group(1))
    ints = re.findall(r"-?\d+", reply)
    return int(ints[-1]) if ints else None


def _build_menu(ranked: list) -> str:
    """Number the candidates for the judge; option 0 is always 'none'.

    Each line shows the candidate's type in [brackets] so the judge can enforce type
    consistency, and flags whether the candidate exists in the knowledge graph.
    """
    lines = ["0. none of the candidates"]
    for i, cand in enumerate(ranked, start=1):
        in_kg = "in KG2" if cand.kg2c_id else "not in KG2"
        lines.append(f"{i}. {cand.label}  [{cand.biolink_type or '?'}]  ({cand.curie}, {in_kg})")
    return "\n".join(lines)


def run_judge(entity: Entity, pool: list, model_id: str):
    """Select the best candidate for one entity from its pool; return the Candidate or None (NIL)."""
    if not pool:
        logger.info("  %r (%s): no candidates -> NIL", entity.text, entity.type)
        return None

    ranked = sorted(pool, key=lambda c: c.rank)          # global rank order: best first
    menu = _build_menu(ranked)
    messages = prompts.judge_messages(entity, menu, FORMAT_INSTRUCTIONS)

    reply = generate(model_id, messages, max_new_tokens=384, temperature=0.0)
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
    return chosen


def run_norm(entities: list[Entity], model_id: str | None = None) -> list[NormalizedEntity]:
    """Map each typed entity to a CURIE (or None) and to its RTX-KG2c canonical id.

    Retrieval runs first (network only); the judge model is then called through Ollama once per
    entity. Identical (text, type) mentions share one judgement so the LLM is not asked twice.
    """
    if not entities:
        logger.info("No entities to normalize.")
        return []

    model_id = model_id or settings.classification_model_id

    # Map each unique term to the set of types it appeared with (drives the type-constrained pass).
    term_types: dict[str, set[str]] = {}
    for e in entities:
        term_types.setdefault(e.text, set()).add(e.type)
    logger.info("Normalizing %d mention(s) across %d unique term(s).", len(entities), len(term_types))

    # Phase 1: retrieval (network + offline canonicalization).
    term_pools = resolve_terms(term_types)

    # Phase 2: judge (LLM), one call per distinct (text, type).
    decisions: dict[tuple[str, str], object] = {}
    results: list[NormalizedEntity] = []
    for i, e in enumerate(entities, start=1):
        key = (e.text, e.type)
        if key not in decisions:
            logger.info("[%d/%d] judging %r (%s)", i, len(entities), e.text, e.type)
            decisions[key] = run_judge(e, term_pools.get(e.text, []), model_id)
        chosen = decisions[key]
        results.append(NormalizedEntity(
            text=e.text, type=e.type, segment=e.segment, page=e.page,
            curie=chosen.curie if chosen else None,
            kg2c_id=chosen.kg2c_id if chosen else None,
            label=chosen.label if chosen else None,
        ))

    linked = sum(1 for r in results if r.curie)
    in_kg = sum(1 for r in results if r.kg2c_id)
    logger.info("Normalization complete: %d/%d linked, %d/%d in KG2c.", linked, len(results), in_kg, len(results))
    return results
