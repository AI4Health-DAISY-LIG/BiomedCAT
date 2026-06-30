import os

# CUDA env vars must be set BEFORE torch is imported so the allocator reads them.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce 8GB-VRAM fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                   # needed for deterministic cuBLAS

import gc
import re
import asyncio
import logging
import concurrent.futures

import aiohttp
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

# Determinism: greedy decoding + fixed seed so the judge is reproducible across
# process launches (4-bit greedy output otherwise drifts run-to-run).
# warn_only=True: ops without a deterministic kernel fall back instead of crashing.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RENCI_URL = "https://name-resolution-sri.renci.org/lookup"
ARAX_URL  = "https://arax.ncats.io/api/arax/v1.4/entity"
API_LIMIT = 10   # top-10 candidates from EACH of RENCI and ARAX (pooled per entity)
MAX_CONCURRENT_REQUESTS = 10

LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

class Choice(BaseModel):
    reasoning: str
    choice: int

choice_parser = PydanticOutputParser(pydantic_object=Choice)

JUDGE_SYSTEM = (
    "You are a biomedical entity normalization expert. Given an entity, the sentence it "
    "appears in, and a numbered list of candidate identifiers, pick the single candidate "
    "that best matches the entity in that context, or 0 if none of them fit."
)

# Short gloss per entity type, handed to the judge so it knows what the type label means.
# Kept identical to ner.py's type_defs so the two stages describe types the same way.
TYPE_DEFINITIONS = {
    "GENE":                    "a gene or gene symbol",
    "DISEASE":                 "a disease or disorder",
    "CHEMICAL":                "a chemical, drug, or compound",
    "CELL_TYPE":               "a type of cell",
    "ANATOMY":                 "an anatomical structure, tissue, or organ",
    "CHROMOSOMAL_LOCUS":       "a chromosomal location, locus, or genomic region",
    "EPIGENETIC_MODIFICATION": "an epigenetic modification such as methylation",
}

# NER type -> RENCI biolink_type filter for the type-constrained retrieval pass, which pulls
# correctly-typed candidates (e.g. UBERON for ANATOMY) the unconstrained lexical lookup misses.
# Only types with a clean biolink category are mapped; the biolink_type field is OPTIONAL in the
# RENCI call, so a type that is not here (e.g. CHROMOSOMAL_LOCUS, EPIGENETIC_MODIFICATION) simply
# gets no constrained pass and relies on the unconstrained lookup.
TYPE_TO_BIOLINK = {
    "GENE":      "biolink:Gene",
    "DISEASE":   "biolink:Disease",
    "CHEMICAL":  "biolink:ChemicalEntity",
    "CELL_TYPE": "biolink:Cell",
    "ANATOMY":   "biolink:AnatomicalEntity",
}

def free_gpu():
    """Forces aggressive VRAM clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- ASYNC RETRIEVAL LOGIC (From Refactored) ---
async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # One failed lookup must not kill the run, but it must be VISIBLE:
            # a silently-swallowed API error looks identical to "no candidates".
            logger.warning("Lookup failed (%s): %s", url, e)
            return None

async def async_renci_lookup(session, term, semaphore, biolink_type=None):
    params = {"string": term, "limit": API_LIMIT}
    if biolink_type:                         # type-constrained pass: ask RENCI for this category only
        params["biolink_type"] = biolink_type
    data = await fetch_json(session, RENCI_URL, params, semaphore)
    if not data: return []
    return [
        {
            "curie": hit["curie"],
            "label": hit.get("label", ""),
            "biolink_type": next(iter(hit.get("types", [])), None),
            "rank": rank,
            "source": "RENCI",
        }
        for rank, hit in enumerate(data[:API_LIMIT]) if hit.get("curie")
    ]

async def async_arax_lookup(session, term, semaphore):
    data = await fetch_json(session, ARAX_URL, [("q", term)], semaphore)
    entry = data.get(term) if data else None
    if not entry: return []
    
    canon = entry.get("id", {})
    nodes = entry.get("knowledge_graph", {}).get("nodes", {})
    out = []
    
    if canon.get("identifier"):
        out.append({
            "curie": canon["identifier"], "label": canon.get("name", ""),
            "biolink_type": canon.get("category"), "rank": 0, "source": "ARAX"
        })
        
    for nid, node in nodes.items():
        if nid != canon.get("identifier") and len(out) < API_LIMIT:
            out.append({
                "curie": nid, "label": node.get("name", ""),
                "biolink_type": node.get("category"), "rank": len(out), "source": "ARAX"
            })
    return out

async def build_pool_async(term, types, session, semaphore):
    # Unconstrained lexical lookup from both resolvers...
    lookups = [
        async_renci_lookup(session, term, semaphore),
        async_arax_lookup(session, term, semaphore),
    ]
    # ...plus one type-constrained RENCI pass per mapped entity type, so correctly-typed
    # candidates (e.g. UBERON for ANATOMY) the lexical lookup misses still enter the pool.
    # This is a UNION: nothing is discarded, the typed candidates are added.
    biolink_types = sorted({TYPE_TO_BIOLINK[t] for t in types if t in TYPE_TO_BIOLINK})
    for bt in biolink_types:
        lookups.append(async_renci_lookup(session, term, semaphore, biolink_type=bt))

    results = await asyncio.gather(*lookups)
    renci_results, arax_results = results[0], results[1]
    typed_results = [c for sub in results[2:] for c in sub]

    pool = {}
    for cand in renci_results + arax_results + typed_results:
        curie = cand["curie"]
        if curie in pool:
            if cand["source"] not in pool[curie]["source"]:
                pool[curie]["source"] += f"+{cand['source']}"
        else:
            pool[curie] = cand

    # Assign a single GLOBAL rank. Before this, each lookup numbered its own hits 0..9,
    # so the merged pool had duplicate ranks. Order: candidates multiple resolvers agree
    # on first, then by best original lookup-rank, then curie for deterministic ties.
    # The final rank is the candidate's position in this order.
    merged = sorted(
        pool.values(),
        key=lambda c: (0 if "+" in c["source"] else 1, c["rank"], c["curie"]),
    )
    for i, cand in enumerate(merged):
        cand["rank"] = i

    logger.info("Retrieval %r: RENCI=%d, ARAX=%d, typed=%d -> %d unique candidate(s)",
                term, len(renci_results), len(arax_results), len(typed_results), len(merged))
    return merged

async def resolve_all_terms(term_types):
    # term_types: {term_text -> set of NER types it appears with}. The types drive the
    # type-constrained retrieval pass; the pool is still keyed by term text.
    logger.info("Resolving %d unique term(s) against RENCI + ARAX (max %d concurrent)...",
                len(term_types), MAX_CONCURRENT_REQUESTS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    term_to_pool = {}

    async with aiohttp.ClientSession() as session:
        tasks = [build_pool_async(term, types, session, semaphore)
                 for term, types in term_types.items()]
        results = await asyncio.gather(*tasks)

        for term, pool in zip(term_types, results):
            term_to_pool[term] = pool

    total = sum(len(p) for p in term_to_pool.values())
    logger.info("Retrieval finished: %d candidate(s) collected across %d term(s).",
                total, len(term_types))
    return term_to_pool

def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running.
    Plain script: asyncio.run drives a fresh loop. Inside an already-running
    loop (e.g. a FastAPI request) asyncio.run / run_until_complete cannot nest,
    so run it in a separate thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No running event loop; driving asyncio.run directly.")
        return asyncio.run(coro)
    logger.debug("Event loop already running; offloading async retrieval to a worker thread.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

# --- LLM INFERENCE LOGIC (From Original) ---
def load_judge_model():
    # 4-bit NF4 to fit Llama-3.1-8B on an 8 GB GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; the 4-bit judge model requires a GPU.")

    logger.info("Loading judge model %s (4-bit NF4)...", LLM_MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    ).eval()

    logger.info("Judge model ready: %.2f GB VRAM.", model.get_memory_footprint() / 1e9)
    return tokenizer, model

def generate_llm_response(messages, model, tokenizer):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=384,   # headroom for reasoning + JSON choice over a long candidate menu
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    logger.debug("LLM generated %d token(s).", len(new_tokens))

    del inputs, output, new_tokens
    return text

def _parse_choice(reply):
    """Pull the integer choice out of the judge reply. Tries the structured
    parser first, then falls back to the first integer in the text. Returns
    None ONLY on a genuine parse failure (no integer at all) -- this is kept
    distinct from the model deliberately answering 0 (= NIL) so the two are
    not silently conflated when scoring abstention quality."""
    try:
        return choice_parser.parse(reply).choice
    except OutputParserException:
        # Fallback on malformed output: anchor to the "choice" field first (the schema field
        # name, robust even when the JSON is broken); else take the LAST integer, since the
        # answer follows the reasoning -- avoids grabbing a number cited inside the reasoning.
        anchored = re.search(r'"?choice"?\s*[:=]\s*(-?\d+)', reply, re.IGNORECASE)
        if anchored:
            return int(anchored.group(1))
        ints = re.findall(r"-?\d+", reply)
        return int(ints[-1]) if ints else None

def run_judge(entity, pool, model, tokenizer):
    if not pool:
        logger.info("  %r (%s): no candidates -> NIL", entity["text"], entity["type"])
        return None

    ranked = sorted(pool, key=lambda c: c["rank"])   # global rank order: best candidate first
    menu = "\n".join(
        ["0. none of the candidates"]
        + [f"{i}. {c['label']}  [{c['biolink_type'] or '?'}]  ({c['curie']})"
           for i, c in enumerate(ranked, start=1)]
    )
    logger.debug("  %r: %d candidate(s) ->\n%s", entity["text"], len(ranked), menu)

    type_def = TYPE_DEFINITIONS.get(entity["type"])
    type_str = f'{entity["type"]} = {type_def}' if type_def else entity["type"]
    user_prompt = (
        f'Entity: "{entity["text"]}"  (type: {type_str})\n'
        f"Sentence: {entity['segment']}\n\n"
        f"Pick the candidate that is the correct identifier for this entity.\n"
        f"Rules:\n"
        f"- The chosen candidate's type (shown in [brackets]) MUST be consistent with the entity "
        f"type {entity['type']}: e.g. a DISEASE maps to a Disease concept, a GENE to a Gene -- "
        f"never link a disease to a gene or protein.\n"
        f"- Among candidates of the correct type, use the sentence for context to choose the best one.\n"
        f"- Answer 0 only if NO candidate matches both the meaning and the type.\n\n"
        f"{menu}\n\n"
        f"{choice_parser.get_format_instructions()}"
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]

    logger.debug("Judge prompt for %r:\n%s", entity["text"], user_prompt)
    reply = generate_llm_response(messages, model, tokenizer)
    logger.debug("Judge raw reply for %r: %s", entity["text"], reply)

    choice_idx = _parse_choice(reply)
    if choice_idx is None:
        logger.warning("Unparseable judge reply for %r -> NIL: %.120s", entity["text"], reply)
        return None
    if choice_idx <= 0 or choice_idx > len(ranked):   # 0, negative, or too large -> NIL
        logger.info("  %r -> NIL (judge chose %s)", entity["text"], choice_idx)
        return None
    chosen = ranked[choice_idx - 1]
    logger.info("  %r -> %s  (%s)  [choice %d/%d]",
                entity["text"], chosen["curie"], chosen["label"], choice_idx, len(ranked))
    return chosen["curie"]

# --- MAIN ENTRY POINT ---
def run_normalization_pipeline(entities: list) -> list:
    """
    Main Entry Point for the Pipeline
    Takes a list of entities (dicts with 'text', 'type', 'segment').
    """
    if not entities:
        logger.info("No entities to normalize; skipping model load.")
        return []

    # Map each unique term text to the set of NER types it appears with, so retrieval can
    # add a type-constrained pass per type. (Usually a text appears with a single type.)
    term_types = {}
    for e in entities:
        term_types.setdefault(e["text"], set()).add(e["type"])
    unique_terms = list(term_types)
    logger.info("Normalizing %d entity mention(s) across %d unique term(s).",
                len(entities), len(unique_terms))

    # 1. Network I/O Phase (CPU + Network only)
    logger.info("Phase 1/3: retrieving candidates from RENCI + ARAX...")
    term_pools = _run_async(resolve_all_terms(term_types))

    empty = sum(1 for t in unique_terms if not term_pools.get(t))
    if empty:
        logger.warning("%d/%d terms got an EMPTY candidate pool (retrieval miss or API failure).",
                       empty, len(unique_terms))

    # 2. Inference Phase (GPU bounds)
    logger.info("Phase 2/3: loading judge LLM and selecting CURIEs...")
    model, tokenizer = None, None
    results = []

    try:
        tokenizer, model = load_judge_model()

        for i, ent in enumerate(entities, start=1):
            logger.info("[%d/%d] judging %r (%s)", i, len(entities), ent["text"], ent["type"])
            pool = term_pools.get(ent["text"], [])
            curie = run_judge(ent, pool, model, tokenizer)
            results.append({
                **ent,
                "curie": curie
            })
            free_gpu()   # per-entity cache clear: avoid fragmentation OOM on 8GB

    finally:
        # 3. Scale-to-Zero Cleanup
        logger.info("Phase 3/3: unloading judge LLM and clearing VRAM...")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        free_gpu()
        logger.info("LLM unloaded and VRAM cleared.")

    resolved = sum(1 for r in results if r["curie"])
    logger.info("Normalization complete: %d/%d mention(s) linked, %d NIL.",
                resolved, len(results), len(results) - resolved)
    return results


if __name__ == "__main__":
    # Smoke test: run dummy entities end-to-end directly from this file
    import json
    from pathlib import Path
    
    # Try to load HF_TOKEN from project root so we don't have to export it
    project_root = Path(__file__).resolve().parent.parent.parent
    hf_token_path = project_root / "hf_token.txt"
    if hf_token_path.exists():
        os.environ["HF_TOKEN"] = hf_token_path.read_text(encoding="utf-8").strip()
        logger.info("Loaded HF_TOKEN from hf_token.txt")
    
    dummy_entities = [
        {
            "text": "FSHD",
            "type": "DISEASE",
            "segment": "Facioscapulohumeral muscular dystrophy (FSHD) is a genetic muscle disorder."
        },
        {
            "text": "DUX4",
            "type": "GENE",
            "segment": "The aberrant expression of the DUX4 gene is the primary cause of the pathology."
        },
        {
            "text": "D4Z4",
            "type": "CHROMOSOMAL_LOCUS",
            "segment": "Contraction of the D4Z4 macrosatellite repeat array on chromosome 4q35 underlies FSHD type 1."
        },
        {
            "text": "DNA methylation",
            "type": "EPIGENETIC_MODIFICATION",
            "segment": "Loss of DNA methylation at the D4Z4 locus relaxes chromatin and permits aberrant DUX4 expression."
        },
        {
            "text": "skeletal muscle",
            "type": "ANATOMY",
            "segment": "FSHD primarily affects the skeletal muscle of the face, shoulders, and upper arms."
        },
        {
            "text": "myoblast",
            "type": "CELL_TYPE",
            "segment": "DUX4 is cytotoxic when expressed in differentiating myoblast cells."
        },
        {
            "text": "losmapimod",
            "type": "CHEMICAL",
            "segment": "Losmapimod is a small-molecule p38 MAPK inhibitor being evaluated as a treatment for FSHD."
        }
    ]

    logger.info("Starting Normalization Smoke Test...")
    results = run_normalization_pipeline(dummy_entities)
    
    print("\n--- Final Results ---")
    print(json.dumps(results, indent=2))

