"""BiomedCAT pipeline: OCR -> NER -> [review gate] -> normalization -> context graph, one file per process.

Each stage loads and frees its own model; this module only sequences them and assembles their
typed outputs into a PipelineResult. The FastAPI backend imports process_file().

Stages and what leaves the machine:
  1. OCR (local vision model) and 2. NER (local LLM + local Biolink index): nothing leaves.
  3. Normalization: entity strings are sent to the RENCI / ARAX resolvers, unless
     RESOLVERS_OFFLINE=1 (local name matching only).
  4. Context graph: local DuckDB over the filtered KG2c.

Human-in-the-loop review (REVIEW_MODE=1, or --review): the run stops after stage 2 and writes
  <stem>_review.csv       one row per extracted entity with a `send` column: yes (may be sent to
                          the resolvers), local (linked offline only), drop (discarded). The
                          default is derived from the privacy filter: personal-entity types and
                          identifier-like strings start as `local`.
  <stem>_stage1.json      slides and entities, so stages 3-4 never re-run OCR or NER.
The user edits the CSV, then resumes with `python -m biomedcat.pipeline --resume output/<stem>_review.csv`.
"""
import csv
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from biomedcat.config import settings, Settings
from biomedcat.stages.ocr import run_ocr
from biomedcat.stages.ner_agent import run_ner_agent
from biomedcat.stages.norm import run_norm
from biomedcat.runtime import unload_models, stage_models
from biomedcat.retrieval import is_shareable
from biomedcat.types import PipelineResult, Slide, Entity
from biomedcat.events import event_emitter


logger = logging.getLogger(__name__)

DATASET_DIR = Path(settings.dataset_path)
OUTPUT_DIR  = Path(settings.output_path)
SUPPORTED   = settings.supported_docs

REVIEW_FIELDS = ["page", "text", "type", "send", "note", "segment"]
SEND_VALUES = ("yes", "local", "drop")


@dataclass
class Stage1Result:
    """OCR + NER output, before anything touches the network."""

    filename: str
    ocr: list[Slide]
    ner: list[Entity]
    ner_model: str = ""


# ------------------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------------------

def run_stage1(path: str, settings: Settings = settings, model_id: str | None = None) -> Stage1Result:
    """OCR then NER; fully local."""
    ner_model = model_id or settings.classification_model_id
    logger.info("=== processing %s ===", path)

    event_emitter.on_stage_change(path, "OCR", "running")
    t0 = time.perf_counter()
    slides = run_ocr(path, settings.ocr_model_id)
    unload_models(settings.ocr_model_id)   # the reading model gives its memory back before typing starts
    event_emitter.on_stage_change(path, "OCR", "done")
    logger.info("OCR done: %d slide(s) in %.1fs", len(slides), time.perf_counter() - t0)

    event_emitter.on_stage_change(path, "NER", "running")
    t0 = time.perf_counter()
    entities = run_ner_agent([slide.text for slide in slides], ner_model)
    unload_models(*stage_models())         # typing, screening and existence models released
    event_emitter.on_stage_change(path, "NER", "done")
    logger.info("NER done: %d entit(y/ies) in %.1f s", len(entities), time.perf_counter() - t0)

    return Stage1Result(filename=Path(path).name, ocr=slides, ner=entities, ner_model=ner_model)


def run_stage2(stage1: Stage1Result, path: str, settings: Settings = settings,
               local_only_terms: set[str] | None = None) -> PipelineResult:
    """Normalization (network unless offline) then context graph (local).

    `local_only_terms` are entity texts the reviewer marked `local`: they are linked by exact
    name against the local KG2c table and never sent to the resolvers.
    """
    from biomedcat import retrieval

    event_emitter.on_stage_change(path, "Norm", "running")
    t0 = time.perf_counter()
    if local_only_terms:
        # Narrow the shareability filter for this run without changing the global settings.
        original = retrieval.is_shareable

        def gated(term: str, types: set[str]) -> bool:
            return term not in local_only_terms and original(term, types)

        retrieval.is_shareable = gated
    try:
        results = run_norm(stage1.ner, model_id=stage1.ner_model or settings.classification_model_id)
    finally:
        if local_only_terms:
            retrieval.is_shareable = original
    unload_models(*stage_models())         # the judge model is released before the graph stage
    event_emitter.on_stage_change(path, "Norm", "done")
    linked = sum(1 for r in results if r.curie)
    logger.info("Norm done: %d/%d linked in %.1fs", linked, len(results), time.perf_counter() - t0)

    context_graph = None
    if Path(settings.kg2c_dir, "edges.parquet").is_file():
        from biomedcat.stages.context_graph import run_context_graph, seeds_from_normalized, ContextGraphParams

        seeds = seeds_from_normalized(results)
        if seeds:
            event_emitter.on_stage_change(path, "Graph", "running")
            t0 = time.perf_counter()
            out_dir = OUTPUT_DIR / f"{Path(path).stem}_context_graph"
            try:
                cg = run_context_graph(seeds, out_dir, ContextGraphParams(profile=settings.predicate_profile))
                context_graph = asdict(cg)
                logger.info("Context graph done: %d nodes, %d edges in %.1fs", cg.n_nodes, cg.n_edges, time.perf_counter() - t0)
            except Exception as e:
                logger.exception("Context graph failed: %s", e)
            event_emitter.on_stage_change(path, "Graph", "done")
        else:
            logger.warning("Context graph skipped: no entity linked to KG2c")
    else:
        logger.warning("Context graph skipped: no filtered KG2c at %s", settings.kg2c_dir)

    return PipelineResult(filename=stage1.filename, ocr=stage1.ocr, ner=stage1.ner, norm=results, context_graph=context_graph)


def process_file(path: str, settings: Settings = settings, model_id: str | None = None) -> PipelineResult:
    """Full run without a review gate (the API path and the default batch runner)."""
    return run_stage2(run_stage1(path, settings, model_id), path, settings)


# ------------------------------------------------------------------------------------------
# Review gate
# ------------------------------------------------------------------------------------------

def write_review(stage1: Stage1Result, review_path: Path, stage1_path: Path) -> None:
    """Write the review CSV and the stage-1 JSON; nothing has left the machine at this point."""
    stage1_path.write_text(json.dumps({
        "filename": stage1.filename, "ner_model": stage1.ner_model,
        "slides": [asdict(s) for s in stage1.ocr], "entities": [asdict(e) for e in stage1.ner],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    seen: set[tuple[str, str]] = set()
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for e in stage1.ner:
            key = (e.text, e.type)
            if key in seen:
                continue
            seen.add(key)
            default = "yes" if is_shareable(e.text, {e.type}) else "local"
            w.writerow({"page": e.page, "text": e.text, "type": e.type, "send": default, "note": "",
                        "segment": (e.segment or "")[:200]})
    logger.info("Review file written: %s (edit the `send` column: yes / local / drop, then --resume)", review_path)


def apply_review(review_path: Path) -> tuple[Stage1Result, set[str], str]:
    """Load the stage-1 JSON next to the review CSV and apply the reviewer's decisions."""
    stem = review_path.name[: -len("_review.csv")]
    stage1_path = review_path.parent / f"{stem}_stage1.json"
    payload = json.loads(stage1_path.read_text(encoding="utf-8"))
    stage1 = Stage1Result(
        filename=payload["filename"],
        ocr=[Slide(**s) for s in payload["slides"]],
        ner=[Entity(**e) for e in payload["entities"]],
        ner_model=payload.get("ner_model", ""),
    )
    decisions: dict[tuple[str, str], str] = {}
    with open(review_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            send = (row.get("send") or "yes").strip().lower()
            if send not in SEND_VALUES:
                raise ValueError(f"invalid send value {send!r} for {row.get('text')!r}; use yes, local or drop")
            decisions[(row["text"], row["type"])] = send
    kept = [e for e in stage1.ner if decisions.get((e.text, e.type), "yes") != "drop"]
    local_only = {e.text for e in kept if decisions.get((e.text, e.type)) == "local"}
    dropped = len(stage1.ner) - len(kept)
    logger.info("Review applied: %d entities kept, %d dropped, %d linked locally only", len(kept), dropped, len(local_only))
    stage1.ner = kept
    return stage1, local_only, stem


# ------------------------------------------------------------------------------------------
# Logging, output
# ------------------------------------------------------------------------------------------

class _ConsoleFilter(logging.Filter):
    """Keep the console readable: stage-level progress and real problems only."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or record.name == logger.name


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(_ConsoleFilter())
    root.addHandler(console)


def _start_file_log(log_path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _stop_file_log(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _write_json(result: PipelineResult, out_path: Path, elapsed: float, model_id: str) -> Path:
    """Serialize one run. `entities` are the normalized records (they carry every NER field)."""
    payload = {
        "schema_version": "1.1",
        "run": {
            "file": result.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": {
                "ocr":  settings.ocr_model_id,
                "ner":  model_id,
                "norm": model_id,
                "embeddings": settings.RAG_embedding_model,
            },
            "resolvers": {
                "renci":     settings.renci_url,
                "arax":      settings.arax_url,
                "api_limit": settings.api_limit,
                "offline":   settings.resolvers_offline,
                "kg2c_dir":  str(settings.kg2c_dir),
            },
            "profile": str(settings.predicate_profile),
            "elapsed_s": round(elapsed, 1),
        },
        "slides":   [asdict(slide) for slide in result.ocr],
        "entities": [asdict(e) for e in result.norm],
        "context_graph": result.context_graph,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def _print_results(result: PipelineResult) -> None:
    # The console summary must never abort a run: on a cp1252 console a Greek letter in the
    # reading (kappa, 4 Oct 2026, VoieA) raised UnicodeEncodeError before the JSON was written.
    # The JSON is now written first (see the call sites) and the console falls back to
    # replacement characters.
    try:
        _print_results_unsafe(result)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
            _print_results_unsafe(result)


def _print_results_unsafe(result: PipelineResult) -> None:
    print("\n" + "=" * 72)
    print(f"RESULTS for {result.filename}")
    print("=" * 72)
    print(f"\n--- OCR ({len(result.ocr)} slide(s)) ---")
    for slide in result.ocr:
        print(f"\n[slide {slide.page}]")
        print(slide.text)
    print(f"\n--- NER ({len(result.ner)} entit(y/ies)) ---")
    for e in result.ner:
        print(f"  {e.type:<24} {e.text}")
    linked = sum(1 for r in result.norm if r.curie)
    print(f"\n--- Normalization ({linked}/{len(result.norm)} linked) ---")
    for r in result.norm:
        print(f"  {r.type:<24} {r.text[:34]:<34} -> {r.curie or 'NIL'}  {('KG2c:' + r.kg2c_id) if r.kg2c_id else ''}")
    if result.context_graph:
        cg = result.context_graph
        print(f"\n--- Context graph ({cg['n_nodes']} nodes, {cg['n_edges']} edges, {cg['n_seeds_in_graph']}/{cg['n_seeds']} seeds) -> {cg['out_dir']}")
    print()


# ------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiomedCAT: slides -> Biolink entities -> KG2c context graph.")
    parser.add_argument("path", nargs="?", help="One file to process; default: every supported file in the dataset directory.")
    parser.add_argument("--review", action="store_true", help="Stop after OCR+NER and write <stem>_review.csv for human review (REVIEW_MODE=1 does the same).")
    parser.add_argument("--resume", metavar="REVIEW_CSV", help="Resume from a reviewed <stem>_review.csv: normalization and context graph only.")
    return parser.parse_args()


if __name__ == "__main__":
    _setup_logging()
    args = _parse_args()
    DEFAULT_MODEL = settings.classification_model_id
    review_mode = args.review or settings.review_mode
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.resume:
        review_path = Path(args.resume)
        file_log = _start_file_log(OUTPUT_DIR / f"{review_path.name[:-len('_review.csv')]}_BiomedCAT.log")
        try:
            t0 = time.perf_counter()
            stage1, local_only, stem = apply_review(review_path)
            result = run_stage2(stage1, stage1.filename, settings, local_only_terms=local_only)
            out_path = OUTPUT_DIR / f"{stem}_BiomedCAT.json"
            _write_json(result, out_path=out_path, elapsed=time.perf_counter() - t0, model_id=stage1.ner_model or DEFAULT_MODEL)
            _print_results(result)
            print(f"\nWrote {out_path}")
        finally:
            _stop_file_log(file_log)
        sys.exit(0)

    if args.path:
        paths = [Path(args.path)]
    else:
        if not DATASET_DIR.is_dir():
            sys.exit(f"No dataset directory at {DATASET_DIR}\nCreate it and add PDF or image files, or run on a single file:\n"
                     f"  python -m biomedcat.pipeline path/to/slides.pdf")
        paths = [p for p in sorted(DATASET_DIR.iterdir()) if p.suffix.lower() in SUPPORTED]
        if not paths:
            sys.exit(f"No supported files in {DATASET_DIR}\nSupported formats: {', '.join(sorted(SUPPORTED))}")

    processed, skipped, failures = 0, 0, []
    for path in paths:
        out_path = OUTPUT_DIR / f"{path.stem}_BiomedCAT.json"
        review_path = OUTPUT_DIR / f"{path.stem}_review.csv"
        path_str = str(path)
        if out_path.exists() or (review_mode and review_path.exists()):
            logger.info("skip %s (%s already exists)", path.name, (review_path if review_mode else out_path).name)
            skipped += 1
            continue

        event_emitter.on_start(path_str)
        file_log = _start_file_log(OUTPUT_DIR / f"{path.stem}_BiomedCAT.log")
        try:
            t0 = time.perf_counter()
            if review_mode:
                stage1 = run_stage1(path_str, settings, model_id=DEFAULT_MODEL)
                write_review(stage1, review_path, OUTPUT_DIR / f"{path.stem}_stage1.json")
                print(f"\nStopped for review: {len(stage1.ner)} entities in {review_path}\n"
                      f"Edit the `send` column (yes / local / drop), then run:\n"
                      f"  python -m biomedcat.pipeline --resume \"{review_path}\"")
                event_emitter.on_stage_change(path_str, "Review", "waiting")
            else:
                result = process_file(path_str, settings, model_id=DEFAULT_MODEL)
                written = _write_json(result, out_path=out_path, elapsed=time.perf_counter() - t0, model_id=DEFAULT_MODEL)
                _print_results(result)
                print(f"\nWrote {written}")
                event_emitter.on_success(path_str, str(out_path))
            processed += 1
        except Exception as e:
            event_emitter.on_error(path_str, str(e))
            logger.exception("FAILED %s", path.name)
            failures.append(path.name)
        finally:
            _stop_file_log(file_log)

    print(f"\nBatch complete: {processed} processed, {skipped} skipped, {len(failures)} failed")
    for name in failures:
        print(f"  FAILED: {name}")
    sys.exit(1 if failures else 0)
