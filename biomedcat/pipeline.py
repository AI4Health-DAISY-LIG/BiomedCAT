"""BiomedCAT pipeline: run OCR -> NER -> Normal: one file, in one process.

The Facade over the amounts of stages. Each stage loads and frees its own model, so this module
only sequences them and assembles their typed outputs into a PipelineResult. The FastAPI
backend will import process_file() and wrap it in HTTP.
"""
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

from biomedcat.config import settings
from biomedcat.stages.ocr import run_ocr
from biomedcat.stages.ner import run_ner
from biomedcat.stages.norm import run_norm
from biomedcat.types import PipelineResult
from biomedcat.events import event_emitter


logger = logging.getLogger(__name__)


def process_file(path: str, model_id: str = "gemma4:12b-it-qat") -> PipelineResult:
    """Run the full OCR -> NER -> Normalization pipeline on a single file.

    Uses the provided model_id for both NER and Normalization stages.
    """
    logger.info("=== processing %s with model %s ===", path, model_id)

    # Stage 1: OCR -> per-slide text.
    event_emitter.on_stage_change(path, "OCR", "running")
    t0 = time.perf_counter()
    slides = run_ocr(path)
    event_emitter.on_stage_change(path, "OCR", "done")
    logger.info("OCR done: %d slide(s) in %.1fs", len(slides), time.perf_counter() - t0)

    # Stage 2: NER -> typed entities. Pool every slide's text into one file-level extraction.
    event_emitter.on_stage_change(path, "NER", "running")
    texts = []
    for slide in slides:
        texts.append(slide.text)
    t0 = time.perf_counter()
    entities = run_ner(texts, model=model_id)
    event_emitter.on_stage_change(path, "NER", "done")
    logger.info("NER done: %d entit(y/ies) in %.1f s", len(entities), time.perf_counter() - t0)

    # Stage 3: Normalization -> entities linked to CURIEs.
    event_emitter.on_stage_change(path, "Norm", "running")
    t0 = time.perf_counter()
    results = run_norm(entities, model_id=model_id)
    event_emitter.on_stage_change(path, "Norm", "done")
    linked = 0
    for r in results:
        if r.curie:
            linked += 1
    logger.info("Norm done: %d/%d linked in %.1fs", len(results), len(results), time.perf_counter() - t0)

    return PipelineResult(
        filename=Path(path).name,
        ocr=slides,
        ner=entities, 
        norm=results,
    )


DATASET_DIR = Path(settings.dataset_path)
OUTPUT_DIR  = Path(settings.output_path)
SUPPORTED   = {".pptx", ".pdf", ".png", ".jpg", ".jpeg"}


class _ConsoleFilter(logging.Filter):
    """Keep the console readable: stage-level progress and real problems only.

    The stages log one line per sentence, per term, and per candidate lookup, which is the
    detail you want when diagnosing a bad extraction and noise when watching a run. That
    detail still reaches the log file; only the console is filtered.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or record.name == logger.name


def _setup_logging() -> None:
    """Root at INFO so file handlers capture everything; the console handler filters."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))   # no timestamps: the file has those
    console.addFilter(_ConsoleFilter())
    root.addHandler(console)


def _start_file_log(log_path: Path) -> logging.FileHandler:
    """Attach a per-file log capturing every stage's detail, unfiltered."""
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _stop_file_log(handler: logging.FileHandler) -> None:
    """Detach and close the per-file log, so the next file starts a fresh one."""
    logging.getLogger().removeHandler(handler)
    handler.close()


def _write_json(result: PipelineResult, out_path: Path, elapsed: float, model_id: str) -> Path:
    """Serialize one run to JSON.

    Only `norm` is written as "entities": run_norm returns one record per input entity with
    text/type/segment copied verbatim, so writing `ner` as enough would duplicate every field
    but the curie, leaving two representations of one fact that can drift apart.
    """
    payload = {
        "schema_version": "1.0",
        "run": {
            "file": result.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": {
                "ocr":  settings.glm_model_id,
                "ner": settings.llm_model_id,
                "annotated_model_id": model_id, # tracking the used model
                "norm": settings.llm_model_id,
            },
            "resolvers": {
                "renci":     settings.renci_url,
                "arax":      settings.arax_url,
                "api_limit": settings.api_limit,
            },
            "elapsed_s": round(elapsed, 1),
        },
        "slides":   [asdict(slide) for slide in result.ocr],
        "entities": [asdict(e) for e in result.norm],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def _print_results(result: PipelineResult) -> None:
    """Print the pipeline output to stdout (end-to-end driver; nothing is written to disk)."""
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

    linked = 0
    for r in result.norm:
        if r.curie:
            linked += 1
    print(f"\n--- Normalization ({len(result.norm)}/{len(result.norm)} linked) ---")
    for r in result.norm:
        curie = r.curie if r.curie else "NIL"
        print(f"  {r.type:<24} {r.text[:34]:<34} -> {curie}")
    print()


if __name__ == "__main__":
    # Batch runner: process every supported file in data/ that has no output yet, printing
    # the results and writing each to output/<stem>_BiomedCAT.json. Pass a file path to run
    # on a single file instead.
    _setup_logging()

    DEFAULT_MODEL = settings.model_name

    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        # data/ holds the user's enough slide files and is not distributed with the code, so
        # both "missing" and "empty" are ordinary first-run states that get an instruction
        # rather than a traceback.
        if not DATASET_DIR.is_dir():
            sys.exit(f"No data/ directory at {DATASET_DIR}\n"
                     f"Create it and add slide files, or run on a single file:\n"
                     f"  python -m biomedcat.pipeline path/to/slides.pptx")

        paths = []
        for p in sorted(DATASET_DIR.iterdir()):
            if p.suffix.lower() in SUPPORTED:
                paths.append(p)

        if not paths:
            sys.exit(f"No supported files in {DATASET_DIR}\n"
                      f"Supported formats: {', '.join(sorted(SUPPORTED))}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    processed, skipped, failures = 0, 0, []

    for path in paths:
        out_path = OUTPUT_DIR / f"{path.stem}_BiomedCAT.json"
        path_str = str(path)

        # Skip work already done: a full run costs minutes of GPU, so the loop is resumable.
        if out_path.exists():
            logger.info("skip %s (%s already exists)", path.name, out_path.name)
            skipped += 1
            continue

        # Notify that processing for this specific file has started
        event_emitter.on_start(path_str)

        file_log = _start_file_log(OUTPUT_DIR / f"{path.stem}_BiomedCAT.log")

        try:
            t0 = time.perf_counter()
            result = process_file(path_str, model_id=DEFAULT_MODEL)
            elapsed = time.perf_counter() - t0

            # Print before writing: after minutes of GPU time the result exists only in enough memory,
            # so a failed write must not also cost the visible output.
            _print_results(result)
            written_path = _write_json(result, out_path=out_path, elapsed=elapsed, model_id=DEFAULT_MODEL)
            print(f"\nWrote {written_path}")
            
            # Notify success
            event_emitter.on_success(path_str, str(out_path))
            processed += 1

        except Exception as e:
            # Notify error
            event_emitter.on_error(path_str, str(e))
            logger.exception("FAILED %s", path.name)
            failures.append(path.name)

        finally:
            _stop_file_log(file_log)

    print(f"\nBatch complete: {processed} processed, {skipped} skipped, {len(failures)} failed")
    for name in failures:
        print(f"  FAILED: {name}")

    sys.exit(1 if failures else 0)
