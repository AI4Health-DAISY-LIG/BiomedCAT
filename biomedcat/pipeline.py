"""BiomedCAT pipeline: run OCR -> NER -> Normalization on one file, in one process.

The Facade over the three stages. Each stage loads and frees its own model, so this module
only sequences them and assembles their typed outputs into a PipelineResult. The FastAPI
backend will import process_file() and wrap it in HTTP.
"""
import sys
import time
import logging
from pathlib import Path

from biomedcat.stages.ocr import run_ocr
from biomedcat.stages.ner import run_ner
from biomedcat.stages.norm import run_norm
from biomedcat.types import PipelineResult

logger = logging.getLogger(__name__)


def process_file(path: str) -> PipelineResult:
    """Run the full OCR -> NER -> Normalization pipeline on a single file."""
    logger.info("=== processing %s ===", path)

    # Stage 1: OCR -> per-slide text.
    t0 = time.perf_counter()
    slides = run_ocr(path)
    logger.info("OCR done: %d slide(s) in %.1fs", len(slides), time.perf_counter() - t0)

    # Stage 2: NER -> typed entities. Pool every slide's text into one file-level extraction.
    texts = []
    for slide in slides:
        texts.append(slide.text)
    t0 = time.perf_counter()
    entities = run_ner(texts)
    logger.info("NER done: %d entit(y/ies) in %.1fs", len(entities), time.perf_counter() - t0)

    # Stage 3: Normalization -> entities linked to CURIEs.
    t0 = time.perf_counter()
    results = run_norm(entities)
    linked = 0
    for r in results:
        if r.curie:
            linked += 1
    logger.info("Norm done: %d/%d linked in %.1fs", linked, len(results), time.perf_counter() - t0)

    return PipelineResult(
        filename=Path(path).name,
        ocr=slides,
        ner=entities,
        norm=results,
    )


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
    print(f"\n--- Normalization ({linked}/{len(result.norm)} linked) ---")
    for r in result.norm:
        curie = r.curie if r.curie else "NIL"
        print(f"  {r.type:<24} {r.text[:34]:<34} -> {curie}")
    print()


if __name__ == "__main__":
    # End-to-end runner: process the bundled FSHD example (or a path argument) and print the
    # results. Display only -- nothing is written to disk. Pass a file path to run on another file.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    default = Path(__file__).resolve().parents[1] / "Dataset" / "FSHD1.pptx"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default)

    result = process_file(path)
    _print_results(result)
