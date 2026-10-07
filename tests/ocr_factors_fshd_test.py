"""OCR factor test (4 Oct 2026, no change to BiomedCAT code): the reading stage of the FSHD deck
under three factors, assembled from the pipeline's own bricks (page rasterisation, reading prompt,
profile loading, Ollama call with the pipeline's options):
  - model: gemma4:e4b-it-qat (the classification model) vs qwen2.5vl:3b (the reading model);
  - temperature: 0 and 1 for gemma, 0 only for qwen (T=1 repeated with distinct seeds, since the
    pipeline sends a fixed seed and two T=1 calls with the same seed return the same text);
  - profile: none (bare prompt: no entity scope, no reading focus), "biochemical actions" and
    "genetic determinants" as shipped in data/profiles (their entity scope and reading focus come
    from the derived strata; no custom reading-focus text is added).
Every call records the exact prompt, the description, the timing and a few descriptive counts
(numbered items, words, [UNCLEAR] marks, presence of the elements printed on each slide of
Dataset/FSHD.pdf). The qualitative reading is done afterwards on descriptions.md.

    uv run python tests/ocr_factors_fshd_test.py [Dataset/FSHD.pdf] [--out output/ocr_factors_fshd]
                                                 [--repeats 2] [--pages 1 2] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))

log = logging.getLogger("ocr_factors")

GEMMA = "gemma4:e4b-it-qat"
QWEN = "qwen2.5vl:3b"
MAX_NEW_TOKENS = 8192          # the pipeline's value (biomedcat.stages.ocr)
PROFILES = {"none": None,
            "biochemical": "data/profiles/biochemical_actions.json",
            "genetic": "data/profiles/genetic_determinants.json"}

# Elements printed or drawn on the two slides of Dataset/FSHD.pdf (case-insensitive regexes).
# Automatic presence check only; it does not judge whether the statement about the element is right.
SLIDE_ELEMENTS = {
    1: {"FSHD": r"\bFSHD\b|facioscapulohumeral", "muscular dystrophy": r"muscular dystrophy",
        "no treatment / cure": r"no (preventative )?treatment|no cure|cures?\b", "face": r"\bface\b|\bfacial\b",
        "shoulders": r"shoulder", "upper arms": r"upper[- ]arm", "hypomethylation": r"hypomethylat",
        "D4Z4": r"D4Z4|D4D4", "chromosome 4": r"chromosome 4|\b4q\b", "toxic DUX4 protein": r"toxic",
        "DUX4": r"DUX4", "FSHD1": r"FSHD1", "contraction of the array": r"contract",
        "FSHD2": r"FSHD2", "SMCHD1": r"SMCHD1", "DNMT3B": r"DNMT3B",
        "epigenetic modifier genes": r"epigenetic|modifier gene", "reduced methylation": r"reduced methylation|methylation",
        "exon 1": r"exon ?1", "exon 2": r"exon ?2|\b2\b", "unstable DUX4 mRNA": r"unstable", "stable DUX4 mRNA": r"\bstable",
        "poly(A) tail": r"poly\(?A\)?|\(A\)n|polyadenyl", "reference line": r"Mariot|Dumonceaux|Front\.? Genome"},
    2: {"DUX4": r"DUX4", "FSHD": r"\bFSHD\b|facioscapulohumeral", "DNA binding domains": r"DNA[- ]binding domain",
        "HD1 / HD2": r"\bHD ?1\b|\bHD ?2\b|homeodomain", "N-ter / C-ter": r"N-ter|C-ter|N-terminal|C-terminal",
        "TAD": r"\bTAD\b|transactivation domain", "binding-site motif": r"TAACC|motif|sequence logo|consensus",
        "DUX4 binding site": r"binding site", "DUX4 target gene": r"target gene", "protein aggregation": r"aggregat",
        "impaired NMD": r"\bNMD\b|nonsense[- ]mediated", "ROS / oxidative stress": r"\bROS\b|oxidative stress|reactive oxygen",
        "mitochondrion": r"mitochond", "impaired myogenesis": r"myogen", "cell death": r"cell death|apopto|necro",
        "toxicity": r"toxic", "damage": r"damage", "control vs FSHD histology": r"control", "inflammatory infiltrate": r"inflamm|infiltrat",
        "fluorescence panels": r"fluoresc|\bGFP\b|green|blue|DAPI", "reference line": r"Banerji|Zammit|EMBO|Mariot|wustl"},
}


def conditions(repeats: int) -> list[dict]:
    """Model x temperature x seed x profile, model-major so that each model is loaded once."""
    out = []
    for model, temps in ((GEMMA, (0.0, 1.0)), (QWEN, (0.0,))):
        for t in temps:
            seeds = [0] if t == 0.0 else list(range(repeats))
            for seed in seeds:
                for pid in PROFILES:
                    out.append({"model": model, "temperature": t, "seed": seed, "profile": pid})
    return out


def label(c: dict) -> str:
    m = "gemma" if c["model"] == GEMMA else "qwen"
    s = f"_s{c['seed']}" if c["temperature"] > 0 else ""
    return f"{m}_T{int(c['temperature'])}{s}_{c['profile']}"


def reading_scope(pid: str) -> tuple[list[str], str]:
    """Entity scope and reading focus of a shipped profile, exactly as biomedcat.stages.ocr reads them."""
    if PROFILES[pid] is None:
        return [], ""
    from biomedcat.profiles import load_profile
    p = load_profile(B / PROFILES[pid])
    return list(p.entity_scope), p.reading_focus


def rasterise(path: Path):
    """All pages of the file as PIL images, through the pipeline's own rasteriser."""
    from biomedcat.stages.ocr import _iter_batches, _to_pdf_or_image
    kind, source = _to_pdf_or_image(str(path))
    return [im for batch in _iter_batches(kind, source) for im in batch]


def to_b64(image) -> str:
    buf = io.BytesIO(); image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def counts(text: str, page: int, stem: str) -> dict:
    items = [ln for ln in text.splitlines() if re.match(r"^\s*(\d+[.)]|[-*•])\s+", ln)]
    rec = {"chars": len(text), "words": len(text.split()), "n_items": len(items),
           "n_unclear": len(re.findall(r"unclear", text, re.I)), "n_lines": len([ln for ln in text.splitlines() if ln.strip()])}
    if stem.upper() == "FSHD" and page in SLIDE_ELEMENTS:
        hits = {k: bool(re.search(rx, text, re.I)) for k, rx in SLIDE_ELEMENTS[page].items()}
        rec["elements_present"] = sum(hits.values()); rec["elements_total"] = len(hits); rec["elements"] = hits
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default=str(B / "Dataset/FSHD.pdf"))
    ap.add_argument("--out", default=str(B / "output/ocr_factors_fshd"))
    ap.add_argument("--repeats", type=int, default=2, help="seeds 0..n-1 for the T=1 conditions")
    ap.add_argument("--pages", type=int, nargs="*", help="1-indexed pages to read (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the conditions and prompts, call nothing")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(out / "ocr_factors.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    from biomedcat.config import settings
    from biomedcat.prompts import ocr_description
    from biomedcat.runtime import chat, unload_models

    path = Path(args.file); stem = path.stem
    scopes = {pid: reading_scope(pid) for pid in PROFILES}
    prompts = {pid: ocr_description(*scopes[pid]) for pid in PROFILES}
    conds = conditions(args.repeats)
    log.info("file %s | %d conditions | seed(pipeline)=%s think=%s keep_alive=%s temperature override=%r",
             path, len(conds), settings.ollama_seed, settings.ollama_think, settings.ollama_keep_alive, settings.ollama_temperature)
    for pid in PROFILES:
        log.info("profile %-12s scope=%d kinds | focus=%d chars | prompt=%d chars", pid, len(scopes[pid][0]), len(scopes[pid][1]), len(prompts[pid]))
    if args.dry_run:
        for c in conds:
            print(label(c))
        for pid in PROFILES:
            print(f"\n===== prompt [{pid}] =====\n{prompts[pid]}")
        return 0
    if str(settings.ollama_temperature).strip():
        log.error("OLLAMA_TEMPERATURE=%r overrides the per-call temperature: unset it for this test", settings.ollama_temperature)
        return 2

    images = rasterise(path)
    pages = [p for p in (args.pages or range(1, len(images) + 1)) if 1 <= p <= len(images)]
    b64 = {p: to_b64(images[p - 1]) for p in pages}
    log.info("%d page(s) rasterised, reading pages %s", len(images), pages)

    results_path = out / "results.json"
    records = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    done = {(r["label"], r["page"]) for r in records}
    seed0 = settings.ollama_seed
    loaded = None
    try:
        for c in conds:
            if loaded and loaded != c["model"]:
                unload_models(loaded)
            loaded = c["model"]
            for page in pages:
                lab = label(c)
                if (lab, page) in done:
                    continue
                # The seed is read by chat() from settings; set it per call, the code is untouched.
                settings.ollama_seed = str(c["seed"])
                messages = [{"role": "user", "content": prompts[c["profile"]], "images": [b64[page]]}]
                t0 = time.perf_counter()
                text = chat(c["model"], messages, MAX_NEW_TOKENS, temperature=c["temperature"])
                seconds = time.perf_counter() - t0
                rec = {"label": lab, "file": str(path.relative_to(B)) if path.is_relative_to(B) else str(path), "page": page,
                       **c, "entity_scope": scopes[c["profile"]][0], "reading_focus": scopes[c["profile"]][1],
                       "prompt_chars": len(prompts[c["profile"]]), "max_new_tokens": MAX_NEW_TOKENS, "think": settings.ollama_think,
                       "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(seconds, 1),
                       **counts(text, page, stem), "text": text}
                records.append(rec)
                results_path.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
                log.info("%-28s page %d | %5.1fs | %4d words | %2d items | %d unclear | elements %s/%s%s", lab, page, seconds,
                         rec["words"], rec["n_items"], rec["n_unclear"], rec.get("elements_present", "-"), rec.get("elements_total", "-"),
                         "" if text else " | EMPTY ANSWER")
    finally:
        settings.ollama_seed = seed0
        if loaded:
            unload_models(loaded)

    # Summary table (no text) and a side-by-side file for the qualitative reading.
    cols = ["label", "page", "model", "temperature", "seed", "profile", "seconds", "chars", "words", "n_items", "n_lines", "n_unclear",
            "elements_present", "elements_total"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in sorted(records, key=lambda r: (r["page"], r["model"], r["temperature"], r["seed"], r["profile"])):
            w.writerow(r)
    if any("elements" in r for r in records):
        with (out / "elements.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["page", "element"] + [label(c) for c in conds])
            for page in pages:
                for el in SLIDE_ELEMENTS.get(page, {}):
                    row = [page, el]
                    for c in conds:
                        r = next((r for r in records if r["label"] == label(c) and r["page"] == page), None)
                        row.append("" if r is None else ("x" if r.get("elements", {}).get(el) else ""))
                    w.writerow(row)
    md = [f"# Reading-stage factor test: {path.name}\n", "Prompts per profile are at the end of this file.\n"]
    for page in pages:
        md.append(f"\n## Page {page}\n")
        for r in sorted((r for r in records if r["page"] == page), key=lambda r: (r["model"], r["temperature"], r["seed"], r["profile"])):
            md.append(f"\n### {r['label']}  ({r['seconds']}s, {r['words']} words, {r['n_items']} items, {r['n_unclear']} unclear"
                      + (f", elements {r['elements_present']}/{r['elements_total']}" if "elements_present" in r else "") + ")\n")
            md.append(r["text"] or "_(empty answer)_")
    md.append("\n\n## Prompts\n")
    for pid in PROFILES:
        md.append(f"\n### profile = {pid}\n\n```\n{prompts[pid]}\n```\n")
    (out / "descriptions.md").write_text("\n".join(md), encoding="utf-8")
    log.info("-> %s, %s, %s", results_path, out / "results.csv", out / "descriptions.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
