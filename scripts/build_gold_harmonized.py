#!/usr/bin/env python
"""Merge the curated gold spreadsheets of every slide deck into one harmonized gold file.

Sources (data/gold/), one deck each, all curated by the expert:
  FSHD.pdf               fshd_slides_gold_v2.json (32 explicit positives, 34 adversarial) and
                         fshd_implicit_expected.json (7 implicit entities, 1 removed term)
  VoieA.pdf              VoieA_benchmark_to_curate.xlsx        sheets VoieA_positives, VoieA_adversarial
  VoieB.pdf              NER_test_Suite_VoieB_candidates.xlsx  one sheet: positives then adversarial block
  VoieC.pdf              VoieC_benchmark_draft.xlsx            sheets positives, adversarial
  VoiesA-C.pdf           NER_test_Suite_VoiesAC_candidates.xlsx one sheet: positives, NA judgements, adversarial
  FSHD1multiscales.pdf   FSHD1multiscales_benchmark_to_curate.xlsx sheets positives (VocabularyType), adversarial
  chemicals.png          glycolysis_benchmark_to_curate.xlsx    sheets positives (Implicit basis), adversarial

Rules
  - Values are copied verbatim from the source cells (read with openpyxl, no type coercion); the
    only new columns are flagged `derived_*` or are structural (gold_id, deck, source, row_kind).
  - row_kind: `positive` (a term on the slide, with a Biolink class), `implicit` (expected by the
    expert but not written on the slide), `adversarial` (must receive neither class nor CURIE),
    `na_on_slide` (written on the slide, judged not to be an entity: VoiesA-C rows with no class
    and a note other than 'adversarial'), `removed` (dropped by the expert).
  - vocabulary_type: `explicit` / `implicit`, from the VocabularyType / Implicit basis column
    when the file has one, otherwise assumed explicit (the drafts list terms read on the slide,
    with their slide zone); `vocabulary_type_origin` says which.
  - included: False when the curation status is delete / drop, True otherwise (an empty status
    means the proposal was kept, as the curation sheets' READMEs state).
  - NameRes / NodeNorm candidate columns are not carried over.
  - derived_curie_label_kg2c: name of the gold CURIE in the local KG2c tables (data/kg2c), used
    only to match English readings of French terms. Identifiers are not scored: the gold serves
    the reading and typing evaluation.

Outputs: data/gold/gold_harmonized.xlsx (sheets terms, provenance, README), gold_harmonized.json,
and gold_harmonized_verification.md (every verbatim cell re-read from the sources and compared).

Usage:  uv run python scripts/build_gold_harmonized.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT_XLSX = GOLD / "gold_harmonized.xlsx"
OUT_JSON = GOLD / "gold_harmonized.json"
OUT_VERIF = GOLD / "gold_harmonized_verification.md"

COLUMNS = ["gold_id", "deck", "document_file", "source_file", "source_sheet", "source_id", "row_kind", "included",
           "vocabulary_type", "vocabulary_type_origin", "term", "english_query", "compound", "biolink_class", "curie",
           "linkable", "expertise_level", "slide_zone", "context", "curation_status", "curation_note",
           "derived_compound_flag", "derived_curie_label_kg2c"]
EXCLUDED_STATUS = {"delete", "drop"}
TRACE: list[tuple[str, str, int, str, str]] = []  # (source_file, sheet, excel_row, source_column, harmonized_column) per copied cell
PROVENANCE: list[dict] = []


def cell(v):
    """Verbatim cell value; None for empty cells."""
    if v is None:
        return None
    if isinstance(v, str):
        return v if v.strip() != "" else None
    return v


def read_sheet(path: Path, sheet: str) -> tuple[list[str], list[tuple[int, dict]]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out = []
    for i, r in enumerate(rows[1:], start=2):
        d = {h: cell(v) for h, v in zip(header, r) if h}
        if d.get(header[0]) is None:
            continue
        out.append((i, d))
    return header, out


def status_included(status) -> bool:
    return str(status or "").strip().lower() not in EXCLUDED_STATUS


def make_row(deck: str, document: str, src: Path, sheet: str, excel_row: int | None, d: dict, mapping: dict[str, str],
             row_kind: str, vocab_type: str, vocab_origin: str) -> dict:
    """mapping: harmonized column -> source column. Values are copied as they are."""
    row = {c: None for c in COLUMNS}
    row.update({"deck": deck, "document_file": document, "source_file": src.name, "source_sheet": sheet, "row_kind": row_kind,
                "vocabulary_type": vocab_type, "vocabulary_type_origin": vocab_origin})
    for hcol, scol in mapping.items():
        row[hcol] = d.get(scol)
        if excel_row is not None:
            TRACE.append((src.name, sheet, excel_row, scol, hcol))
    row["gold_id"] = f"{deck}:{sheet}:{row['source_id']}"  # ids restart per sheet (FSHD v2 positives and adversarial both use 33-41)
    row["included"] = status_included(row["curation_status"])
    c = row["compound"]
    row["derived_compound_flag"] = None if c is None else bool(int(float(c))) if not isinstance(c, bool) else c
    return row


# ----------------------------------------------------------------------------------------------
def load_fshd() -> list[dict]:
    rows = []
    src = GOLD / "fshd_slides_gold_v2.json"
    g = json.loads(src.read_text(encoding="utf-8"))
    m = {"source_id": "id", "term": "term", "compound": "compound", "biolink_class": "biolink_class", "curie": "curie", "linkable": "linkable"}
    for p in g["positives"]:
        rows.append(make_row("FSHD", "FSHD.pdf", src, "positives", None, p, m, "positive", "explicit", "gold v2: terms written on the slides"))
    for a in g["adversarial"]:
        rows.append(make_row("FSHD", "FSHD.pdf", src, "adversarial", None, a, {"source_id": "id", "term": "term", "compound": "compound", "curation_note": "note"}, "adversarial", "", ""))
    PROVENANCE.append({"deck": "FSHD", "source_file": src.name, "sheet": "positives / adversarial", "rows_read": len(g["positives"]) + len(g["adversarial"]), "source_note": g["source"]})
    src2 = GOLD / "fshd_implicit_expected.json"
    g2 = json.loads(src2.read_text(encoding="utf-8"))
    for e in g2["entities"]:
        rows.append(make_row("FSHD", "FSHD.pdf", src2, "entities", None, e, m, "implicit", "implicit", "fshd_implicit_expected.json: expected by the expert, not on the slides"))
    for e in g2["removed_by_expert"]:
        r = make_row("FSHD", "FSHD.pdf", src2, "removed_by_expert", None, e, {**m, "curation_note": "reason"}, "removed", "", "")
        r["included"] = False
        rows.append(r)
    PROVENANCE.append({"deck": "FSHD", "source_file": src2.name, "sheet": "entities / removed_by_expert", "rows_read": len(g2["entities"]) + len(g2["removed_by_expert"]), "source_note": g2["source"]})
    return rows


def load_two_sheet(deck: str, document: str, src: Path, pos_sheet: str, adv_sheet: str, pos_map: dict, adv_map: dict,
                   vocab_col: str | None = None) -> list[dict]:
    rows = []
    for sheet, mapping, kind in ((pos_sheet, pos_map, "positive"), (adv_sheet, adv_map, "adversarial")):
        header, data = read_sheet(src, sheet)
        for excel_row, d in data:
            if kind == "positive":
                if vocab_col:
                    vt = str(d.get(vocab_col) or "").strip().lower()
                    origin = f"column '{vocab_col}'"
                else:
                    vt, origin = "explicit", "assumed: term read on the slide (no explicit/implicit column in the source)"
            else:
                vt, origin = "", ""
            rows.append(make_row(deck, document, src, sheet, excel_row, d, mapping, kind, vt, origin))
        PROVENANCE.append({"deck": deck, "source_file": src.name, "sheet": sheet, "rows_read": len(data), "columns": "; ".join(h for h in header if h)})
    return rows


def load_single_sheet(deck: str, document: str, src: Path, sheet: str, mapping: dict, class_col: str, note_col: str) -> list[dict]:
    """One sheet holding positives (a Biolink class), NA judgements and an adversarial block."""
    rows = []
    header, data = read_sheet(src, sheet)
    kinds = {"positive": 0, "adversarial": 0, "na_on_slide": 0}
    for excel_row, d in data:
        note = str(d.get(note_col) or "").strip().lower()
        if d.get(class_col) is not None:
            kind, vt, origin = "positive", "explicit", "assumed: term read on the slide (no explicit/implicit column in the source)"
        elif note.startswith("adversarial") or deck == "VoieB":
            kind, vt, origin = "adversarial", "", ""
        else:
            kind, vt, origin = "na_on_slide", "", ""
        kinds[kind] += 1
        rows.append(make_row(deck, document, src, sheet, excel_row, d, mapping, kind, vt, origin))
    PROVENANCE.append({"deck": deck, "source_file": src.name, "sheet": sheet, "rows_read": len(data), "columns": "; ".join(h for h in header if h),
                       "row_kind_rule": f"positive = '{class_col}' filled; adversarial = note starting with 'adversarial'" + (" or any row without a class (VoieB: the adversarial block has no class)" if deck == "VoieB" else "; other rows without a class = na_on_slide"), **{f"n_{k}": v for k, v in kinds.items()}})
    return rows


def kg2c_lookup(curies: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """CURIE -> (canonical id, name) from the local KG2c tables."""
    try:
        import duckdb
    except ImportError:
        return {}
    kg = ROOT / "data" / "kg2c"
    if not (kg / "equivalents.parquet").is_file():
        return {}
    con = duckdb.connect()
    con.execute("CREATE TABLE q(curie VARCHAR)")
    con.executemany("INSERT INTO q VALUES (?)", [(c,) for c in curies])
    res = con.execute(f"""
        SELECT q.curie, COALESCE(e.canonical_id, q.curie) AS canon, n.name
        FROM q LEFT JOIN read_parquet('{(kg / 'equivalents.parquet').as_posix()}') e ON e.curie = q.curie
               LEFT JOIN read_parquet('{(kg / 'nodes.parquet').as_posix()}') n ON n.id = COALESCE(e.canonical_id, q.curie)
    """).fetchall()
    return {c: (canon if name is not None else None, name) for c, canon, name in res}


def md_table(counts: pd.Series) -> str:
    """Markdown table of a grouped count series (no tabulate dependency)."""
    names = list(counts.index.names)
    out = ["| " + " | ".join(names + ["n"]) + " |", "|" + "---|" * (len(names) + 1)]
    for idx, n in counts.items():
        idx = idx if isinstance(idx, tuple) else (idx,)
        out.append("| " + " | ".join(str(i) for i in idx) + f" | {n} |")
    return "\n".join(out)


def main() -> int:
    rows: list[dict] = []
    rows += load_fshd()
    rows += load_two_sheet("VoieA", "VoieA.pdf", GOLD / "VoieA_benchmark_to_curate.xlsx", "VoieA_positives", "VoieA_adversarial",
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "slide_zone": "slide zone", "english_query": "English query", "curation_note": "notes", "curation_status": "curation decision"},
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "curation_note": "rationale", "curation_status": "curation decision"})
    rows += load_single_sheet("VoieB", "VoieB.pdf", GOLD / "NER_test_Suite_VoieB_candidates.xlsx", "Sheet1",
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "context": "mention on slide (FR)", "slide_zone": "slide zone", "curation_note": "comment", "curation_status": "curation status"},
        class_col="Biolink class", note_col="comment")
    rows += load_two_sheet("VoieC", "VoieC.pdf", GOLD / "VoieC_benchmark_draft.xlsx", "positives", "adversarial",
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "english_query": "english query", "slide_zone": "slide zone", "context": "context (slide text)", "curation_note": "notes", "curation_status": "status"},
        {"source_id": "ID", "term": "input", "compound": "compound term", "curation_note": "why adversarial", "curation_status": "status"})
    rows += load_single_sheet("VoiesA-C", "VoiesA-C.pdf", GOLD / "NER_test_Suite_VoiesAC_candidates.xlsx", "Sheet1",
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "slide_zone": "zone slide", "context": "contexte (slide)", "curation_note": "note", "curation_status": "statut curation"},
        class_col="Biolink class", note_col="note")
    rows += load_two_sheet("FSHD1multiscales", "FSHD1multiscales.pdf", GOLD / "FSHD1multiscales_benchmark_to_curate.xlsx", "positives", "adversarial",
        {"source_id": "ID", "expertise_level": "ExpertiseLevel (1: Beginner, 2: Intermediate, 3: Expert)", "term": "Vocabulary", "curie": "known CURIE",
         "curation_note": "Remark, justification", "biolink_class": "Biolink class (proposed)", "compound": "compound term", "english_query": "English query",
         "slide_zone": "slide zone", "context": "context (slide text)", "curation_status": "status"},
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "curation_note": "rationale", "curation_status": "curation decision"}, vocab_col="VocabularyType")
    rows += load_two_sheet("glycolysis", "chemicals.png", GOLD / "glycolysis_benchmark_to_curate.xlsx", "positives", "adversarial",
        {"source_id": "ID", "expertise_level": "ExpertiseLevel (1: Beginner, 2: Intermediate, 3: Expert)", "term": "Vocabulary", "curie": "known CURIE",
         "curation_note": "Remark, justification", "biolink_class": "Biolink class (proposed)", "compound": "compound term", "english_query": "English query",
         "slide_zone": "slide zone", "context": "what the slide shows", "curation_status": "status"},
        {"source_id": "ID", "biolink_class": "Biolink class", "term": "input", "compound": "compound term", "curie": "expected, given context",
         "curation_note": "rationale", "curation_status": "curation decision"}, vocab_col="Implicit basis")

    curies = sorted({str(r["curie"]).strip() for r in rows if r["curie"] and str(r["curie"]).strip().upper() != "NA"})
    look = kg2c_lookup(curies)
    for r in rows:
        c = str(r["curie"]).strip() if r["curie"] else ""
        if c in look:
            r["derived_curie_label_kg2c"] = look[c][1]

    df = pd.DataFrame(rows, columns=COLUMNS)
    assert df["gold_id"].is_unique, df[df["gold_id"].duplicated()]["gold_id"].tolist()
    prov = pd.DataFrame(PROVENANCE)
    readme = [
        f"Harmonized gold of the BiomedCAT slide decks, built on {date.today().isoformat()} by scripts/build_gold_harmonized.py.",
        "Values are copied verbatim from the curated sources; derived_* columns are computed (compound flag, KG2c label of the gold CURIE, used to match English readings of French terms). Identifiers are not scored.",
        "row_kind: positive = term on the slide with a Biolink class; implicit = expected by the expert, not written on the slide; adversarial = must receive neither class nor CURIE; na_on_slide = written on the slide, judged not an entity; removed = dropped by the expert.",
        "vocabulary_type comes from the VocabularyType (FSHD1multiscales) or Implicit basis (glycolysis) column; FSHD implicit terms come from fshd_implicit_expected.json; for VoieA, VoieB, VoieC and VoiesA-C the terms were read on the slide and are assumed explicit (vocabulary_type_origin says so).",
        "included = False when the curation status is delete/drop; empty status = proposal kept (as the sheets' READMEs state).",
        "A CURIE of NA or empty on a positive row means a term kept without identifier (type-only). NameRes / NodeNorm candidate columns were not carried over.",
        "Scoring uses included rows only: positive + implicit for reading recall, positive for typing, adversarial for false positives.",
        "See gold_harmonized_verification.md for the cell-by-cell check against the sources.",
    ]
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="terms", index=False)
        prov.to_excel(xw, sheet_name="provenance", index=False)
        pd.DataFrame({"README": readme}).to_excel(xw, sheet_name="README", index=False)
    OUT_JSON.write_text(json.dumps({"built": date.today().isoformat(), "readme": readme, "terms": json.loads(df.to_json(orient="records"))}, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- verification: re-read every traced cell with pandas and compare with the harmonized value
    hx = pd.read_excel(OUT_XLSX, sheet_name="terms")
    mism, checked = [], 0
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    by_row: dict[tuple[str, str, int], dict[str, str]] = {}
    for fname, sheet, excel_row, scol, hcol in TRACE:
        by_row.setdefault((fname, sheet, excel_row), {})[hcol] = scol
    def norm(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, (bool, type(pd.NA))) or str(v) in ("True", "False"):  # JSON booleans are stored as 1/0 next to the sheets' 0/1
            return "1" if str(v) == "True" else "0"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()
    for (fname, sheet, excel_row), cols in by_row.items():
        if (fname, sheet) not in cache:
            cache[(fname, sheet)] = pd.read_excel(GOLD / fname, sheet_name=sheet, header=0)
        srow = cache[(fname, sheet)].iloc[excel_row - 2]
        hrow = hx[(hx["source_file"] == fname) & (hx["source_sheet"] == sheet) & (hx["source_id"].astype(str) == norm(srow[cols["source_id"]]))]
        assert len(hrow) == 1, (fname, sheet, excel_row)
        for hcol, scol in cols.items():
            checked += 1
            a, b = norm(srow[scol]), norm(hrow.iloc[0][hcol])
            if a != b:
                mism.append((fname, sheet, excel_row, scol, a[:60], b[:60]))
    # JSON sources: compare field by field
    for src_name, key, kinds in (("fshd_slides_gold_v2.json", "positives", "positive"), ("fshd_slides_gold_v2.json", "adversarial", "adversarial"),
                                 ("fshd_implicit_expected.json", "entities", "implicit"), ("fshd_implicit_expected.json", "removed_by_expert", "removed")):
        g = json.loads((GOLD / src_name).read_text(encoding="utf-8"))[key]
        for e in g:
            hrow = hx[(hx["source_file"] == src_name) & (hx["row_kind"] == kinds) & (hx["source_id"] == e["id"])]
            assert len(hrow) == 1, (src_name, e["id"])
            for f, hcol in (("term", "term"), ("biolink_class", "biolink_class"), ("curie", "curie"), ("compound", "compound"), ("linkable", "linkable"), ("note", "curation_note"), ("reason", "curation_note")):
                if f in e:
                    checked += 1
                    if norm(e[f]) != norm(hrow.iloc[0][hcol]):
                        mism.append((src_name, key, e["id"], f, norm(e[f])[:60], norm(hrow.iloc[0][hcol])[:60]))
    lines = [f"# Verification of {OUT_XLSX.name} against its sources ({date.today().isoformat()})", "",
             f"Rows: {len(df)}; verbatim cells compared: {checked}; mismatches: {len(mism)}", ""]
    if mism:
        lines += ["| source | sheet | row | column | source value | harmonized value |", "|---|---|---|---|---|---|"]
        lines += [f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in mism]
    lines += ["", "## Rows per deck and kind", "", md_table(df.groupby(["deck", "row_kind", "included"]).size())]
    lines += ["", "## Explicit / implicit among included positives", "", md_table(df[df["included"] & df["row_kind"].isin(["positive", "implicit"])].groupby(["deck", "vocabulary_type"]).size())]
    OUT_VERIF.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {OUT_XLSX}, {OUT_JSON}, {OUT_VERIF}")
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
