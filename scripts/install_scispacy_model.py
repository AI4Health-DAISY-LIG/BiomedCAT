#!/usr/bin/env python
"""Install the scispaCy tokenizer model (en_core_sci_sm) used by the sparse (BM25) leg of the Biolink retriever. The scispacy package itself is not needed: the model only needs spaCy.

Why a script and not a pyproject dependency: the only published en_core_sci_sm build (0.5.4)
declares `spacy>=3.7.4,<3.8.0`, which conflicts with BiomedCAT's spaCy 3.8 pin, and its
config.cfg stores `include_static_vectors = "False"` as a string, which the pydantic-2 config
validation of spaCy 3.7+ rejects ("'False' is not <class 'bool'>"). The model itself works with
spaCy 3.8 once that value is a real boolean. This script therefore installs the tarball without
its dependency pins and patches the two config lines; it is idempotent.

Usage (from the BiomedCAT root):
  uv run python scripts/install_scispacy_model.py
Without the model, rag_engine falls back to dense retrieval only and logs a warning.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"


def main() -> int:
    try:
        importlib.import_module("en_core_sci_sm")
    except ImportError:
        print(f"[*] installing {URL} (no dependency pins)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", URL])
    pkg = Path(importlib.import_module("en_core_sci_sm").__file__).parent
    patched = 0
    for cfg in pkg.rglob("config.cfg"):
        text = cfg.read_text(encoding="utf-8")
        new = re.sub(r'include_static_vectors = "(False|True)"', lambda m: f"include_static_vectors = {m.group(1).lower()}", text)
        if new != text:
            cfg.write_text(new, encoding="utf-8")
            patched += 1
    print(f"[*] config files patched: {patched}")
    import spacy
    nlp = spacy.load("en_core_sci_sm")
    print(f"[+] en_core_sci_sm {nlp.meta['version']} loads with spaCy {spacy.__version__}: {[t.text for t in nlp('DUX4 mRNA stability in FSHD1')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
