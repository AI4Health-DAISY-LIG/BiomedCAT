#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_class_frequencies.py

Génère data/biolink_class_frequencies.json à partir d’un corpus PubMed.
Le script :

1️⃣ récupère N abstracts (par défaut 10 000) via Entrez,
2️⃣ extrait les entités biomédicales avec le NER Agent déjà présent,
3️⃣ mappe chaque entité à la classe Biolink la plus probable (RAG.search),
4️⃣ agrège les comptes et écrit le JSON.

Utilisation :
    export MAX_ABSTRACTS=15000   # (optionnel) nombre d’abstraits à analyser
    python scripts/build_class_frequencies.py
"""

import os
import json
import time
import math
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from Bio import Entrez
from biomedcat.config import Settings
from biomedcat.stages.rag_engine import build_rag
from biomedcat.stages.ner_agent import NERAgentPipeline
import logging

# -------------------------------------------------------------------------
# 1️⃣  Paramètres (modifiable via variables d’environnement)
# -------------------------------------------------------------------------
# Adresse e‑mail obligatoire pour Entrez (NCBI)
Entrez.email = "sandrine.muller@univ-granoble-alpes.org"
Entrez.tool = "biomedcat_frequencies"

MAX_ABSTRACTS = int(os.getenv("MAX_ABSTRACTS", "10000"))   # nb d’abstraits à récupérer
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "100"))       # nb d’IDs par appel Entrez
RETRY_COUNT   = int(os.getenv("RETRY_COUNT", "3"))       # retries sur les appels HTTP
RETRY_DELAY   = float(os.getenv("RETRY_DELAY", "1.0"))   # délai initial (s) entre retries
OUTPUT_PATH   = Path("data/biolink_class_frequencies.json")

# Set up logger
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 2️⃣  Initialisation des composants du pipeline
# -------------------------------------------------------------------------
settings = Settings()
rag_engine = build_rag(settings)   # moteur hybride dense+BM25
ner_agent = NERAgentPipeline(
    rag_unseen=rag_engine,
    classification_model_id=settings.classification_model_id,
    sanitization_model_id=settings.sanitization_model_id,
)

# -------------------------------------------------------------------------
# 3️⃣  Fonctions utilitaires
# -------------------------------------------------------------------------
def _retry(func):
    """Décorateur simple de retry avec back‑off exponentiel."""
    def wrapper(*args, **kwargs):
        delay = RETRY_DELAY
        for attempt in range(RETRY_COUNT):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == RETRY_COUNT - 1:
                    raise
                logger.warning(f"[!] {func.__name__} failed (attempt {attempt+1}/{RETRY_COUNT}): {e}")
                time.sleep(delay)
                delay *= 2
        return None
    return wrapper

@_retry
def fetch_pubmed_ids(term: str = "cancer", max_ids: int = MAX_ABSTRACTS) -> list[str]:
    """Retourne une liste de PMIDs (strings)."""
    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_ids, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

@_retry
def fetch_abstracts_batch(pmids: list[str]) -> list[str]:
    """Récupère les résumés (texte brut) pour un lot de PMIDs."""
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
    raw = handle.read()
    handle.close()
    # Entrez renvoie les abstracts concaténés, séparés par deux sauts de ligne
    return [a.strip() for a in raw.split("\n\n") if a.strip()]

def fetch_all_abstracts(pmids: list[str]) -> list[str]:
    """Itère sur les PMIDs par BATCH_SIZE et agrège les abstracts."""
    abstracts = []
    for start in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[start:start + BATCH_SIZE]
        abstracts.extend(fetch_abstracts_batch(batch))
    return abstracts

def map_term_to_class(term: str) -> str | None:
    """Utilise le RAG pour obtenir la classe la plus probable (top‑1)."""
    try:
        top = rag_engine.search(term, top_k=1)
        return top[0] if top else None
    except Exception:
        return None

def count_classes(texts: list[str]) -> Counter:
    """Parcourt chaque texte, extrait les entités, les mappe → classe, incrémente le compteur."""
    freq = Counter()
    for txt in tqdm(texts, desc="Processing abstracts"):
        # Extraction brute d’entités (retourne [(term, sentence), ...])
        candidates = ner_agent._extract_candidates_model(txt, txt)
        if not candidates:
            continue
        for term, _sentence in candidates:
            cls = map_term_to_class(term)
            if cls:
                freq[cls] += 1
    return freq

# -------------------------------------------------------------------------
# 4️⃣  Exécution principale
# -------------------------------------------------------------------------
def main():
    logger.info("[*] Récupération des PMIDs PubMed …")
    pmids = fetch_pubmed_ids()
    logger.info(f"    → {len(pmids)} IDs récupérés.")

    logger.info("[*] Téléchargement des abstracts …")
    abstracts = fetch_all_abstracts(pmids)
    logger.info(f"    → {len(abstracts)} abstracts chargés.")

    logger.info("[*] Comptage des classes Biolink …")
    class_counts = count_classes(abstracts)

    # Export JSON (compte brut)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in class_counts.items()}, f, indent=2, ensure_ascii=False)

    # Affichage de quelques stats rapides
    total = sum(class_counts.values())
    logger.info("\n[+] Fichier généré : %s", OUTPUT_PATH)
    logger.info(f"    Total d’occurrences comptées : {total}")
    logger.info(f"    Nombre de classes différentes : {len(class_counts)}")
    logger.info("\nTop‑10 classes (fréquence brute) :")
    for cls, cnt in class_counts.most_common(10):
        logger.info(f"    {cls:30s} → {cnt}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()
