import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re
import os
import spacy
import torch
import logging

# Désactiver les logs HTTP de huggingface_hub et httpx
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from biomedcat.config import Settings
from biomedcat.stages.biolink_yml_processor import run_smart_update, biolink_yml_processor

class BiomedRAG:
    """
    Moteur de recherche hybride (Dense + Sparse) pour le modèle Biolink.
    Implémente la Phase 2 : Double indexation et Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config: Settings, force_rebuild: bool = False):
        self.config = config
        self.data_path = Path(config.internal_data_path) / "biolink_classes_flat.json"
        self.flat_data: Dict[str, Any] = {}
        self.needs_reindexing = force_rebuild or is_empty_dir(config.chroma_db_path)
        self.exemplars = None
        # Weight of the exemplar leg in the fusion (RAG_EXEMPLAR_WEIGHT, default 1.0 = same as dense).
        self.exemplar_weight = float(os.getenv("RAG_EXEMPLAR_WEIGHT", "1.0"))

        # Load Models
        # Bind embedding model                                                                                                                          
        logger.info(f"[*] Loading embedding model: {self.config.RAG_embedding_model}")
        
        # Utiliser directement l'embedding function de ChromaDB
        # Elle gérera automatiquement le caching via SentenceTransformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.RAG_embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("[*] Loading scispaCy model for tokenization...")
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except Exception as e:  # missing model or a model built for another spaCy version
            logger.warning("[!] scispaCy model unavailable (%s). Falling back to basic tokenizer.", type(e).__name__)
            self.nlp = None

        # Indexeurs
        self.chroma_client = chromadb.PersistentClient(path=config.chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="biomedcat_dense",embedding_function=self.embedding_fn)

        # Load Data   
        if self.data_path.exists():                                                                                                                                                                       
            self._load_data()
            self._setup_bm25()                                                                                                                                                                             
        else:                                                                                                                                                                                             
            logger.warning("[!] Biolink data file not found. Indexing.")                                                                                                                    
            self.flat_data = {}
            self.needs_reindexing = True                                                                                                                                                             
                                                                                                                                                                                                          
        # A stale index scheme is rebuilt only when RAG_REINDEX=1 is set, so that a running
        # pipeline sharing the index is never disturbed by another process; a warning says so.
        if not self.needs_reindexing and not self._index_is_current():
            if os.getenv("RAG_REINDEX", "0") == "1":
                self.needs_reindexing = True
            else:
                logger.warning("[!] ChromaDB index built with an older scheme; set RAG_REINDEX=1 to rebuild it (%s).", self.INDEX_VERSION)

        if self.needs_reindexing:
            logger.info("[*] Rebuilding the ChromaDB index (%s)...", self.INDEX_VERSION)
            if self.collection.count() > 0:
                self.chroma_client.delete_collection("biomedcat_dense")
                self.collection = self.chroma_client.get_or_create_collection(name="biomedcat_dense", embedding_function=self.embedding_fn)
            self._build_chroma_index()
            self._mark_index_version()
            self._setup_bm25()
        else:                                                                                                                                                                                             
            logger.info("[*] ChromaDB doesn't need existing, charging existing DB.")
            if not hasattr(self, 'bm25') or self.bm25 is None:                                                                                                                                 
                self._setup_bm25()

        # Exemplar collection (optional, env RAG_EXEMPLARS or data/biolink_exemplars.parquet);
        # disabled with RAG_EXEMPLARS=off so the definition-only index can be measured alone.
        if os.getenv("RAG_EXEMPLARS", "") != "off":
            try:
                self._load_exemplars()
            except Exception as e:
                logger.error("[!] exemplar index unavailable: %s", e)

    INDEX_VERSION = "v3-aliases-ancestors"  # bump when _doc_text or the indexed set changes

    @staticmethod
    def _doc_text(class_name: str, meta: Dict[str, Any]) -> str:
        """Text indexed for one class, by both the dense and the sparse index.

        Name, aliases (e.g. disease: condition, disorder), definition, the ancestor chain (so a
        query about a 'disease' also touches 'disease or phenotypic feature' documents) and the
        examples. The examples are the ';'-joined string written by the processor.
        """
        parts = [class_name]
        aliases = meta.get("aliases") or ""
        if aliases:
            parts.append(f"also called {aliases}")
        definition = (meta.get("definition") or "").strip()
        if definition:
            parts.append(f": {definition}")
        ancestors = [a for a in (meta.get("ancestors") or []) if a not in ("entity", "named thing")]
        if ancestors:
            parts.append(f"A kind of {' > '.join(ancestors)}.")
        examples = meta.get("examples") or ""
        if isinstance(examples, list):
            examples = "; ".join(examples)
        if examples:
            parts.append(f"Examples: {examples}")
        return " ".join(parts)

    @staticmethod
    def _indexable(meta: Dict[str, Any]) -> bool:
        """Deprecated and abstract classes are navigation nodes, not retrieval targets."""
        return not meta.get("deprecated") and not meta.get("abstract")

    @staticmethod
    def _chroma_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma accepts scalars only: lists become comma-joined strings."""
        out = {}
        for k, v in meta.items():
            if isinstance(v, list):
                out[k] = ", ".join(map(str, v))
            elif v is None:
                out[k] = ""
            else:
                out[k] = v
        return out

    def _index_is_current(self) -> bool:
        """True when the persisted index was built with the current _doc_text scheme."""
        marker = Path(self.config.chroma_db_path) / "index_version.txt"
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == self.INDEX_VERSION

    def _mark_index_version(self) -> None:
        (Path(self.config.chroma_db_path) / "index_version.txt").write_text(self.INDEX_VERSION, encoding="utf-8")

    def _load_data(self) -> None:
        """Charge les données du fichier JSON produit par biolink_yml_processor"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.flat_data = json.load(f)

    def _tokenize(self, text: str) -> List[str]:                                                                                                                                               
        """                                                                                                                                                                                    
        Tokenisation experte :                                                                                                                                                                 
        1. Utilise la structure grammaticale de scispaCy.                                                                                                                                      
        2. Ne garde que les entités sémantiques (Noms, Noms propres, Adjectifs).                                                                                                               
        3. Supprime les doublons et le bruit (stop words, ponctuation).                                                                                                                        
        """                                                                                                                                                                                    
        if not text:                                                                                                                                                                           
            return []                                                                                                                                                                          
                                                                                                                                                                                            
        # Fallback si le modèle n'est pas chargé                                                                                                                                               
        if not self.nlp:                                                                                                                                                                       
            tokens = re.findall(r'\w+', text.lower())                                                                                                                                          
            return list(set(tokens)) # Supprime les doublons                                                                                                                                   
                                                                                                                                                                                            
        # Traitement avec scispaCy                                                                                                                                                             
        doc = self.nlp(text)                                                                                                                                                                   
                                                                                                                                                                                            
        # Utilisation d'un set pour garantir l'unicité (suppression des doublons)                                                                                                              
        important_tokens = set()                                                                                                                                                               
                                                                                                                                                                                            
        for token in doc:                                                                                                                                                                      
            # On ne garde que les tokens qui sont :                                                                                                                                            
            # - Pas un stop word, pas de la ponctuation, pas un espace                                                                                                                         
            # - Un Nom (NOUN), un Nom Propre (PROPN) ou un Adjectif (ADJ)                                                                                                                      
            if (not token.is_stop and                                                                                                                                                          
                not token.is_punct and                                                                                                                                                         
                not token.is_space and                                                                                                                                                         
                token.pos_ in {"NOUN", "PROPN", "ADJ"}):                                                                                                                                       
                                                                                                                                                                                            
                important_tokens.add(token.text.lower())                                                                                                                                       
                                                                                                                                                                                            
        return list(important_tokens)

    def _setup_bm25(self) -> None:                                                                                                                                                             
        """Initialise l'index Sparse (BM25) à partir des données chargées."""                                                                                                                  
        if not self.flat_data:                                                                                                                                                                 
            return                                                                                                                                                                             
                                                                                                                                                                                            
        # Without scispaCy the sparse leg tokenizes with a bare regex, returns nothing for half of
        # the queries and lowers Hit@1 below the dense leg alone (measured): it is disabled then.
        if self.nlp is None:
            logger.warning("[!] scispaCy unavailable: sparse (BM25) leg disabled, dense retrieval only.")
            self.bm25 = None
            self._bm25_corpus_map = []
            return

        logger.info("[*] Initializing BM25 index...")
        bm25_corpus_tokens = []
        self._bm25_corpus_map = []
        for class_name, entry in self.flat_data.items():
            meta = entry["metadata"]
            if not self._indexable(meta):
                continue
            tokens = self._tokenize(self._doc_text(class_name, meta))
            bm25_corpus_tokens.append(tokens)
            self._bm25_corpus_map.append(class_name)

        self.bm25 = BM25Okapi(bm25_corpus_tokens) if bm25_corpus_tokens else None

    def _build_chroma_index(self) -> None:
        """Construit les index ChromaDB (Dense) et BM21/BM25 (Sparse)."""
        logger.info(f"[*] Indexation starts for {len(self.flat_data)} classes...")
        
        dense_ids = []
        dense_documents = []
        dense_metadatas = []
        
        for class_name, entry in self.flat_data.items():
            meta = entry["metadata"]
            if not self._indexable(meta):
                continue
            dense_ids.append(class_name)
            dense_documents.append(self._doc_text(class_name, meta))
            dense_metadatas.append(self._chroma_metadata(meta))
        logger.info("[*] %d indexable classes (deprecated and abstract excluded)", len(dense_ids))

        # Injection dans ChromaDB
        if dense_ids:
            self.collection.add(
                ids=dense_ids,
                documents=dense_documents,
                metadatas=dense_metadatas
            )

        logger.info("[+] Indexation : done.")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Recherche hybride utilisant le Reciprocal Rank Fusion (RRF).
        """
        if not self.flat_data:
            return []

        # Tokenize query once and cache it
        query_tokens = self._tokenize(query)

        # --- 1. Recherche Dense (ChromaDB) ---
        dense_results = []
        try:
            # Utiliser le modèle directement pour encoder la requête
            query_res = self.collection.query(query_texts=[query], n_results=top_k)
            
            # query_res['ids'] est une liste de listes [[id1, id2...]]
            dense_results = query_res['ids'][0] if query_res['ids'] else []
        except Exception as e:
            logger.error(f"[!] Erreur recherche Dense : {e}")

        # --- 2. Recherche Sparse (BM25) ---
        sparse_results = []
        if self.bm25:
            # On récupère les scores pour toutes les classes
            scores = self.bm25.get_scores(query_tokens)
            # On trie les indices par score décroissant
            top_indices = np.argsort(scores)[::-1][:top_k]
            sparse_results = [self._bm25_corpus_map[i] for i in top_indices if scores[i] > 0]

        # --- 2b. Exemplar leg: nearest generated mentions, one class per best exemplar ---
        exemplar_results = self._search_exemplars(query, top_k) if self.exemplars is not None else []

        # --- 3. Fusion RRF (Reciprocal Rank Fusion) ---
        # Formule: Score(d) = sum( 1 / (k + rank_dense) + 1 / (k + rank_sparse) )
        k = 60
        rrf_scores = defaultdict(float)

        for rank, class_id in enumerate(dense_results):
            rrf_scores[class_id] += 1.0 / (k + rank)

        for rank, class_id in enumerate(sparse_results):
            rrf_scores[class_id] += 1.0 / (k + rank)

        for rank, class_id in enumerate(exemplar_results):
            rrf_scores[class_id] += self.exemplar_weight / (k + rank)

        # Trier les classes par le score RRF final
        sorted_classes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in sorted_classes[:top_k]]

    # ------------------------------------------------------------------------------------
    # Exemplar index: mention-to-mention similarity, the leg the definitions cannot provide
    # ------------------------------------------------------------------------------------

    def _load_exemplars(self) -> None:
        """Index data/biolink_exemplars.parquet (columns class, exemplar) if present.

        Each exemplar is its own document with the class as metadata; the collection is
        rebuilt when the file changes (row count or mtime differ from the stored marker).
        """
        self.exemplars = None
        path = Path(os.getenv("RAG_EXEMPLARS") or Path(self.config.internal_data_path) / "biolink_exemplars.parquet")
        if not path.is_file():
            return
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["class", "exemplar"])
        classes = table.column("class").to_pylist()
        texts = table.column("exemplar").to_pylist()
        keep = [i for i, c in enumerate(classes) if c in self.flat_data and self._indexable(self.flat_data[c]["metadata"])]
        marker = Path(self.config.chroma_db_path) / "exemplars_version.txt"
        stamp = f"{len(keep)}:{int(path.stat().st_mtime)}"
        collection = self.chroma_client.get_or_create_collection(name="biomedcat_exemplars", embedding_function=self.embedding_fn)
        if not (marker.is_file() and marker.read_text(encoding="utf-8").strip() == stamp and collection.count() == len(keep)):
            logger.info("[*] Indexing %d exemplars for %d classes...", len(keep), len(set(classes[i] for i in keep)))
            if collection.count():
                self.chroma_client.delete_collection("biomedcat_exemplars")
                collection = self.chroma_client.get_or_create_collection(name="biomedcat_exemplars", embedding_function=self.embedding_fn)
            for start in range(0, len(keep), 500):
                idx = keep[start:start + 500]
                collection.add(ids=[f"ex{i}" for i in idx], documents=[texts[i] for i in idx], metadatas=[{"class": classes[i]} for i in idx])
            marker.write_text(stamp, encoding="utf-8")
        self.exemplars = collection
        logger.info("[+] Exemplar index ready (%d documents).", collection.count())

    def _search_exemplars(self, query: str, top_k: int) -> List[str]:
        """Classes ranked by their best-matching exemplar (first occurrence in the nearest list)."""
        try:
            res = self.exemplars.query(query_texts=[query], n_results=min(top_k * 6, self.exemplars.count()))
        except Exception as e:
            logger.error("[!] exemplar search failed: %s", e)
            return []
        ranked: List[str] = []
        for meta in (res.get("metadatas") or [[]])[0]:
            c = meta.get("class")
            if c and c not in ranked:
                ranked.append(c)
            if len(ranked) >= top_k:
                break
        return ranked

    def get_context(self, class_names: List[str]) -> str:
        """
        Génère le bloc de texte contextuel pour un agent.
        Transforme les noms de classes en descriptions textuelles complètes.
        """
        if not class_names:
            return "Aucune classe pertinente trouvée."

        context_parts = []
        for name in class_names:
            if name in self.flat_data:
                entry = self.flat_data[name]
                meta = entry["metadata"]
                definition = meta.get("definition", "No definition available.")
                examples = ", ".join(meta.get("examples", []))
                
                part = f"Class: {name}. Definition: {definition}. Examples: [{examples}]."
                context_parts.append(part)
        
        return "\n".join(context_parts)

def is_empty_dir(path):
    is_empty = True
    for _ in os.scandir(path):
        is_empty = False
        break
    return is_empty

def build_rag(settings: str = Settings()):                                                                                                                                                                              
                                                                                                                                                                                                          
    # 1. Verify biolink yml update                                                                                                                          
    was_updated = run_smart_update(                                                                                                                                                                       
        source_url=settings.biolink_model_data,                                                                                                                                                         
        processor_func=biolink_yml_processor                                                                                                                                                              
    )                                                                                                                                                                                      
                                                                                                                                                                                                          
    # 2. Initialize RAG                                                                                                                                  
    rag_engine = BiomedRAG(settings, force_rebuild=was_updated)                                                                                                                                           
                                                                                                                                                                                                          
    logger.info("[+] RAG ready.")
    return rag_engine                                                                                                                                                                     

if __name__ == "__main__":                                                                                                                                                                                
    rag_engine = build_rag()
    results = rag_engine.search("CACNA1C gene")
    context = rag_engine.get_context(["gene"])
    print('--- Results:')
    print(results)
    print('--- Context:')
    print(context)
