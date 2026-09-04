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



        # self._load_data()

    INDEX_VERSION = "v2-name-in-doc"  # bump when _doc_text changes; the index is rebuilt when it differs

    @staticmethod
    def _doc_text(class_name: str, definition: str, examples_text: str) -> str:
        """Text indexed for one class, by both the dense and the sparse index."""
        parts = [f"{class_name}: {definition}".strip(": ")]
        if examples_text:
            parts.append(f"Examples: {examples_text}")
        return " ".join(parts)

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
                                                                                                                                                                                            
        logger.info("[*] Initializing BM25 index...")                                                                                                                                                
        bm25_corpus_tokens = []                                                                                                                                                                
        self._bm25_corpus_map = []                                                                                                                                                             
                                                                                                                                                                                            
        for class_name, entry in self.flat_data.items():
            meta = entry["metadata"]
            definition = meta.get("definition", "")
            examples_text = " ".join(meta.get("examples", []))
            combined_text = self._doc_text(class_name, definition, examples_text)

            tokens = self._tokenize(combined_text)                                                                                                                                             
            bm25_corpus_tokens.append(tokens)                                                                                                                                                  
            self._bm25_corpus_map.append(class_name)                                                                                                                                           
                                                                                                                                                                                            
        if bm25_corpus_tokens:                                                                                                                                                                 
            self.bm25 = BM25Okapi(bm25_corpus_tokens)                                                                                                                                          
        else:                                                                                                                                                                                  
            self.bm25 = None

    def _build_chroma_index(self) -> None:
        """Construit les index ChromaDB (Dense) et BM21/BM25 (Sparse)."""
        logger.info(f"[*] Indexation starts for {len(self.flat_data)} classes...")
        
        dense_ids = []
        dense_documents = []
        dense_metadatas = []
        
        for class_name, entry in self.flat_data.items():
            # Dense document = class name + definition + examples. The name matters: a query such
            # as "a protein" must land on 'protein', whose definition alone never says "protein".
            meta = entry["metadata"]
            definition = meta.get("definition", "")
            examples_text = " ".join(meta.get("examples", []))
            doc_text = self._doc_text(class_name, definition, examples_text)
            
            dense_ids.append(class_name)
            dense_documents.append(doc_text)
            dense_metadatas.append(entry["metadata"])

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

        # --- 3. Fusion RRF (Reciprocal Rank Fusion) ---
        # Formule: Score(d) = sum( 1 / (k + rank_dense) + 1 / (k + rank_sparse) )
        k = 60
        rrf_scores = defaultdict(float)

        for rank, class_id in enumerate(dense_results):
            rrf_scores[class_id] += 1.0 / (k + rank)

        for rank, class_id in enumerate(sparse_results):
            rrf_scores[class_id] += 1.0 / (k + rank)

        # Trier les classes par le score RRF final
        sorted_classes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in sorted_classes[:top_k]]

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
