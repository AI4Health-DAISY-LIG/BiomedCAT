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

        # Load ChromaDB
        # Bind embedding model                                                                                                                          
        print(f"[*] Loading embedding model: {self.config.RAG_embedding_model}")                                                                                                                          
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(                                                                                                                     
            model_name=self.config.RAG_embedding_model                                                                                                                                                    
        )
        # Indexeurs
        self.chroma_client = chromadb.PersistentClient(path=config.chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="biomedcat_dense",embedding_function=self.embedding_fn)

        # Load Data   
        if self.data_path.exists():                                                                                                                                                                       
            self._load_data()                                                                                                                                                                             
        else:                                                                                                                                                                                             
            print("[!] Biolink data file not found. Indexing.")                                                                                                                    
            self.flat_data = {}
            self.needs_reindexing = True                                                                                                                                                             
                                                                                                                                                                                                          
        # Indexation                                                                                                                                                                        
        if self.needs_reindexing:                                                                                                                                                                         
            print("[*] Reconstruction de l'index ChromaDB en cours...")                                                                                                                                   
            self.build_indices()                                                                                                                                                                          
        else:                                                                                                                                                                                             
            print("[*] ChromaDB doesn't need existing, charging existing DB.")

        # Corpus pour BM25 (mapping index -> class_name)
        self._bm25_corpus_map: List[str] = []
        self._bm25_tokenized_corpus: List[List[str]] = []

        
        # self._load_data()

    def _load_data(self) -> None:
        """Charge les données du fichier JSON produit par biolink_yml_processor"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.flat_data = json.load(f)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenisation simple pour l'index Sparse (BM25)."""
        return re.findall(r'\w+', text.lower())

    def build_indices(self) -> None:
        """Construit les index ChromaDB (Dense) et BM21/BM25 (Sparse)."""
        print(f"[*] Indexation starts for {len(self.flat_data)} classes...")
        
        dense_ids = []
        dense_documents = []
        dense_metadatas = []
        
        bm25_corpus_tokens = []
        self._bm25_corpus_map = []

        for class_name, entry in self.flat_data.items():
            # 1. Préparation pour l'index Dense (ChromaDB)
            # On utilise le doc_text qui contient déjà la définition + les exemples
            doc_text = entry["doc_text"]
            dense_ids.append(class_name)
            dense_documents.append(doc_text)
            dense_metadatas.append(entry["metadata"])

            # 2. Préparation pour l'index Sparse (BM25)
            # On crée un corpus textuel basé sur la définition et les exemples pour le matching mot-clé
            meta = entry["metadata"]
            definition = meta.get("definition", "")
            examples_text = " ".join(meta.get("examples", []))
            combined_text = f"{definition} {examples_text}"
            
            tokens = self._tokenize(combined_text)
            bm25_corpus_tokens.append(tokens)

        # Indexing BM25
        if bm25_corpus_tokens:
            self.bm25 = BM25Okapi(bm25_corpus_tokens)
            self._bm25_corpus_map = list(self.flat_data.keys()) # Assurer la correspondance

        # Injection dans ChromaDB
        if dense_ids:
            self.collection.add(
                ids=dense_ids,
                documents=dense_documents,
                metadatas=dense_metadatas
            )

        print("[+] Indexation : done.")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Recherche hybride utilisant le Reciprocal Rank Fusion (RRF).
        """
        if not self.flat_data:
            return []

        # --- 1. Recherche Dense (ChromaDB) ---
        dense_results = []
        try:
            query_res = self.collection.query(query_texts=[query], n_results=top_k)
            # query_res['ids'] est une liste de listes [[id1, id2...]]
            dense_results = query_res['ids'][0] if query_res['ids'] else []
        except Exception as e:
            print(f"[!] Erreur recherche Dense : {e}")

        # --- 2. Recherche Sparse (BM25) ---
        sparse_results = []
        if self.bm25:
            query_tokens = self._tokenize(query)
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
        Génère le bloc de texte contextuel pour l'Agent (Phase 3).
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
                                                                                                                                                                                                          
    print("[+] RAG ready.")
    return rag_engine                                                                                                                                                                     
                                                                                                                                                                                                          
if __name__ == "__main__":                                                                                                                                                                                
    rag_engine = build_rag()

    print('bob')