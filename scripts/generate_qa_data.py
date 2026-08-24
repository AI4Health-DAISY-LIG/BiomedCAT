import json
import requests
import sys
from pathlib import Path
from typing import List, Dict, Any
import ollama

# Importation du moteur RAG pour accéder aux données réelles de l'index
try:
    from biomedcat.stages.rag_engine import build_rag
    from biomedcat.config import Settings
except ImportError as e:
    print(f"Error: Could not import BiomedRAG components. {e}")
    sys.exit(1)

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b-it-qat"
SAMPLES_PER_CLASS = 5  # Nombre de requêtes à générer par classe pour la significativité statistique
OUTPUT_FILE = Path("data/qa_data.json")

class QAGenerator:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.qa_pairs: List[Dict[str, str]] = []

    def _query_ollama(self, class_name: str, definition: str) -> List[str]:
        """
        Interroge Ollama pour générer des termes biomédicaux basés sur la définition.
        Utilise le mode JSON d'Ollama pour une extraction robuste.
        """

        qa_rag_schema = {                                                                                                                                                                          
            "type": "array",                                                                                                                                                                       
            "items": {                                                                                                                                                                             
                "type": "object",                                                                                                                                                                  
                "properties": {                                                                                                                                                                    
                    "query": {"type": "string"},                                                                                                                                                   
                    "expected_class": {"type": "string"}                                                                                                                                           
                },                                                                                                                                                                                 
                "required": ["query", "expected_class"]                                                                                                                                            
            }                                                                                                                                                                                      
        } 
        
        prompt = (
            f"You are a biomedical expert. Given the following Biolink class definition: '{definition}'\n"
            f"Generate EXACTLY {SAMPLES_PER_CLASS} distinct, short biomedical terms or queries "
            f"(1-3 words each) that belong to the class '{class_name}'.\n"
            f"Return the result ONLY as a valid list of EXACTLY {SAMPLES_PER_CLASS} JSON arrays of strings as a list. Example: ["
            "{"
            "    'query': 'insulin',"
            "    'expected_class': 'protein'"
            "},"
            "{"
            "    'query': 'diabetes mellitus',"
            "    'expected_class': 'disease'"
            "} ... ]"
        )

        generation_options = {
            "temperature": 0.0,       # Low = deterministic/factual, High = creative
            "top_k": 40,              # Limits pool of next-word choices
            "top_p": 0.9,             # Nucleus sampling threshold
            "num_ctx": 4096          # Sets the context window length in tokens
        }

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                format=qa_rag_schema,
                messages=[
                    {
                        "role": "system", 
                        "content": "You must bypass your thinking process. Do not use <think> tags. Provide the final output immediately."
                    },
                        
                    {
                        "role": "user", 
                         "content": prompt
                    },
                ],
                options=generation_options
            )
            
            raw_content = response["message"]["content"]
            generated_queries = json.loads(raw_content)
            
            if isinstance(generated_queries, list):
                return generated_queries
            return []
        except Exception as e:
            print(f"  [!] Error querying Ollama for {class_name}: {e}")
            return []

    def run(self):
        """Parcourt toutes les classes de l'index et génère le dataset."""
        # On récupère toutes les classes présentes dans l'index flat (Chroma/BM25)
        classes_to_process = self.rag_engine.flat_data
        
        if not classes_to_process:
            print("Error: No classes found in RAG engine. Is the index built?")
            return

        print(f"[*] Starting QA generation for {len(classes_to_process)} classes...")

        for class_name, entry in classes_to_process.items():
            definition = entry["metadata"].get("definition", "")
            if not definition:
                continue

            print(f"[*] Generating samples for: {class_name}...")
            queries = self._query_ollama(class_name, definition)

            for q in queries:
                self.qa_pairs.append({
                    "query": q.strip(),
                    "expected_class": class_name
                })

        # Sauvegarde du résultat
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Success! Generated {len(self.qa_pairs)} test cases.")
        print(f"[+] Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Initialisation du moteur avec les paramètres par défaut
    settings = Settings()
    try:
        engine = build_rag(settings)
        generator = QAGenerator(engine)
        generator.run()
    except Exception as e:
        print(f"Critical Error during execution: {e}")
        sys.exit(1)
