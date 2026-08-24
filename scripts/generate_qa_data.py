import json
import yaml
import ollama
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from biomedcat.stages.rag_engine import build_rag
from biomedcat.config import Settings

# Configuration des chemins
CONFIG_PATH = Path("scripts/er_test-suite_configuration.yml")
NESTED_DATA_PATH = Path("data/biolink_classes_nested.json")
OUTPUT_PARQUET = Path("data/qa_dataset.parquet")
MODEL_NAME = "gemma4:e_4b-it-qat"

class StratifiedQAGenerator:
    def __init__(self, rag_engine, nested_data: Dict[str, Any], config: Dict[str, Any]):
        self.rag_engine = rag_engine
        self.nested_data = nested_data
        self.config = config
        self.all_samples = []
        # Map pour retrouver la profondeur et le statut de chaque classe
        self.class_metadata: Dict[str, Dict[str, Any]] = {}
        self._analyze_hierarchy()

    def _analyze_hierarchy(self):
        """Calcule la profondeur et le statut (feuille) de chaque classe."""
        print("[*] Analyzing hierarchy for stratified sampling...")
        
        # On parcourt l'arborescence pour calculer les profondeurs
        def traverse(node_name: str, depth: int):
            is_leaf = True
            children = self.nested_data.get(node_name, {}).get("children", {})
            
            if children:
                is_leaf = False
                for child in children:
                    traverse(child, depth + 1)
            
            self.class_metadata[node_name] = {
                "depth": depth,
                "is_leaf": is_leaf
            }

        # On identifie les racines (classes sans parent dans le JSON)
        roots = [name for name, data in self.nested_data.items() if not data.get("parent")]
        for root in roots:
            traverse(root, 1)

    def _get_sampling_plan(self) -> List[str]:
        """Détermine quelles classes échantillonner selon les contraintes du YAML."""
        classes_to_sample = []
        
        # Extraction des contraintes
        leaf_cfg = self.config.get("leaf_class", {})
        depth_cfg = self.config.get("hierarchy_depth", {})
        
        # 1. Groupement par bins de profondeur
        bins = depth_cfg.get("bins", [[1, 99]])
        
        # On prépare les classes par strate
        strata: Dict[str, List[int]] = defaultdict(list) # Note: using list of class names
        class_strata: Dict[str, List[str]] = defaultdict(list)

        for cls, meta in self.class_metadata.items():
            # Trouver le bin correspondant
            assigned_bin = "other"
            for b in bins:
                if b[0] <= meta["depth"] <= b[1]:
                    assigned_bin = f"{b[0]}-{b[1]}"
                    break
            
            # On ajoute un suffixe pour la distinction feuille/non-feuille
            strat_key = f"depth_{assigned_bin}_leaf_{meta['is_leaf']}"
            class_strata[strat_key].append(cls)

        # 2. Sélection des classes (Stratified Sampling)
        for strat_name, classes in class_strata.items():
            if not classes:
                continue
            # On prend un échantillon arbitraire (ex: 3 classes par strate pour l'exemple)
            sample_size = min(len(classes), 3) 
            import random
            selected = random.sample(classes, sample_size)
            classes_to_sample.extend(selected)

        return classes_to_sample

    def _query_ollama(self, class_name: str, definition: str) -> List[Dict[str, str]]:
        """Génère des paires query/class via Ollama avec schéma JSON strict."""
        prompt = (
            f"You are a biomedical expert. Given the definition: '{definition}'\n"
            f"Generate 3 distinct, short biomedical queries (1-3 words) that belong to the class '{class_name}'.\n"
            f"Return ONLY a JSON array of objects with keys 'query' and 'expected_class'."
        )

        schema = {
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

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                format=schema,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response["message"]["content"])
        except Exception as e:
            print(f"  [!] Error generating for {class_name}: {e}")
            return []

    def run(self):
        """Exécute le processus complet de génération."""
        classes_to_process = self._get_sampling_plan()
        print(f"[*] Selected {len(classes_to_process)} classes for sampling.")

        for class_name in classes_to_process:
            # On récupère la définition depuis le moteur RAG (flat_data)
            if class_name not in self.rag_engine.flat_data:
                continue
                
            definition = self.rag_engine.flat_data[class_name]["metadata"].get("definition", "")
            if not definition:
                continue

            print(f"[*] Generating queries for: {class_name}...")
            new_samples = self._query_ollama(class_name, definition)
            
            for sample in new_samples:
                # On s'assure que la classe attendue est bien celle qu'on traite
                sample["expected_class"] = class_name 
                self.all_samples.append(sample)

        if not self.all_samples:
            print("[!] No samples generated.")
            return

        # Conversion en DataFrame et export Parquet
        df = pd.DataFrame(self.all_samples)
        df.to_parquet(OUTPUT_PARQUET, engine='pyarrow', index=False)
        print(f"\n[+] Success! Saved {len(df)} samples to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    # 1. Load Config
    with open(CONFIG_PATH, "r") as f:
        config_data = yaml.safe_load(f)

    # 2. Init RAG and Data
    settings = Settings()
    engine = build_rag(settings)
    
    with open(NESTED_DATA_PATH, "r", encoding="utf-8") as f:
        nested_structure = json.load(f)

    # 3. Run Generator
    generator = StratifiedQAGenerator(engine, nested_structure, config_data)
    generator.run()
