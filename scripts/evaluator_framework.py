import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from abc.abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json

# ============================================================
# 1. INTERFACE ABSTRAITE (Modularité)
# ============================================================

class BaseEvaluatorEngine(ABC):
    """Interface commune pour comparer RAG et Agent."""
    @abstractmethod
    def get_top_k_results(self, query: str, k: int) -> List[str]:
        """Return a list of biomedical classes, ranked by pertinence."""
        pass

# =  Wrappers pour vos moteurs existants
class BiomedRAGEngineWrapper(BaseEvaluatorEngine):
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    def get_top_k_results(self, query: str, k: int) -> List[str]:
        # Utilise la méthode search existante du RAG
        return self.rag_engine.search(query, top_k=k)

class NERAgentEngineWrapper(BaseEvaluatorEngine):
    def __init__(self, agent_pipeline):
        self.agent_pipeline = agent_pipeline

    def get_top_k_results(self, query: str, k: int) -> List[str]:
        # L'agent extrait et classifie. On simule un top-K en extrayant les entités.
        # Note: Pour une comparaison juste, on traite la query comme une phrase.
        entities = self.agent_pipeline.extract([query])
        # On retourne les types trouvés (on peut adapter selon le besoin)
        return [e.type for e_text, e in entities]

# ============================================================
# 2. MOTEUR D'ANALYSE (Single-Pass & Semantic Metrics)
# ============================================================

class RAGAnalyzer:
    def __init__(self, engine: BaseEvaluatorEngine, nested_data: Dict[str, Any], max_k: int = 100):
        self.engine = engine
        self.max_k = max_k
        self.hierarchy = {}
        self._build_enhanced_hierarchy(nested_data)

    def _build_enhanced_hierarchy(self, nested_data: Dict[str, Any]):
        """Construit une structure enrichie pour les métriques sémantiques."""
        print("[*] Building enhanced hierarchy for semantic metrics...")
        
        def traverse(node_name: str, node_data: Dict[str, Any], ancestors: List[str]):
            path = ancestors + [node_name]
            children = node_data.get("children", {})
            child_names = list(children.keys()) if isinstance(children, dict) else []

            self.hierarchy[node_name] = {
                "path": path,
                "ancestors": set(ancestors),
                "depth": len(ancestors),
                "siblings_count": 0, # Sera mis à jour plus tard
                "is_leaf": len(child_names) == 0
            }

            for child_name, child_data in children.items():
                traverse(child_name, child_data, path)

        # Initialisation de la traversée
        for root_name, root_data in nested_data.items():
            traverse(root_name, root_data, [])

        # Deuxième passe pour compter les siblings (frères)
        for node, meta in self.hierarchy.items():
            # On cherche le parent dans la structure pour compter ses enfants
            # Pour simplifier ici, on suppose que l'on peut retrouver le parent via le path
            parent_path = meta["path"][:-1]
            if parent_path:
                parent_name = parent_path[-1]
                # On compte combien d'enfants ce parent a dans la structure globale
                # (Cette partie nécessite une structure de parentage inverse)
                pass 

    def run_evaluation(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        """Exécute l'évaluation en un seul passage (Single-Pass)."""
        all_rows = []
        print(f"[*] Starting evaluation on {len(queries_df)} queries (Max K={self.max_k})...")

        for _, row in queries_df.iterrows():
            query = row['query']
            expected = row['expected_class']
            
            # Appel unique au moteur pour le top_k maximal
            results = self.engine.get_top_k_results(query, self.max_k)
            
            for rank, retrieved in enumerate(results, 1):
                if rank > self.max_k: break
                
                # --- Calcul des Métriques Sémantiques (Low Weight) ---
                is_hit = (retrieved == expected)
                
                # 1. Hierarchical Distance & Depth Drift
                dist = self._get_hierarchical_distance(expected, retrieved)
                depth_drift = self._get_depth_drift(expected, retrieved)
                
                # 2. Specificity Score (1: Exact, 2: Ancestor, 3: Descendant, 4: Other)
                spec_score = self._get_specificity(expected, retrieved)
                
                # 3. Path Overlap (Jaccard similarity of paths)
                overlap = self._get_path_overlap(expected, retrieved)

                all_rows.append({
                    "query": query,
                    "expected_class": expected,
                    "rank": rank,
                    "retrieved_class": retrieved,
                    "is_hit": is_hit,
                    "hierarchical_distance": dist,
                    "depth_drift": depth_drift,
                    "specificity_score": spec_score,
                    "path_overlap": overlap
                })

        return pd.DataFrame(all_rows)

    def _get_hierarchical_distance(self, gold: str, pred: str) -> float:
        if gold not in self.hierarchy or pred not in self.hierarchy: return 99.0
        g_path = self.hierarchy[gold]["path"]
        p_path = self.hierarchy[pred]["path"]
        common = 0
        for a, b in zip(g_path, p_path):
            if a == b: common += 1
            else: break
        return float((len(g_path) - common) + (len(p_path) - common))

    def _get_depth_drift(self, gold: str, pred: str) -> float:
        if gold not in self.hierarchy or pred not in self.hierarchy: return 0.0
        return float(abs(self.hierarchy[gold]["depth"] - self.hierarchy[pred]["depth"]))

    def _get_specificity(self, gold: str, pred: str) -> int:
        if gold == pred: return 1
        if gold in self.hierarchy and pred in self.hierarchy:
            if gold in self.hierarchy[pred]["ancestors"]: return 2 # Over-generalization
            if pred in self.hierarchy[gold]["ancestors"]: return 3 # Over-specialization
        return 4

    def _get_path_overlap(self, gold: str, pred: str) -> float:
        if gold not in self.hierarchy or pred not in self.hierarchy: return 0.0
        set_g = set(self.hierarchy[gold]["path"])
        set_p = set(self.hierarchy[pred]["path"])
        intersection = len(set_g.intersection(set_p))
        union = len(set_g.union(set_p))
        return intersection / union if union > 0 else 0.0

# ============================================================
# 3. VISUALIZATION MODULE
# =================================================                
# ============================================================

class EvaluatorVisualizer:
    @staticmethod
    def plot_performance(df_flat: pd.DataFrame, output_path: Path):
        """Génère les graphiques de performance."""
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Precision & Recall vs K
        # On reconstruit le top-K par requête pour calculer les métriques cumulées
        k_values = sorted(df_flat['rank'].unique())
        precisions = []
        recalls = []
        
        # Pour le recall, on a besoin du total de queries originales
        total_queries = df_flat.groupby('query')['is_hit'].any().sum()

        for k in k_values:
            subset = df_flat[df_flat['rank'] <= k]
            # Precision@K = hits / (k * num_queries) -> No, it's hits / total_possible_slots
            # Correct way: precision is hits / number of predictions made at this K
            precision = subset['is_hit'].sum() / len(subset)
            # Recall@K = hits / total_queries
            recall = subset['is_hit'].sum() / total_queries
            precisions.append(precision)
            recalls.append(recall)

        axes[0].plot(k_values, precisions, label='Precision@K', color='blue', marker='o')
        axes[0].plot(k_values, recalls, label='Recall@K', color='green', marker='s')
        axes[0].set_title("Precision & Recall vs K")
        axes[0].set_xlabel("K")
        axes[0].legend()

        # 2. MRR (Mean Reciprocal Rank)
        mrr_values = []
        for k in k_values:
            subset = df_flat[df_flat['rank'] <= k]
            # Find first hit for each query
            first_hits = subset[subset['is_hit'] == True].groupby('query')['rank'].min()
            mrr = (1 / first_hits).mean() if not first_hits.empty else 0
            mrr_values.append(mrr)
        
        axes[1].plot(k_values, mrr_values, color='red', marker='d')
        axes[1].set_title("MRR vs K")
        axes[1].set_xlabel("K")

        # 3. Error Distribution (Hierarchical Distance)
        errors = df_flat[df_flat['is_hit'] == False]
        if not errors.empty:
            sns.histplot(errors['hierarchical_distance'], bins=20, ax=axes[2], kde=True, color='purple')
            axes[2].set_title("Error Proximity (Hierarchical Distance)")
            axes[2].set_xlabel("Distance")
        else:
            axes[2].text(0.5, 0.5, "No errors found", ha='center')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"[+] Visualizations saved to {output_path}")