import pytest
import json
import os
from pathlib import Path
from biomedcat.stages.rag_engine import build_rag

# Chemin vers les données de test
QA_DATA_PATH = Path(__file__).parent / "qa_data.json"

class QAMetrics:
    """Classe pour calculer et stocker les métriques de performance du RAG."""
    def __run_metrics(self, hits: int, total: int, k: int, ranks: list[int]):
        recall = hits / total if total > 0 else 0
        precision = hits / k if total > 0 else 0
        mrr = sum(1.0 / r for r in ranks) / total if total > 0 else 0
        return {
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
            "mrr": round(mrr, 4),
            "hits": hits,
            "total_queries": total
        }

@pytest.fixture(scope="module")
def rag_engine():
    """Initialise le moteur RAG pour les tests."""
    return build_rag()

@pytest.fixture(scope="module")
def test_cases():
    """Charge les cas de test depuis le fichier JSON."""
    with open(QA_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_rag_retrieval_performance(rag_engine, test_cases, top_k):
    """
    Test scientifique de la capacité de récupération des classes Biolink.
    Mesure le Recall@K, la Precision@K et le MRR (Mean Reciprocal Rank).
    """
    hits = 0
    ranks = []
    total = len(test_mask := test_cases)

    for case in test_cases:
        query = case["query"]
        expected = case["expected_class"]
        
        # Exécution de la recherche avec le paramètre top_k actuel
        results = rag_engine.search(query, top_k=top_k)
        
        if expected in results:
            hits += 1
            # Calcul du rang (index + 1) pour le MRR
            rank = results.index(expected) + 1
            ranks.append(rank)

    # Calcul des métriques finales
    metrics_calculator = QAMetrics()
    stats = metrics_corps = metrics_calculator._run_metrics(hits, total, top_k, ranks)

    # Affichage des métadonnées de test dans la console pytest
    print(f"\n--- QA Test Metadata (top_k={top_k}) ---")
    print(json.dumps(stats, indent=2))
    print("----------------------------------------")

    # Assertions pour valider que le système n'est pas totalement défaillant
    # On exige au moins un succès pour valider la suite de test
    assert hits > 0, f"Échec critique : Aucune classe correcte trouvée pour top_k={top_k}"

if __name__ == "__main__":
    # Permet de lancer le script directement pour voir les résultats
    pytest.main([__file__, "-s"])
