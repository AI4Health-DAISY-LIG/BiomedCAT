import pandas as pd
import json
from pathlib import Path
from biomedcat.config import Settings
from biomedcat.stages.rag_engine import build_rag
# ner_agent should be acceswsible via PYTHONPATH



try:
    from biomedcat.stages.ner_agent import NERAgentPipeline
except ImportError:
    NERAgentPipeline = None

from evaluator_framework import BiomedRAGEngineWrapper, NERAgentEngineWrapper, RAGAnalyzer, EvaluatorVisualizer

def main():
    settings = Settings()
    output_dir = Path("data/eval_results")
    output_impl = output_dir / "experiment_results.parquet"
    output_plot = output_dir / "performance_report.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement de l'ontologie (pour les métriques sémantiques)
    nested_data_path = Path("data/biolink_classes_nested.json")
    with open(nested_data_path, "r", encoding="utf-8") as f:
        ontology_structure = json.load(f)

    # 2. Chargement du Dataset de Test (Queries)
    # On suppose un fichier parquet avec 'query' et 'expected_class'
    test_data_path = Path("data/qa_dataset.parquet")
    if not test_data_path.exists():
        print(f"[!] Test dataset not found at {test_data_path}. Please run generator first.")
        return
    queries_df = pd.read_parquet(test_data_path)

    # 3. Choix du moteur (MODE: RAG ou AGENT)
    # Changez 'RAG' par 'AGENT' pour comparer les performances
    MODE = "RAG" 
    print(f"\n=== STARTING EVALUATION MODE: {MODE} ===")

    if MODE == "RAG":
        rag_engine = build_rag(settings)
        engine_wrapper = BiomedRAGEngineWrapper(rag_engine)
    elif MODE == "AGENT" and NERAgentPipeline is not None:
        # Note: Nécessite une implémentation de build_rag pour l'agent ou configuration manuelle
        from biomedcat.stages.rag_engine import build_rag
        rag = build_rag(settings)
        # On simule le besoin de RAG pour l'agent
        from biomedcat.stages.rag_engine import BiomedRAG 
        # (Ici on devrait adapter selon votre structure réelle d'initialisation de l'agent)
        # Pour l'exemple, on suppose que l'agent est initialisé ainsi :
        agent = NERAgentPipeline(rag_engine=rag, extraction_model_id=settings.extraction_model_id, classification_model_id=settings.classification_model_id)
        engine_wrapper = NERAgentEngineWrapper(agent)
    else:
        raise ValueError("Invalid MODE or Agent class not found.")

    # 4. Exécution de l'Analyseur
    analyzer = RAGAnalyzer(engine_wrapper, ontology_structure, max_k=50)
    results_df = analyzer.run_evaluation(queries_df)

    # 5. Sauvegarde des résultats bruts (Format Long/Flat)
    results_df.to_parquet(output_impl)
    print(f"[+] Raw results saved to {output_impl}")

    # 6. Visualisation
    visualizer = EvaluatorVisualizer()
    visualizer.plot_performance(results_df, output_plot)

    # 7. Affichage de résumé rapide en console
    print("\n=== FINAL SUMMARY ===")
    precision_avg = results_df[results_df['rank'] == 1]['is_hit'].mean()
    print(f"Precision @ Rank 1: {precision_avg:.4f}")
    print(f"Total Predictions Analyzed: {len(results_df)}")
    print("======================\n")

if __name__ == "__main__":
    main()