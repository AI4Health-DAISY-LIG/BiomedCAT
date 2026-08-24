import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from biomedcat.stages.rag_engine import build_rag
from biomedcat.config import Settings

# Configuration des chemins
QA_DATA_PATH = Path("tests/qa_</strong>data.json")
NESTED_DATA_PATH = Path("data/biolink_classes_nested.json")
K_RANGE = [1, 5, 10, 20, 50, 100]

class RAGAnalyzer:
    def __init__(self, rag_engine, nested_data: Dict[str, Any]):
        self.rag_engine = rag_engine
        self.nested_data = nested_data
        # On prépare une map inverse pour retrouver les parents rapidement
        self.parent_map = self._build_parent_map(self.nested_data)

    def _build_parent_map(self, nested: Dict[str, Any]) -> Dict[str, str]:
        """Builds a mapping from child class to its parent class."""
        parents = {}
        def traverse(node_name, parent_name=None):
            if parent_name:
                parents[node_name] = parent_name
            if node_name in nested:
                children = nested[node_name].get("children", {})
                for child_name in children:
                    traverse(child_name, node_name)
        
        # Start traversal from roots (classes without a defined parent in the JSON)
        roots = [name for name, data in nested.items() if not data.get("parent")]
        for root in roots:
            traverse(root, None)
        return parents

    def load_qa_data(self) -> List[Dict[str, str]]:
        """Loads the QA test cases from the JSON file."""
        if not QA_DATA_PATH.exists():
            print(f"[!] Warning: {QA_DATA_PATH} not found.")
            return []
        with open(QA_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze(self):
        """Performs the performance analysis across different top_k values."""
        test_cases = self.load_qa_data()
        if not test_cases:
            print("[!] No test cases to analyze.")
            return

        all_results = {}

        print(f"[*] Starting analysis across K={K_RANGE}...")

        for k in K_RANGE:
            print(f"  [>] Evaluating top_k={k}")
            metrics_per_class = defaultdict(lambda: {"hits": 0, "total": 0})
            failures = []

            for case in test_cases:
                query = case["query"]
                expected = case["expected_class"]
                
                # Increment total for this class
                metrics_per_class[expected]["total"] += 1
                
                results = self.rag_engine.search(query, top_k=k)
                
                if expected in results:
                    metrics_per_class[expected]["hits"] += 1
                else:
                    # Record failure for proximity analysis
                    failures.append(expected)

            # Calculate recall per class for this K
            recall_at_k = {}
            for cls, counts in metrics_per_class.items():
                recall_at_k[cls] = counts["hits"] / counts["total"] if counts["total"] > 0 else 0
            
            all_results[k] = {
                "recall_per_class": recall_at_k,
                "failures": failures
            }

        self._print_report(all_results)

    def _print_report(self, all_results: Dict[int, Any]):
        """Prints the final performance and failure mode report."""
        print("\n" + "="*50)
        print("       RAG PERFORMANCE & HHIERARCHY REPORT")
        print("="*50)

        # 1. Find optimal K (the one that maximizes average global recall)
        best_k = 0
        max_avg_recall = -1.0

        for k, data in all_results.items():
            recalls = list(data["recall_per_class"].values())
            avg_recall = np.mean(recalls) if recalls else 0
            print(f"Top_K={k:3} | Avg Recall: {avg_recall:.4f} | Failures: {len(data['failures'])}")
            
            if avg_recall > max_avg_recall:
                max_avg_recall = avg_recall
                best_k = k

        print(f"\n[!] OPTIMAL TOP_K IDENTIFIED: {best_k}")

        # 2. Failure Mode Analysis (Proximity to other terms)
        print("\n" + "-"*50)
        print("FAILURE MODE ANALYSIS (Hierarchy Proximity)")
        print("-" * 50)

        # We analyze failures at the best K found
        final_failures = all_results[best_k]["failures"]
        
        if not final_failures:
            print("No failures detected at optimal K.")
            return

        failure_clusters = defaultdict(int)
        for f_class in final_failures:
            # Trace back up the tree to see which ancestors are impacted by these failures
            path = []
            curr = f_class
            while curr in self.parent_map:
                curr = self.parent_map[curr]
                path.append(curr)
            
            for ancestor in path:
                failure_clusters[ancestor] += 1

        # Sort clusters by impact (number of impacted descendants)
        sorted_clusters = sorted(failure_clusters.items(), key=lambda x: x[1], reverse=True)

        print("Detected failure clusters in hierarchy (Class -> Number of impacted descendants):")
        for cluster_class, count in sorted_clusters[:5]: # Top 5 clusters
            status = "CRITICAL" if count > len(final_failures)/2 else "MODERATE"
            print(f"  [{status}] {cluster_class}: {count} failures propagating through this branch")

        print("\n[!] Analysis Complete.")

if __name__ == "__main__":
    # Initialization of the RAG engine
    settings = Settings()
    try:
        engine = build_rag(settings)
        
        # Load the hierarchical structure
        with open(NESTED_DATA_PATH, "r", encoding="utf-8") as f:
            nested_structure = json.load(f)

        analyzer = RAGAnalyzer(engine, nested_structure)
        analyzer.analyze()
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
