import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from biomedcat.stages.rag_engine import build_rag
from biomedcat.config import Settings


# ============================================================
# Configuration
# ============================================================

QA_DATA_PATH = Path("data/qa_dataset.parquet")
NESTED_DATA_PATH = Path("data/biolink_classes_nested.json")
OUTPUT_DIR = Path("data/rag_analysis")
K_RANGE = list(range(1, 100)) # Evaluate retrieval at K = 1 ... 99
# Structural ontology nodes that are not evaluated:
EXCLUDED_CLASSES = {"entity","named thing"}
RANDOM_SEED = 77

class RAGAnalyzer:

    def __init__(self,rag_engine,nested_data: Dict[str, Any]):

        self.rag_engine = rag_engine
        self.nested_data = nested_data
        self.class_metadata = {}

        self.parent_map = {}
        self.ancestor_map = {}
        self.path_map = {}

        self._build_hierarchy()

    # Hierarchy
    def _build_hierarchy(self):
        """
        Reconstruct the hierarchy from the recursively nested
        Biolink JSON structure.
        """

        print("[*] Building hierarchy...")
        visited = set()
        def traverse(node_name: str,node_data: Dict[str, Any],depth: int,ancestors: List[str]): 

            if node_name in visited:
                return

            visited.add(node_name)
            children = node_data.get("children", {})

            if isinstance(children, dict):
                child_names = list(children.keys())
            else:     
                child_names = []

            path = ancestors + [node_name]

            self.class_metadata[node_name] = {
                "depth": depth, 
                "is_leaf": (len(child_names) == 0), 
                "ancestors": ancestors.copy(), 
                "path": path, 
                "children": child_names.copy()}

            self.ancestor_map[node_name] = ancestors.copy()
            self.path_map[node_name] = path.copy()
            if ancestors:     
                self.parent_map[node_name] = ancestors[-1]

            for child_name, child_data in (children.items()):     
                traverse(node_name=child_name,node_data=child_data,depth=depth + 1,ancestors=path)

        # Traverse from top-level JSON nodes:
        for root_name, root_data in (self.nested_data.items()): 
            traverse(node_name=root_name, node_data=root_data, depth=0, ancestors=[])

        print(f"[*] Total ontology nodes: {len(self.class_metadata)}")
        sampleable = [cls for cls in self.class_metadata if cls not in EXCLUDED_CLASSES]
        print(f"[*] Sampleable classes: {len(sampleable)}")

        # Depth distribution
        depth_counts = defaultdict(int)
        for cls in sampleable: 
            depth = self.class_metadata[cls]["depth"]
            depth_counts[depth] += 1
        print("[*] Classes by depth:")

        for depth in sorted(depth_counts): 
            print(f"Depth {depth}: {depth_counts[depth]}")

    # Load dataset:
    def load_qa_data(self) -> pd.DataFrame:
        if not QA_DATA_PATH.exists(): 
            print(f"[!] Dataset not found: {QA_DATA_PATH}")
            return pd.DataFrame()

        print(f"[*] Loading dataset: {QA_DATA_PATH}")
        df = pd.read_parquet(QA_DATA_PATH)

        required_columns = {"query","expected_class"}

        missing = (required_columns - set(df.columns))

        if missing: 
            raise ValueError(f"Dataset missing required columns: {missing}")

        print(f"[*] Loaded {len(df)} samples.")
        print(f"[*] Classes: {df['expected_class'].nunique()}")

        return df

    # Hierarchy utilities
    def get_ancestors(self,class_name: str) -> List[str]:
        return self.ancestor_map.get(class_name,[])

    def get_depth(self,class_name: str) -> Optional[int]:

        metadata = self.class_metadata.get(class_name)
        if metadata is None: 
            return None

        return metadata["depth"]

    def get_path(self,class_name: str) -> List[str]:
        return self.path_map.get(class_name,[class_name],)

    # Hierarchical distance
    def hierarchical_distance(self, gold: str, predicted: str) -> Optional[int]:
        """
        Compute the number of edges between two ontology
        classes.

        Example: 
        Small molecule
              |
          Chemical Entity
              |
           Named Thing

        distance(Small molecule, Chemical Entity) = 1
        distance(Small molecule, Named Thing) = 2

        Returns None if either class is not known.
        """

        if (gold not in self.class_metadata
            or predicted not in self.class_metadata): return None

        gold_path = self.get_path(gold)
        pred_path = self.get_path(predicted)
        
        # Find lowest common ancestor:
        common_length = 0
        for g, p in zip(gold_path,pred_path): 
            if g != p:
                break
            common_length += 1

        # Number of edges from each node to LCA.
        gold_distance = (len(gold_path)- common_length)
        pred_distance = (len(pred_path)- common_length)

        return (gold_distance + pred_distance)

    # Is ancestor
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:

        if ancestor == descendant: 
            return True

        return (ancestor in self.get_ancestors(descendant))

    # Search normalization
    def _normalize_results(self, results) -> List[str]:
        """
        Normalize the output of rag_engine.search().

        The existing analyzer assumed that search() returns
        a list of class names. This function makes the
        assumption explicit and handles a few common formats.
        """

        if results is None: return []

        # Dictionary result:

        if isinstance(results,dict): return list(results.keys())

        # List / tuple / set:

        if isinstance(results,(list, tuple, set)): 
            normalized = []

            for item in results:     
                # Plain class name.
                if isinstance(item,str):         
                     normalized.append(item)

                # Objects represented as dicts.
                elif isinstance(item,dict):         
                    for key in ("class","class_name","name","label"):             
                        if key in item:
                            normalized.append(item[key])
                            break

            return normalized

        return []


    # Single query evaluation:
    def evaluate_query(self, query: str, expected: str, k: int) -> Dict[str, Any]:
        """
        Evaluate one query at one K.
        """

        raw_results = (self.rag_engine.search(query, top_k=k))

        results = self._normalize_results(raw_results)

        # Remove duplicates while preserving ranking.
        ranked_results = list(dict.fromkeys(results))

        # Exact rank:
        exact_rank = None
        for rank, cls in enumerate(ranked_results,start=1): 
            if cls == expected:     
                exact_rank = rank
                break

        exact_hit = (exact_rank is not None
            and exact_rank <= k)

        # Ancestor retrieval:
        ancestors = self.get_ancestors(expected)
        retrieved_ancestors = [cls for cls in ranked_results if cls in ancestors]
        ancestor_hit = (len(retrieved_ancestors) > 0)

        # Best hierarchical distance:
        # Among all retrieved classes, find the one that is
        # closest to the gold class:
        distances = []
        for rank, predicted in enumerate(ranked_results,start=1): 
            distance = (self.hierarchical_distance(expected,predicted))
            if distance is not None:     
                distances.append((distance,rank,predicted))

        if distances: 
            best_distance, best_distance_rank, closest_class = min(distances, key=lambda x: (x[0],x[1]))
        else: 
            best_distance = None
            best_distance_rank = None
            closest_class = None

        # Is the closest prediction an ancestor?
        closest_is_ancestor = False

        if closest_class is not None: 
            closest_is_ancestor = (closest_class in ancestors)

        return {
            "query": query,
            "expected_class": expected,
            "k": k,
            "results": ranked_results[:k],
            "exact_hit": exact_hit,
            "exact_rank": exact_rank,
            "ancestor_hit": ancestor_hit,
            "retrieved_ancestors": (retrieved_ancestors),
            "best_hierarchical_distance": (best_distance),
            "closest_class": closest_class,
            "closest_class_rank": (best_distance_rank),
            "closest_is_ancestor": (closest_is_ancestor),
            "gold_depth": self.get_depth(expected)
        }

    # Main analysis
    def analyze(self):
        df = self.load_qa_data()
        if df.empty: 
            print("[!] No test cases to analyze.")
            return

        # Validate classes:
        unknown_classes = sorted(set(df["expected_class"]) - set(self.class_metadata.keys()))
        if unknown_classes: 
            print(f"[!] WARNING: {len(unknown_classes)} dataset classes are not present in the ontology.")
            for cls in unknown_classes:     
                print(f"{cls}")

        # Dataset distribution:
        print("\n[*] Dataset distribution:")
        class_counts = (df["expected_class"].value_counts().sort_index())
        print(f"    Min/class: {class_counts.min()}")
        print(f"    Max/class: {class_counts.max()}")
        print(f"    Mean/class: {class_counts.mean():.2f}")
        print(f"    Median/class: {class_counts.median():.2f}")

        # Storage for results:
        all_results = {}
        class_results = []
        example_results = []
        print(f"\n[*] Starting analysis across K={K_RANGE[0]}...{K_RANGE[-1]}...")
        
        # Evaluate each K:
        for k in K_RANGE: 

            print(f"  [>] Evaluating top_k={k}")
            per_class = defaultdict(lambda: {"hits": 0,"total": 0,"ancestor_hits": 0,"distances": []})

            exact_hits = 0
            ancestor_hits = 0
            distances = []

            for row in df.itertuples(index=False):     
                query = row.query
                expected = (row.expected_class)
                result = (self.evaluate_query(query=query,expected=expected,k=k))

                # Global metrics:
                exact_hits += int(result["exact_hit"])
                ancestor_hits += int(result["ancestor_hit"])

                if (result["best_hierarchical_distance"] is not None):
                    distances.append(result["best_hierarchical_distance"])

                # Per-class metrics:
                stats = per_class[expected]
                stats["total"] += 1
                stats["hits"] += int(result["exact_hit"])
                stats["ancestor_hits"] += int(result["ancestor_hit"])

                if (result["best_hierarchical_distance"] is not None):
                    stats["distances"].append(result["best_hierarchical_distance"])
            
                example_results.append(result) # save example-level result

            # Calculate per-class metrics:
            recalls = []
            ancestor_recalls = []

            for cls, stats in (per_class.items()):     
                recall = (stats["hits"] / stats["total"] if stats["total"] else 0.0 )
                ancestor_recall = (stats["ancestor_hits"] / stats["total"] if stats["total"] else 0.0)

                mean_distance = (np.mean(stats["distances"]) if stats["distances"] else np.nan)
                depth = self.get_depth(cls)

                class_results.append({
                        "k": k,
                        "class": cls,
                        "depth": depth,
                        "n": stats["total"],
                        "hits": stats["hits"],
                        "recall": recall,
                        "ancestor_hits": (stats["ancestor_hits"]),
                        "ancestor_recall": (ancestor_recall),
                        "mean_hierarchical_distance": (mean_distance)}
                )
                recalls.append(recall)
                ancestor_recalls.append(ancestor_recall)

            # Overall metrics
            total = len(df)
            micro_recall = (exact_hits / total if total else 0.0)
            macro_recall = (np.mean(recalls) if recalls else 0.0)
            macro_ancestor_recall = (np.mean(ancestor_recalls) if ancestor_recalls else 0.0)
            mean_distance = (np.mean(distances) if distances else np.nan)

            all_results[k] = {
                "micro_recall": (micro_recall), 
                "macro_recall": (macro_recall), 
                "ancestor_recall": (ancestor_hits / total if total else 0.0), 
                "macro_ancestor_recall": (macro_ancestor_recall), 
                "mean_hierarchical_distance": (mean_distance)}

            print(f"Micro Recall: {micro_recall:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Ancestor Recall: {macro_ancestor_recall:.4f}")

            if not np.isnan(mean_distance):     print(f"Mean Hierarchical "
                    f"Distance: "
                    f"{mean_distance:.4f}"
                )

        # Reports
        class_df = pd.DataFrame(class_results)
        example_df = pd.DataFrame(example_results)
        self._print_report(all_results)
        self._depth_analysis(class_df)
        self._class_analysis(class_df)
        self._save_results(class_df=class_df,example_df=example_df,overall_results=all_results,)

    # Overall report:
    def _print_report(self, all_results: Dict[int, Any]):
        print("\n" + "=" * 70)
        print("RAG HIERARCHICAL PERFORMANCE REPORT")
        print("=" * 70)
        print("\n K    Micro Recall    Macro Recall    Ancestor Recall    Mean Hier. Dist.")
        print("-" * 70)

        for k, metrics in (all_results.items()): 
            distance = metrics["mean_hierarchical_distance"]
            distance_text = (f"{distance:.4f}" if not np.isnan(distance) else "N/A")
            print(f"{k:2d}\n{metrics['micro_recall']:.4f}\n{metrics['macro_recall']:.4f}\n{metrics['macro_ancestor_recall']:.4f}\n{distance_text}")

        # Best K by macro recall
        best_k = max(all_results,key=lambda k: all_results[k]["macro_recall"])
        best_metrics = (all_results[best_k])

        print("\n"  + "=" * 70)
        print(f"BEST K BY MACRO RECALL: {best_k}")
        print(f"Macro Recall: {best_metrics['macro_recall']:.4f}")
        print(f"Micro Recall: {best_metrics['micro_recall']:.4f}")
        print(f"Macro Ancestor Recall: {best_metrics['macro_ancestor_recall']:.4f}")
        print(f"Mean Hierarchical Distance: {best_metrics['mean_hierarchical_distance']:.4f}")
        print("=" * 70)

    # Depth analysis:
    def _depth_analysis(self,class_df: pd.DataFrame):

        print("\n" + "-" * 70)
        print("PERFORMANCE BY HIERARCHY DEPTH")
        print("-" * 70)

        # Find best K by macro recall:
        macro_by_k = (class_df.groupby("k")["recall"].mean())
        best_k = (macro_by_k.idxmax())
        best = class_df[class_df["k"] == best_k]
        depth_summary = (    best.groupby("depth").agg(classes=("class","count"), 
            mean_recall=("recall","mean"), 
            std_recall=("recall","std"), 
            mean_ancestor_recall=("ancestor_recall","mean"), 
            mean_hierarchical_distance=("mean_hierarchical_distance","mean")).reset_index())

        print(depth_summary.to_string(index=False))


    # Class analysis:
    def _class_analysis(self, class_df: pd.DataFrame):
        # Best K:
        macro_by_k = (class_df.groupby("k")["recall"].mean())
        best_k = (macro_by_k.idxmax())
        best = class_df[class_df["k"] == best_k].copy()

        # Worst classes:
        worst = (best.sort_values("recall", ascending=True).head(15))
        print("\n" + "-" * 70)
        print(f"WORST CLASSES AT K={best_k}")
        print("-" * 70)
        print(worst[["class","depth","n","recall","ancestor_recall","mean_hierarchical_distance"]].to_string(index=False))

        # Best classes:
        best_classes = (best.sort_values("recall", ascending=False).head(15))
        print("\n" + "-" * 70)
        print(f"BEST CLASSES AT K={best_k}")
        print("-" * 70)
        print(best_classes[["class","depth","n","recall","ancestor_recall","mean_hierarchical_distance"]].to_string(index=False))

    # Save results
    def _save_results(self,class_df: pd.DataFrame, example_df: pd.DataFrame, overall_results: Dict[int, Any]):

        OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
        # Class-level metrics:
        class_path = (OUTPUT_DIR / "class_metrics.parquet")
        class_df.to_parquet(class_path,index=False,)

        # Example-level metrics:
        example_path = (OUTPUT_DIR / "example_metrics.parquet")

        # serialize results:
        if "results" in example_df.columns: 
            example_df["results"] = example_df["results"].apply(lambda x: json.dumps(x,ensure_ascii=False))

        if ("retrieved_ancestors" in example_df.columns): 
            example_df["retrieved_ancestors"] = example_df["retrieved_ancestors"].apply(lambda x: json.dumps(x,ensure_ascii=False))

        example_df.to_parquet(example_path,index=False)

        # Overall metrics:
        overall_rows = []
        for k, metrics in (overall_results.items()): 
            overall_rows.append({"k": k,**metrics})

        overall_df = pd.DataFrame(overall_rows)
        overall_path = (OUTPUT_DIR / "overall_metrics.csv")
        overall_df.to_csv(overall_path,index=False,)

        print("\n" + "=" * 70)
        print("RESULTS SAVED")
        print("=" * 70)
        print(f"Class metrics: {class_path}")
        print(f"Example metrics: {example_path}")
        print(f"Overall metrics: {overall_path}")

# Main
if __name__ == "__main__":

    settings = Settings()

    try:

        print("[*] Building RAG engine...")
        engine = build_rag(settings)

        print("[*] Loading ontology...")
        with open(NESTED_DATA_PATH,"r",encoding="utf-8",) as f: 
            nested_structure = json.load(f)

        analyzer = RAGAnalyzer(rag_engine=engine,nested_data=nested_structure,)
        analyzer.analyze()

    except Exception as e:
        print(f"[!] Analysis failed: {e}")
        import traceback
        traceback.print_exc()