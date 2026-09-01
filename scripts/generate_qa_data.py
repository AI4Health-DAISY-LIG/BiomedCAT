import json
import random
import yaml
import ollama
import pandas as pd

from pathlib import Path
from typing import List, Dict, Any, Tuple

from biomedcat.stages.rag_engine import build_rag
from biomedcat.config import Settings


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path("scripts/ner_test-suite_configuration.yml")
NESTED_DATA_PATH = Path("data/biolink_classes_nested.json")
OUTPUT_PARQUET = Path("data/qa_dataset.parquet")

MODEL_NAME = "gemma4:e4b-it-qat"

###################################################################################
# ------------------------------------------------------------
# Dataset design
# ------------------------------------------------------------

TARGET_PER_CLASS = 30

BATCH_SIZE = 25
MAX_ATTEMPTS_PER_CLASS = 6

RANDOM_SEED = 77

# These are structural ontology nodes, NOT entity classes
# that should receive generated examples.
EXCLUDED_CLASSES = {"entity","named thing"}



class StratifiedQAGenerator:

    def __init__(self,rag_engine,nested_data: Dict[str, Any],config: Dict[str, Any]):
        self.rag_engine = rag_engine
        self.nested_data = nested_data
        self.config = config

        self.all_samples: List[Dict[str, Any]] = []

        self.class_metadata: Dict[str, Dict[str, Any]] = {}

        # Parent → children representation.
        self.children_map: Dict[str, List[str]] = {}

        # Reproducibility.
        self.random = random.Random(RANDOM_SEED)

        self._analyze_hierarchy()

    # Hierarchy
    def _analyze_hierarchy(self):
        """
        Recursively analyze the nested ontology.

        The JSON structure is:

            entity
            └── named thing
                ├── class A
                │   ├── class A1
                │   └── class A2
                └── class B

        There are NO explicit parent fields.

        Parent/ancestor information is reconstructed from the
        nested `children` dictionaries.
        """

        # Recursive traversal:
        print("[*] Analyzing hierarchy...")
        self.class_metadata = {}
        def traverse(node_name: str,node_data: Dict[str, Any],depth: int,ancestors: List[str]):
            children = node_data.get("children",{})

            if isinstance(children, dict):
                child_items = children.items()
            else:
                child_items = []

            child_names = [child_name for child_name, _ in child_items]
            is_leaf = len(child_names) == 0

            path = ancestors + [node_name]

            # Store metadata:
            self.class_metadata[node_name] = {
                "depth": depth,
                "is_leaf": is_leaf,
                "ancestors": ancestors.copy(),
                "path": path,
                "children": child_names.copy(),
            }

            # Recurse into children:
            for child_name, child_data in (node_data.get("children",{}).items()):
                traverse(node_name=child_name,node_data=child_data,depth=depth + 1,ancestors=path)

        # Start from the actual JSON wrapper:
        root_name = "entity"
        if root_name not in self.nested_data:
            raise ValueError(f"Expected top-level '{root_name}' node, but found: {list(self.nested_data.keys())}")

        traverse(node_name=root_name,node_data=self.nested_data[root_name],depth=0,ancestors=[])

        # Report before exclusion:
        print(f"[*] Total nodes including wrappers: {len(self.class_metadata)}")


        # Remove structural nodes from sampling metadata:
        for excluded in EXCLUDED_CLASSES:
            if excluded in self.class_metadata:
                print(f"[*] Excluding structural class: {excluded}")
                del self.class_metadata[excluded]

        # Count actual entity classes:
        sampleable_classes = [name for name in self.class_metadata]
        print(f"[*] Sampleable entity classes: {len(sampleable_classes)}")

        # Leaf statistics:
        leaf_classes = [name for name, metadata in self.class_metadata.items() if metadata["is_leaf"]]
        print(f"[*] Leaf classes: {len(leaf_classes)}")

        # Depth statistics:
        depth_counts = {}
        for name, metadata in (self.class_metadata.items()):
            depth = metadata["depth"]
            depth_counts[depth] = (depth_counts.get(depth, 0) + 1)

        print("[*] Classes by depth:")
        for depth in sorted(depth_counts):
            print(f"depth {depth}: {depth_counts[depth]}")

    # Configuration helpers:
    def _get_sampling_config(self) -> Dict[str, Any]:
        """
        Read the sampling configuration.

        The YAML may contain:

        sampling:
            total_target_samples: 20000
            min_per_leaf: 100
            priority_target_per_leaf: 200
            priority_classes: []
            priority_depths: []
            sample_leaves_only: true
        """

        return self.config.get("sampling",{})

    # Sampling plan:
    def _get_sampling_plan(self):

        sampling_cfg = self.config.get("sampling",{})
        target_per_class = int(sampling_cfg.get("target_per_class",TARGET_PER_CLASS))
        sample_leaves_only = sampling_cfg.get("sample_leaves_only",True)

        classes = []
        for class_name, metadata in (self.class_metadata.items()):
            if (sample_leaves_only and not metadata["is_leaf"]):
                continue
            classes.append((class_name,target_per_class))
        classes.sort(key=lambda x: x[0])

        return classes

    # Definition lookup:
    def _get_definition(self,class_name: str) -> str:
        if class_name not in self.rag_engine.flat_data:
            return ""

        metadata = self.rag_engine.flat_data[class_name].get("metadata",{})
        return (metadata.get("definition","") or "")

    # LLM generation:
    def _query_ollama(self,class_name: str,definition: str,n: int) -> List[Dict[str, str]]:

        metadata = self.class_metadata[class_name]
        ancestors = metadata["ancestors"]
        parent_context = (ancestors[-1] if ancestors else "None")

        prompt = f"""
                    You are a biomedical ontology expert generating high-quality evaluation data.
                    Target entity class:
                    {class_name}
                    Definition:
                    {definition}
                    Immediate parent class:
                    {parent_context}

                    Generate exactly {n} DISTINCT biomedical entity mentions
                    that are valid instances of the target class.

                    Requirements:
                    - Each query must be 1-3 words.
                    - Use realistic biomedical terminology.
                    - Every query must denote an entity that belongs to
                    "{class_name}".
                    - Prefer concrete, recognizable biomedical entities.
                    - Do NOT simply repeat the class name.
                    - Do NOT generate synonyms repeatedly.
                    - Do NOT generate multiple spelling variants of the same entity.
                    - Do NOT generate examples belonging primarily to a
                    child/subclass of "{class_name}".
                    - Do NOT generate examples that are only valid for a broader
                    parent class.
                    - Avoid generic or highly ambiguous biomedical words.
                    - Avoid duplicate entities.
                    - Vary terminology and entity forms.
                    - The examples should be useful for evaluating fine-grained
                    hierarchical entity recognition.

                    Return ONLY a JSON array.
                    """

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "expected_class": {
                        "type": "string"
                    },
                },
                "required": [
                    "query",
                    "expected_class",
                ],
            },
        }

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                format=schema,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response["message"]["content"]
            data = json.loads(content)
            if not isinstance(data, list):
                return []
            
            valid = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                query = (item.get("query","").strip())

                if not query:
                    continue

                # Basic whitespace normalization:
                query = " ".join(query.split())

                # Enforce requested length.
                word_count = len(query.split())

                if word_count < 1 or word_count > 3:
                    continue

                valid.append({"query": query,"expected_class": class_name}
                             )
            return valid

        except Exception as e:
            print(f"  [!] Generation error for {class_name}: {e}")
            return []

    # Generate one class
    def _generate_for_class(self,class_name: str,target: int) -> List[Dict[str, Any]]:

        definition = self._get_definition(class_name)

        if not definition:
            print(f"[!] No definition for {class_name}")
            return []

        metadata = self.class_metadata[class_name]
        collected: List[Dict[str, Any]] = []
        seen = set()
        attempts = 0
        while (len(collected) < target and attempts < MAX_ATTEMPTS_PER_CLASS):
            attempts += 1
            remaining = (target - len(collected))
            request_n = min(BATCH_SIZE,remaining)
            generated = self._query_ollama(class_name=class_name,definition=definition,n=request_n)

            for sample in generated:
                query = sample["query"].strip()
                key = query.casefold()
                if key in seen:
                    continue
                seen.add(key)

                collected.append(    {
                        "query": query,
                        "expected_class": class_name,
                        # Hierarchy metadata:
                        "depth": metadata["depth"],
                        "is_leaf": metadata["is_leaf"],
                        "ancestors": metadata["ancestors"],
                        "path": metadata["path"],
                        "parent_class": (
                            metadata["ancestors"][-1]
                            if metadata["ancestors"]
                            else None
                        ),
                        "generation_attempt": attempts
                    }
                )
                if len(collected) >= target:
                    break

            print(f" {len(collected)} / {target} (attempt {attempts} / {MAX_ATTEMPTS_PER_CLASS})"
            )
        if len(collected) < target:
            print(f"[!] Only generated {len(collected)} / {target} for {class_name}")

        return collected

    # Final dataset validation
    def _validate_dataset(self,df: pd.DataFrame):

        print("\n[*] Dataset validation")
        print(f"Total samples: {len(df)}")
        print(f"Unique classes: {df['expected_class'].nunique()}")

        sampling_cfg = (self._get_sampling_config())
        minimum = int(sampling_cfg.get("min_per_leaf"))
        counts = (df["expected_class"].value_counts())

        # missing:
        missing = []
        for class_name, metadata in (self.class_metadata.items()):
            if not metadata["is_leaf"]:
                continue
            count = int(counts.get(class_name,0,))
            if count < minimum:
                missing.append( (class_name, count,))
        if missing:
            print(f"    [!] {len(missing)} leaf classes below minimum {minimum}")
            for class_name, count in missing:
                print(f" {class_name}: {count}/{minimum}")
        else:
            print(f"[+] All leaf classes meet minimum {minimum}")

        # Depth distribution.
        print("\nSamples by depth:")
        depth_counts = (df["depth"].value_counts().sort_index())
        for depth, count in depth_counts.items():
            print(f"depth {depth}: {count}")

    # Main execution
    def run(self):

        sampling_plan = (self._get_sampling_plan())
        print(f"\n[*] Leaf classes selected: {len(sampling_plan)}")
        print(f"[*] Planned total samples: {sum(target for _, target in sampling_plan)}")

        # Generate:
        for i, (class_name,target) in enumerate(sampling_plan,start=1):
            metadata = self.class_metadata[class_name]
            print(f"\n[{i}/{len(sampling_plan)}] Generating: {class_name}")
            print(f"Depth: {metadata['depth']}")
            print(f"Target: {target}")
            samples = (self._generate_for_class(class_name=class_name,target=target))
            self.all_samples.extend(samples)

        if not self.all_samples:
            print("[!] No samples generated.")
            return

        # Final dataframe
        df = pd.DataFrame(self.all_samples)

        # ----------------------------------------------------
        # Global deduplication.
        #
        # A biomedical mention should ideally correspond to
        # one target class in this evaluation dataset.
        #
        # Therefore we deduplicate globally by normalized
        # query, not only by query + expected class.
        # ----------------------------------------------------
        df["query_normalized"] = (df["query"].str.strip().str.casefold())
        before = len(df)
        duplicated_queries = (df["query_normalized"].duplicated(keep=False))
        duplicate_count = int(duplicated_queries.sum())

        if duplicate_count:
            print(f"\n[!] Found {duplicate_count} rows participating in cross-class duplicates.")

            # Keep the first occurrence.
            df = df.drop_duplicates(subset=["query_normalized"],keep="first")

        after = len(df)
        print(f"[*] Removed {before - after} duplicate rows.")

        # Add stable sample ID.
        df.insert(0,"sample_id",[f"sample_{i:06d}" for i in range(1,len(df) + 1,)],)

        # Remove helper column:
        df = df.drop(columns=["query_normalized"])

        # Ensure deterministic ordering
        df = df.sort_values(by=["expected_class","query",],kind="stable",).reset_index(drop=True)

        # Reassign IDs after sorting.
        df["sample_id"] = [f"sample_{i:06d}" for i in range(1,len(df) + 1,)]

        # Validate.
        self._validate_dataset(df)

        # Save
        OUTPUT_PARQUET.parent.mkdir(parents=True,exist_ok=True)
        df.to_parquet(OUTPUT_PARQUET,engine="pyarrow",index=False)
        print(f"\n[+] Success!")
        print(f"Samples: {len(df)}")
        print(f"Classes: {df['expected_class'].nunique()}")
        print(f"Output: {OUTPUT_PARQUET}")

        # Distribution report
        print("\n[*] Samples per class:")
        counts = (df["expected_class"].value_counts().sort_index())
        print(counts.to_string())
        print("\n[*] Samples by depth:")
        depth_counts = (df["depth"].value_counts().sort_index())
        print(depth_counts.to_string())
        
        # Leaf distribution
        print("\n[*] Leaf count by depth:")
        leaf_depth_counts = {}
        for class_name, metadata in (self.class_metadata.items()):
            if not metadata["is_leaf"]:
                continue
            depth = metadata["depth"]
            leaf_depth_counts[depth] = (leaf_depth_counts.get(depth,0)+ 1)

        for depth in sorted(leaf_depth_counts):
            print(f"    depth {depth}: {leaf_depth_counts[depth]} leaf classes")

# Main
if __name__ == "__main__":

    # Load configuration:
    with open(CONFIG_PATH,"r",encoding="utf-8") as f:
        config_data = (yaml.safe_load(f) or {})

    # Build RAG
    settings = Settings()
    engine = build_rag(settings)

    # Load hierarchy
    with open(NESTED_DATA_PATH,"r",encoding="utf-8") as f:
        nested_structure = json.load(f)

    # Generate dataset
    generator = (StratifiedQAGenerator(rag_engine=engine,nested_data=nested_structure,config=config_data))

    generator.run()