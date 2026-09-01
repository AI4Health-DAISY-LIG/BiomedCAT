import json
import random
import yaml
import ollama
import pandas as pd
from openai import OpenAI # Import for OpenAI compatibility

from pathlib import Path
from typing import List, Dict, Any, Tuple

# Assuming these modules exist and provide necessary functionality
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
# Dataset design (Controlled Batch Iteration)
# ------------------------------------------------------------

TARGET_PER_CLASS = 30

BATCH_SIZE = 5  # Reduced batch size for better control and diversity
MAX_ATTEMPTS_PER_CLASS = 12 # Increased attempts to compensate for smaller batches

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


    def _analyze_hierarchy(self):
        """
        Recursively analyze the nested ontology.
        ... (omitted for brevity, logic remains same as original) ...
        """

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
        ... (omitted for brevity, logic remains same as original) ...
        """
        return self.config.get("sampling",{})

    # Sampling plan:
    def _get_sampling_plan(self):
        """
        Determine which classes to sample and how many, incorporating quality constraints.
        Returns a list of tuples: (class_name, target, style)
        """
        dataset_cfg = self.config.get("dataset", {})
        target_size = int(dataset_cfg.get("target_size", 4000))

        # --- Contrôle de Profondeur (Hierarchy Depth) ---
        depth_cfg = self.config.get("constraints", {}).get("hierarchy_depth", {})
        bins = depth_cfg.get("bins", []) # Ex: [[1, 2], [3, 4]]

        # Initialisation du plan de génération (classe -> {target, style})
        generation_plan: Dict[str, Dict[str, Any]] = {}
        total_planned_samples = 0

        for class_name, metadata in self.class_metadata.items():
            if not metadata["is_leaf"]:
                continue

            # Déterminer le target de base (par défaut)
            base_target = int(self.config.get("constraints", {}).get("leaf_class", {}).get("target", TARGET_PER_CLASS))
            
            # --- LOGIQUE DE BIAIS PAR PROFONDEUR ---
            depth = metadata["depth"]
            adjusted_target = base_target

            for bin_start, bin_end in bins:
                if bin_start <= depth <= bin_end:
                    # Exemple simple : si on est dans un bin ciblé, augmenter le target de 20%
                    adjusted_target = int(base_target * 1.2) 
                    break

            # --- LOGIQUE DE STYLE (Ambiguity & Surface Form) ---
            style_cfg = self.config.get("constraints", {}).get("ambiguity", {})
            surface_cfg = self.config.get("constraints", {}).get("surface_form", {})
            
            required_style = {
                "ambiguity": style_cfg.get("categories", {}).get("ambiguous") if style_cfg.get("enabled") else None,
                "surface_form": surface_cfg.get("categories", {}).get("uncommon") if surface_cfg.get("enabled") else None
            }

            # Stockage du plan avec le style requis
            generation_plan[class_name] = {
                "target": adjusted_target,
                "style": required_style
            }
            total_planned_samples += adjusted_target


        # --- LOGIQUE DE CIBLE GLOBALE (Global Target Size) ---
        if total_planned_samples > target_size:
             print(f"[!] Warning: Total planned samples ({total_planned_samples}) exceeds global target size ({target_size}). Scaling down targets proportionally.")
             scaling_factor = target_size / total_planned_samples
             for class_name, plan in generation_plan.items():
                 plan["target"] = int(plan["target"] * scaling_factor)

        # Convertir le dictionnaire en liste de tuples pour l'itération (classe, target, style)
        final_sampling_list = []
        for class_name, plan in generation_plan.items():
            final_sampling_list.append((class_name, plan["target"], plan["style"]))

        return final_sampling_list


    # Definition lookup:
    def _get_definition(self,class_name: str) -> str:
        if class_name not in self.rag_engine.flat_data:
            return ""

        metadata = self.rag_engine.flat_data[class_name].get("metadata",{})
        return (metadata.get("definition","") or "")

    # LLM generation (Batch implementation)
    def _generate_batch(self, class_name: str, definition: str, n: int, rag_context: Dict[str, Any], negative_queries: List[str], style: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generates a batch of queries using LLM with structural constraints and negative filtering.

        rag_context: Structured data from RAG engine (parents, children, etc.).
        negative_queries: List of already seen query strings to avoid repetition.
        style: Dictionary containing stylistic requirements (ambiguity, surface_form).
        """
        ancestors = self.class_metadata[class_name]["ancestors"]
        parent_context = ancestors[-1] if ancestors else "None"

        # Format RAG context for the prompt (Positive Constraints)
        rag_constraints = f"""
        --- STRUCTURAL CONSTRAINTS ---
        Target Class: {class_name}
        Definition: {definition}
        Immediate Parent: {parent_context}
        Known Ancestors/Context: {', '.join(ancestors)}
        Relevant Subclasses (Children): {rag_context.get('children', 'None')}
        """

        # Format Negative Queries for the prompt (Anti-Bias)
        negative_list = "\n".join([f"- {q}" for q in negative_queries])
        negative_constraints = f"""
        --- NEGATIVE CONSTRAINTS ---
        DO NOT generate any query that is identical to or too similar to these previously used queries:
        {negative_list if negative_list else 'None'}
        """

        # --- NOUVEAU : Construction des instructions stylistiques basées sur le YAML ---
        style_instructions = []
        if style and style.get("ambiguity"):
            level = "highly ambiguous" if style["ambiguity"] == 0.3 else "unambiguous" # Logique simple basée sur les catégories
            style_instructions.append(f"The query must be {level} in its semantic scope.")

        if style and style.get("surface_form"):
            form = "common terminology" if style["surface_form"] == 0.5 else "uncommon or highly technical jargon"
            style_instructions.append(f"Use language that reflects a {form} surface form.")
        # --- FIN NOUVEAU ---

        prompt = f"""
                    You are a biomedical ontology expert generating high-quality evaluation data.
                    Your task is to generate exactly {n} DISTINCT biomedical entity mentions for the target class.

                    {rag_constraints}
                    {negative_constraints}
                    
                    --- STYLISTIC CONSTRAINTS ---
                    {'\n'.join(style_instructions) if style_instructions else 'No specific stylistic constraints applied.'}

                    Requirements:
                    1. Each query must be 1-3 words long.
                    2. Use realistic, concrete biomedical terminology specific to "{class_name}".
                    3. The generated entities MUST belong to "{class_name}" and not its broader parent classes or unrelated concepts.
                    4. Avoid generic or highly ambiguous terms (unless explicitly requested).
                    5. Return ONLY a JSON array of objects with 'query' and 'expected_class'.

                    Example Output Format:
                    [{"query": "Query 1", "expected_class": "{class_name}"}, {"query": "Query 2", "expected_class": "{class_name}"}]
                    """

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "expected_class": {"type": "string"},
                },
                "required": ["query", "expected_class"],
            },
        }

        # --- LLM Client Abstraction Logic ---
        llm_cfg = self.config.get("llm", {})
        use_openai_compatible = llm_cfg.get("use_openai_compatible", False)
        model_to_use = MODEL_NAME # Default model name

        if use_openai_compatible:
            client = OpenAI(
                api_key=llm_cfg["api_key"], 
                base_url=llm_cfg.get("base_url", "http://localhost:8080/v1") # Default base URL if not provided
            )
            # Note: We use the model name defined in config or default to MODEL_NAME
            model_to_use = llm_cfg.get("model", MODEL_NAME)

            try:
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} # OpenAI uses response_format for JSON output
                )
                # The structure of the response content differs from Ollama
                content = response.choices[0].message.content
            except Exception as e:
                print(f"  [!] OpenAI generation error for {class_name}: {e}")
                return []

        else:
            # Fallback to Ollama implementation (Original logic)
            response = ollama.chat(
                model=MODEL_NAME,
                format=schema, # Note: Ollama format parameter is used here
                messages=[{"role": "user", "content": prompt}],
            )

            content = response["message"]["content"]
        # --- End LLM Client Abstraction Logic ---


        try:
            data = json.loads(content)
            if not isinstance(data, list):
                return []
            
            valid_batch = []
            for item in data:
                query = (item.get("query","").strip())
                if query and 1 <= len(query.split()) <= 3:
                    # Basic whitespace normalization
                    normalized_query = " ".join(query.split()).casefold()
                    valid_batch.append({"query": normalized_query, "expected_class": class_name})

            return valid_batch

        except Exception as e:
            print(f"  [!] Parsing error for {class_name}: {e}")
            return []


    # Generate one class (Iterative Controller)
    def _generate_for_class(self, class_name: str, target: int, style: Dict[str, Any]) -> List[Dict[str, Any]]:

        definition = self._get_definition(class_name)
        if not definition:
            print(f"[!] No definition for {class_name}")
            return []

        metadata = self.class_metadata[class_name]
        collected: List[Dict[str, Any]] = []
        seen_queries: set = set() # Set of normalized queries already collected
        attempts = 0

        print(f"\n--- Starting generation for {class_name} (Target: {target}, Style: {style}) ---")

        while len(collected) < target and attempts < MAX_ATTEMPTS_PER_CLASS:
            attempts += 1
            remaining = target - len(collected)
            request_n = min(BATCH_SIZE, remaining)

            print(f"  [Attempt {attempts}/{MAX_ATTEMPTS_PER_CLASS}] Requesting {request_n} samples...")

            # Phase 2.2: Get RAG Context (Positive Constraints)
            try:
                rag_context = self.rag_engine.get_context(class_names=[class_name])
            except Exception as e:
                print(f"  [!] Failed to retrieve RAG context for {class_name}: {e}. Skipping attempt.")
                continue

            # Phase 2.2/3.3: Get Negative Constraints (Anti-Bias)
            negative_queries = list(seen_queries)

            # Phase 2.1: Générer Batch (Passage du style au générateur)
            generated_batch = self._generate_batch(
                class_name=class_name,
                definition=definition,
                n=request_n,
                rag_context=rag_context,
                negative_queries=negative_queries,
                style=style # Passage du style ici
            )

            newly_accepted_samples: List[Dict[str, Any]] = []
            for sample in generated_batch:
                query = sample["query"] # Already normalized by _generate_batch
                key = query.casefold()

                # Phase 2.2/3.3: Validation and Filtering (Strict Uniqueness)
                if key not in seen_queries:
                    seen_queries.add(key)
                    newly_accepted_samples.append(sample)

            # Update collected samples
            for sample in newly_accepted_samples:
                collected.append({
                        "query": sample["query"],
                        "expected_class": class_name,
                        # Hierarchy metadata
                        "depth": metadata["depth"],
                        "is_leaf": metadata["is_leaf"],
                        "ancestors": metadata["ancestors"],
                        "path": metadata["path"],
                        "parent_class": (
                            metadata["ancestors"][-1]
                            if metadata["ancestors"]
                            else None
                        ),
                        # Phase 3.1: Tracking generation context
                        "generation_attempt": attempts,
                        "rag_context_used": rag_context # Store constraints used for traceability
                    }
                )

            print(f"  -> Accepted {len(newly_accepted_samples)} new unique samples.")
            print(f"  Current total: {len(collected)} / {target}")


        if len(collected) < target:
            print(f"[!] Generation finished. Only generated {len(collected)} / {target} for {class_name}. Max attempts reached or no new unique samples found.")

        return collected

    # Final dataset validation (Unchanged)
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

    # Main execution (Modified to use the new iterative generator)
    def run(self):

        sampling_plan = self._get_sampling_plan() # Utilise le nouveau plan structuré
        print(f"\n[*] Leaf classes selected: {len(sampling_plan)}")
        print(f"[*] Planned total samples: {sum(target for _, target, _ in sampling_plan)}")

        # Générer itérativement :
        for i, (class_name, target, style) in enumerate(sampling_plan, start=1): # Déstructuration du plan
            metadata = self.class_metadata[class_name]
            print(f"\n=====================================================")
            print(f"[{i}/{len(sampling_plan)}] Starting generation for: {class_name}")
            print(f"Depth: {metadata['depth']}, Target: {target}, Style: {style}") # Affichage du style

            # Appel de la méthode avec le target et le style
            samples = (self._generate_for_class(class_name=class_name, target=target, style=style)) 
            self.all_samples.extend(samples)

        if not self.all_samples:
            print("[!] No samples generated.")
            return

        # Final dataframe creation and deduplication (Unchanged logic)
        df = pd.DataFrame(self.all_samples)

        # Global deduplication based on normalized query
        df["query_normalized"] = (df["query"].str.strip().str.casefold())
        before = len(df)
        duplicated_queries = (df["query_normalized"].duplicated(keep=False))
        duplicate_count = int(duplicated_queries.sum())

        if duplicate_count:
            print(f"\n[!] Found {duplicate_count} rows participating in cross-class duplicates.")
            # Keep the first occurrence.
            df = df.drop_duplicates(subset=["query_normalized"],keep="first")

        after = len(df)
        print(f"[*] Removed {before - after} duplicate rows globally.")

        # Add stable sample ID and clean up columns
        df.insert(0,"sample_id",[f"sample_{i:06d}" for i in range(1,len(df) + 1,)],)
        df = df.drop(columns=["query_normalized"])

        # Ensure deterministic ordering
        df = df.sort_values(by=["expected_class","query",],kind="stable").reset_index(drop=True)
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

        # Distribution report (Unchanged)
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

# Main execution (Unchanged)
if __name__ == "__main__":

    with open(CONFIG_PATH,"r",encoding="utf-8") as f:
        config_data = (yaml.safe_load(f) or {})

    settings = Settings()
    engine = build_rag(settings)

    with open(NESTED_DATA_PATH,"r",encoding="utf-8") as f:
        nested_structure = json.load(f)

    generator = (StratifiedQAGenerator(rag_engine=engine, nested_data=nested_structure, config=config_data))

    generator.run()
