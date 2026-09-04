from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional
from biomedcat.stages.biolink_yml_processor import biolink_yml_processor


# The flat Biolink class table lives in <repo root>/data, next to the ChromaDB index (see config.py).
BASE_DIR = Path(__file__).parent.parent.absolute()
BIOLINK_FILE_PATH = BASE_DIR / "data/biolink_classes_flat.json"

if BIOLINK_FILE_PATH.is_file():
    # checks if file exists
    print ("Loading biolink info...")
    with open(str(BIOLINK_FILE_PATH)) as json_data:
        biolink_info_flat = json.load(json_data)
else:
    print ("Loading state of the art data model and computing local knowledge base...")
    nested_result, biolink_info_flat = biolink_yml_processor()
    print("biolink data model processing... done.")


# The seven biomedical entity types an Entity.type can hold, and one gloss each.
# Single source of truth shared by NER (typing), the prompts, and Norm (the judge),
# so those never drift on what a type label means. Kept here (torch-free) rather than
# in runtime, so any module can import the vocabulary without pulling in torch.
ENTITY_TYPES: list[str] = list(biolink_info_flat.keys())

TYPE_DEFINITIONS: dict[str, str] = {k:v["metadata"]["definition"] for k,v in biolink_info_flat.items()}

@dataclass
class Slide:
    page: int
    text: str

@dataclass
class Entity:
    text: str
    type: str
    segment: str
    # 1-based slide/page the mention was found on; provenance for the context graph stage.
    page: int | None = None

@dataclass
class Candidate:
    curie: str
    label: str
    biolink_type: str | None
    rank: int
    source: str
    # Canonical RTX-KG2c id for this CURIE (from the offline equivalents table), None if unknown.
    kg2c_id: str | None = None

@dataclass
class NormalizedEntity(Entity):
    curie: str | None = None
    # Canonical RTX-KG2c id of the chosen CURIE; this is the seed used by the context-graph stage.
    kg2c_id: str | None = None
    # Human-readable label of the chosen candidate, kept for the report and the exports.
    label: str | None = None

@dataclass
class PipelineResult:
    filename: str
    ocr: list[Slide]
    ner: list[Entity]
    norm: list[NormalizedEntity]
    # Summary of the context-graph stage (ContextGraphResult as a dict), None when skipped.
    context_graph: Optional[dict] = None