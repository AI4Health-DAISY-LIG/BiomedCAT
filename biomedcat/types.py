from dataclasses import dataclass
from pathlib import Path
import json
from biolink_yml_processor import biolink_yml_processor


BASE_DIR = Path(__file__).parent.absolute()
BIOLINK_FILE_PATH = BASE_DIR / "data/biolink_classes_flat.json"

if BIOLINK_FILE_PATH.is_file():
    # checks if file exists
    print ("Loading biolink info...")
    with open(str(BIOLINK_FILE_PATH)) as json_data:
        biolink_info_flat = json.load(json_data)
else:
    print ("Loading state of the art data model and computing local knowledge base...")
    nested_result, flat_result = biolink_yml_processor()


# The seven biomedical entity types an Entity.type can hold, and one gloss each.
# Single source of truth shared by NER (typing), the prompts, and Norm (the judge),
# so those never drift on what a type label means. Kept here (torch-free) rather than
# in runtime, so any module can import the vocabulary without pulling in torch.
ENTITY_TYPES: list[str] = list(biolink_info_flat.keys())
# [
#     "GENE", "DISEASE", "CHEMICAL", "CELL_TYPE",
#     "ANATOMY", "CHROMOSOMAL_LOCUS", "EPIGENETIC_MODIFICATION",
# ]

TYPE_DEFINITIONS: dict[str, str] = {k:v["definition"] for k,v in biolink_info_flat.items()}
# {
#     "GENE":                    "a gene or gene symbol",
#     "DISEASE":                 "a disease or disorder",
#     "CHEMICAL":                "a chemical, drug, or compound",
#     "CELL_TYPE":               "a type of cell",
#     "ANATOMY":                 "an anatomical structure, tissue, or organ",
#     "CHROMOSOMAL_LOCUS":       "a chromosomal location, locus, or genomic region",
#     "EPIGENETIC_MODIFICATION": "an epigenetic modification such as methylation",
# }


@dataclass
class Slide:
    page: int
    text: str

@dataclass
class Entity:
    text: str
    type: str
    segment: str

@dataclass
class Candidate:
    curie: str
    label: str
    biolink_type: str | None
    rank: int
    source: str

@dataclass
class NormalizedEntity(Entity):
    curie: str | None = None

@dataclass
class PipelineResult:
    filename: str
    ocr: list[Slide]
    ner: list[Entity]
    norm: list[NormalizedEntity]