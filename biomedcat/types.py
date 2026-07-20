from dataclasses import dataclass


# The seven biomedical entity types an Entity.type can hold, and one gloss each.
# Single source of truth shared by NER (typing), the prompts, and Norm (the judge),
# so those never drift on what a type label means. Kept here (torch-free) rather than
# in runtime, so any module can import the vocabulary without pulling in torch.
ENTITY_TYPES: list[str] = [
    "GENE", "DISEASE", "CHEMICAL", "CELL_TYPE",
    "ANATOMY", "CHROMOSOMAL_LOCUS", "EPIGENETIC_MODIFICATION",
]

TYPE_DEFINITIONS: dict[str, str] = {
    "GENE":                    "a gene or gene symbol",
    "DISEASE":                 "a disease or disorder",
    "CHEMICAL":                "a chemical, drug, or compound",
    "CELL_TYPE":               "a type of cell",
    "ANATOMY":                 "an anatomical structure, tissue, or organ",
    "CHROMOSOMAL_LOCUS":       "a chromosomal location, locus, or genomic region",
    "EPIGENETIC_MODIFICATION": "an epigenetic modification such as methylation",
}


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