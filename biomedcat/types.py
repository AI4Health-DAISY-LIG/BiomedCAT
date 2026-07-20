from dataclasses import dataclass


# The ten biomedical entity types an Entity.type can hold, and one gloss each.
# Single source of truth shared by NER (typing), the prompts, and Norm (the judge),
# so those never drift on what a type label means. Kept here (torch-free) rather than
# in runtime, so any module can import the vocabulary without pulling in torch.
# Each type maps to a Biolink class in retrieval.TYPE_TO_BIOLINK.
ENTITY_TYPES: list[str] = [
    "GENE", "PROTEIN", "DISEASE", "PHENOTYPIC_FEATURE", "CHEMICAL",
    "CELL_TYPE", "CELLULAR_COMPONENT", "ANATOMY", "BIOLOGICAL_PROCESS",
    "SEQUENCE_VARIANT",
]

# Adapted from the Biolink Model 4.4.3 class descriptions: each gloss keeps the boundary
# the model draws but is worded to discriminate. GENE and PROTEIN mirror each other, and
# ANATOMY / CELL_TYPE / CELLULAR_COMPONENT are disjoint by construction.
TYPE_DEFINITIONS: dict[str, str] = {
    "GENE":               "a gene or gene locus: the DNA region encoding a transcript",
    "PROTEIN":            "a protein or gene product, produced by translation of mRNA",
    "DISEASE":            "a disease, disorder, or medical condition",
    "PHENOTYPIC_FEATURE": "an observable sign, symptom, phenotype, or trait",
    "CHEMICAL":           "a chemical compound, drug, or other chemical entity",
    "CELL_TYPE":          "a whole cell",
    "CELLULAR_COMPONENT": "a location in or around a cell, such as an organelle",
    "ANATOMY":            "a tissue, organ, or other anatomical structure of more than one cell",
    "BIOLOGICAL_PROCESS": "a biological process, pathway, or molecular activity",
    "SEQUENCE_VARIANT":   "a mutation, variant, or allele",
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