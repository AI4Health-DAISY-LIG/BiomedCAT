"""Prompt templates for the NER and normalization stages.

All model-facing wording lives here, so the prompts -- the research-tuned core of the
pipeline -- can be read and revised in one place, separate from the parsing and
orchestration logic in the stages. Each function returns a chat `messages` list ready to
hand to runtime.generate. The module is torch-free: it formats the type vocabulary itself
and takes the remaining dynamic pieces (candidate menu, verdict listing) as strings.
"""
import json
from biomedcat.types import Entity, TYPE_DEFINITIONS


# --- OCR ---
def ocr_description():
    prompt = """Role Definition:
    You are an esteemed Principal Investigator (PI) at a top-tier biomedical research institution.
    Your expertise spans molecular biology, genetics, chemistry, and clinical translational science.
    Task Goal: Deconstruct the provided visual data as if you are preparing the executive summary for an international 
    scientific symposium or writing the critical background of a major grant proposal (R01/Genetics/NIH-equivalent). 
    Your aim is to translate complex figures into a coherent narrative of current findings, unresolved questions, and core hypotheses.
    Analysis Directives:
    Comprehensiveness: Identify every distinct component within the visualization—including structural elements (chromosomes, RNA loops, small molecules), 
    biochemical markers (proteins, metabolites), morphological features, and experimental conditions.
    Visual Granularity: Provide exhaustive descriptive language for all visual evidence: specify colors, geometries, scales, specific annotations, 
    relationships between components (e.g., 'A direct correlation is shown where...'), and differences between comparative data sets or images.
    Do not focus on descripton of 'healthy', 'control', 'normal' cases.
    Synthesize Concepts: Go beyond simple listing; describe the observed phenomena, hypothesize the underlying mechanisms linking structure to function, 
    and articulate how different visual elements support or refute a scientific hypothesis. If multiple similar processes are depicted (e.g., two pathways for FSHD),
    systematically contrast their defining features.
    Output Constraints:
    Your output must be formatted as a numbered list of highly complex scientific statements/concepts.
    Avoid conversational titles, introductory phrases, or summarizing sentences. Each entry must convey a singular, dense concept.
    If any visual element's function is ambiguous, add [UNCLEAR] as a concept prefix."""

    return prompt



# --- NER Module 1: extract all candidate terms (recall-first, few-shot) ---
_EXTRACTOR_SYSTEM = (
    "You are a biomedical ontologist and bioinformatician. "
    "Extract professional biomedical terms from the given text."
)


def extraction_messages(sentence: str) -> list[dict[str, str]]:
    """List every professional biomedical term in the sentence (no filtering)."""
    return [
        {"role": "system", "content": _EXTRACTOR_SYSTEM},
        {"role": "user", "content": (
            "Identify ALL biomedical professional terms and concepts mentioned in the text below."
            "Do NOT filter or judge them -- list every professional term to maximise recall."
            "Ensure that multi-word concepts (composite terms) are extracted as a single complete phrase representing a concept."
            "Return ONLY a valid JSON array of strings, exactly as they appear in the text."
            "Text: The TP53 gene mutation is common in non-small cell lung cancer.Terms:"
        )},
        {"role": "assistant", "content": '["TP53 gene mutation", "non-small cell lung cancer"]'},
        {"role": "user", "content": f"Text: {sentence}\nTerms:"},
    ]


# --- NER Module 2: classify one grounded term into a single type (or NONE) ---
_CLASSIFIER_SYSTEM = "You are a strict biomedical annotation expert."


def classification_messages(term: str, sentence: str) -> list[dict[str, str]]:
    """Explain a term in context, then assign it one entity type or NONE."""
    parts = []
    for name, gloss in TYPE_DEFINITIONS.items():
        parts.append(f"{name} ({gloss})")
    type_defs = ", ".join(parts)

    return [
        {"role": "system", "content": _CLASSIFIER_SYSTEM},
        {"role": "user", "content": (
            f"Sentence: {sentence}\n\n"
            f'Entity: "{term}"\n\n'
            f'Step 1: Explain what "{term}" means in the context of this sentence.\n'
            "Step 2: Based on that meaning, choose the SINGLE most relevant type, "
            f"or NONE if it does not clearly belong to any of:\n{type_defs}.\n"
            "Answer NONE for anything that is not itself a biomedical concept, such as "
            "a person's name, an author citation, a journal name, or a URL.\n"
            "End your answer with a final line in exactly this form:\n"
            "TYPE: <one type or NONE>"
        )},
    ]

def classification_messages_biolink(term: str, sentence: str) -> list[dict[str, str]]:
    """Explain a term in context, then assign it one entity type or NONE."""
    global BIOLINK_TYPES_CACHE
    if BIOLINK_TYPES_CACHE is None:
        parts = []
        try:
            print("Loading Biolink:")
            with open(DATA_FILE_PATH, "r", encoding="utf-8") as f: 
                data = json.load(f)
            for name, entry in data.items():
                definition = entry.get("definition")
                if definition:
                    parts.append(f"{name} ({definition})")
            BIOLINK_TYPES_CACHE = ", ".join(parts)
            print("Loading Biolink: done.")
        except Exception:
            # Fallback to TYPE_DEFINITIONS if file loading fails
            parts = [f"{n} ({g})" for n, g in TYPE_DEFINITIONS.items()]
            BIOLINK_TYPES_CACHE = ", ".join(parts)

    type_defs = BIOLINK_TYPES_CACHE

    return [
        {"role": "system", "content": _CLASSIFIER_SYSTEM},
        {"role": "user", "content": (
            f"Sentence: {sentence}\n\n"
            f'Entity: "{term}"\n\n'
            f'Step 1: Explain what "{term}" means in the context of this sentence.\n'
            "Step 2: Based on that meaning, choose the SINGLE most relevant type, "
            f"or NONE if it does not clearly belong to any of:\n{type_defs}.\n"
            "Answer NONE for anything that is not itself a biomedical concept, such as "
            "a person as a person's name, an author citation, a journal name, or a URL, or any personal information.\n"
            "End your answer with a final line in exactly this form:\n"
            "TYPE: <one type or NONE>"
        )},
    ]

# --- NER Module 3: flag any type assignment that is wrong ---
_VERIFIER_SYSTEM = "You are a strict biomedical annotation verifier."


def error_filter_messages(listing: str, sentence: str) -> list[dict[str, str]]:
    """List only the '(term = type)' lines whose type is wrong, else NONE.

    `listing` is the caller-formatted 'term = TYPE' block, one entry per line.
    """
    return [
        {"role": "system", "content": _VERIFIER_SYSTEM},
        {"role": "user", "content": (
            f"Sentence: {sentence}\n\n"
            f"Each line proposes a biomedical entity and its type:\n{listing}\n\n"
            "Re-examine each line in the context of the sentence. List ONLY the terms whose type "
            "is WRONG, one term per line, exactly as written. If every typing is correct, reply NONE."
        )},
    ]


# --- Normalization: pick the best CURIE from a numbered candidate menu ---
_JUDGE_SYSTEM = (
    "You are a biomedical entity normalization expert. Given an entity, the sentence it "
    "appears in, and a numbered list of candidate identifiers, pick the single candidate "
    "that best matches the entity in that context, or 0 if none of them fit."
)


def judge_messages(entity: Entity, menu: str, format_instructions: str) -> list[dict[str, str]]:
    """Choose the candidate whose type, organism, and meaning all match the entity, or 0 (NIL).

    `menu` is the numbered candidate list, `format_instructions` the parser's schema.
    """
    definition = TYPE_DEFINITIONS.get(entity.type)
    type_str = f"{entity.type} = {definition}" if definition else entity.type

    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": (
            f'Entity: "{entity.text}"  (type: {type_str})\n'
            f"Sentence: {entity.segment}\n\n"
            f"Pick the candidate that is the correct identifier for this entity.\n"
            f"Rules:\n"
            f"- The chosen candidate's type (shown in [brackets]) MUST be consistent with the entity "
            f"type {entity.type}: e.g. a DISEASE maps to a Disease concept, a GENE to a Gene -- "
            f"never link a disease to a gene or protein.\n"
            f"- The chosen candidate must match the organism under discussion. Unless the sentence "
            f"indicates another species, assume human, and do not choose a candidate whose label "
            f"names a different organism (for example 'mouse', 'Arabidopsis').\n"
            f"- Among candidates of the correct type, use the sentence for context to choose the best one.\n"
            f"- Answer 0 only if NO candidate matches the meaning, the type, and the organism.\n\n"
            f"{menu}\n\n"
            f"{format_instructions}"
        )},
    ]
