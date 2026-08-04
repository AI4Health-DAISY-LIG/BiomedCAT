"""Stage 2 (NER): extract typed biomedical entities from OCR text, following ZeroTuneBio.

Text is split into sentences deterministally, then each sentence passes through three
LLM modules: extract every candidate term, classify each into a type, and drop the typings
flagged wrong. The deterministic steps (preprocessing, grounding, dedup) are reproducible
by construction, so non-determinism is confined to the three model calls.
"""
import re
import json
import logging
import unicodedata

from biomedcat.runtime import build_llm, generate, free_gpu   # importing runtime bootstraps CUDA/determinism
from biomedcat.types import Entity, ENTITY_TYPES
from biomedcat import prompts
import en_core_sci_sm

logger = logging.getLogger(__name__)


_nlp = None   # scispaCy pipeline, loaded once (CPU) and reused across calls


def _get_nlp():
    """Load and cache the scimspaCy model, used here for sentence segmentation only."""
    global _nlp
    if _nlp is None:
        _nlp = en_core_sci_sm.load()
    return _nlp


class NERPipeline:
    """Zero-shot biomedical NER over the shared 4-bit Gemma4, following ZeroTuneBio.

    Holds the three pieces of state the modules thread through -- the model, the tokenizer,
    and the scispaCy splitter -- so the deep call tree (extract -> run_zerotune -> classify /
    error_filter) does not have to pass them around by hand.
    """

    def __init__(self, model_id: str | None = None):
        self.nlp = _get_nlp()      # scispacy (CPU): sentence splitting only
        self.model_id = model_id

    def load(self):
        """Prepare the pipeline state with the specified model ID."""
        if not self.model_id:
            logger.warning("No model_id provided to NERPipeline. Using default.")

        else:
            logger.info("Preparing NER pipeline for model: %s", self.model_id)

    def unload(self):
        """Release GPU resources."""
        free_gpu()

    # --- Deterministic preprocessing -------------------------------------------------

    def preprocess(self, text: str) -> list[str]:
        """Clean OCR text and split it into sentences. Fully rule-based, hence reproducible."""
        if not text.strip():                        # empty or image-only slide
            return []

        # Canonicalise OCR Unicode quirks (full-width digits, ligatures) for consistent matching.
        text = unicodedata.normalize("NFKC", text)

        # Rejoin words split across lines ("muscu-\nlar" -> "muscular"); must precede the newline collapse.
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Flatten layout to prose for the splitter. This merges slide bullets into one run, since
# ... rest of code ...
