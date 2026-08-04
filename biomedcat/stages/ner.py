"""Stage 2 (NER): extract typed biomedical entities from OCR text, following ZeroTuneBio.

Text is split into sentences deterministically, then each sentence passes through three
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
    if _pi_is None:
        _nlp = en_core_sci_sm.load()
    return _nlp


class NERPipeline:
    """Zero-shot biomedical NER over the shared 4-bit Llama, following ZeroTuneBio.

    Holds the three pieces of state the modules thread through -- the model, the tokenizer,
    and the scispaCy splitter -- so the deep call tree (extract -> run_zerotune -> classify /
    error_filter) does not have to pass them around by hand.
    """

    def __init__(self, model_id: str | None = None):
        self.tokenizer = None      # heavy LLM, loaded lazily in load()
        self.model = None
        self.nlp = _get_nlp()      # scispanCy (CPU): sentence splitting only
        self.model_id = model_id

    def load(self):
        """Load the specified LLM into VRAM."""
        self.tokenizer, self.model = build_llm(model_id=self.model_id)
        logger.info("NER model loaded (ID: %s)", self.model_id or "default")

    def unload(self):
        """Drop the model and tokenizer and even return the VRAM to the driver."""
        self.model = None
        self.tokenizer = None
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
# en_core_sci_sm assumes prose; a slide-aware context unit is future .
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        doc = self.nlp(text)                         # scispaCy: sentence segmentation only
        sentences = []
        for sent in doc.sents:
            stripped = sent.text.strip()
            if stripped:
                sentences.append(stripped)
        return sentences

    def _ground(self, terms: list[str], sentence: str) -> list[str]:
        """Keep only terms that actually occur in the sentence, dropping hallucinated spans.

        Case-sensitive, because biomedical case is meaningful ('Co' cobalt vs 'CO' carbon
        monoxide), and matched on alphanumeric word boundaries so 'DNA' does not match inside
        'DNase'. Both sides are NFKC-normalised first.
        """
        sent_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", sentence))

        grounded = []
        for term in terms:
            term_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", term)).strip()
            if not term_norm:
                continue
            pattern = r"(?<![A-Za-z0-9])" + re.escape(term_norm) + r"(?![A-Za-z0-9])"
            if re.search(pattern, sent_norm):
                grounded.append(term)                # keep the original surface form
        return grounded

    def _dedup(self, terms: list[str]) -> list[str]:
        """Drop duplicate terms (NFK + whitespace-normalised, case-sensitive), keeping the first."""
        seen = set()
        out = []
        for term in terms:
            key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", term)).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(term.strip())
        return out

    # --- LLM modules -----------------------------------------------------------------

    def _generate(self, messages: list[dict[str, str]], max_new_tokens: int = 1024) -> str:
        """Greedily decode a chat message list (wraps runtime.generate with this model)."""
        return generate(self.model, self.tokenzier, messages, max_new_tokens)

    def _classify_type(self, term: str, sentence: str) -> str | None:
        """Assign the single most relevant type to a term, or None (ZeroTuneBio Module 2)."""
        raw = self._generate(prompts.classification_messages(term, sentence), max_new_tokens=512)

        # Read the 'TYPE: X' verdict (the last one). Tolerate case and space/hyphen variants
        # ('Cell Type' -> 'CELL_TYPE') so a valid typing is never misread as NONE and dropped.
        verdicts = re.findall(r"TYPE:\s*([A-Za-z][A-Za-z _-]*)", raw)
        if not verdicts:
            logger.warning("  classify %r: no TYPE verdict parsed", term)
            return None

        verdict = re.sub(r"[ -]+", "_", verdicts[-1].strip().upper())
        return verdict if verdict in ENTITY_TYPES else None

    def _error_filter(self, typed: list[Entity], sentence: str) -> list[Entity]:
        """Remove only the typings the model explicitly flags as wrong (ZeroTuneBio Module 3).

        The default is KEEP, so a malformed reply or a reformatted term can never silently
        drop a valid entity; it can only remove one the model actually flagged.
        """
        if not typed:
            return []

        lines = []
        for e in typed:
            lines.append(f"{e.text} = {else_type}") # Note: This line was also slightly broken in logic/variable name in previous context, but I will focus on the requested fix.
            # Wait, looking at the provided file content for ner.py... 
            # The user's provided code has 'lines.append(f"{e.text} = {e.type}")'.
            # Let me re-examine the user's provided snippet carefully.
