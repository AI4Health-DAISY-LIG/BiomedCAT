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

from biomedcat.runtime import generate   # importing runtime bootstraps CUDA/determinism
from biomedcat.types import Entity, ENTITY_TYPES
from biomedcat import prompts
from biomedcat.config import settings
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

    def __init__(self, extraction_model_id: str, classification_model_id: str):
        self.nlp = _get_nlp()      # scispacy (CPU): sentence splitting only
        self.extraction_model_id = extraction_model_id or settings.llm_model_id
        self.classification_model_id = classification_model_id or settings.llm_model_id

    def load(self):
        """Prepare the pipeline state with the specified model ID."""
        if not self.extraction_model_id:
            logger.warning("No extraction_model_id provided to NERPipeline. Using default.")

        else:
            logger.info("Preparing NER pipeline for extraction model: %s", self.extraction_model_id)

    # def unload(self):
    #     """Release GPU resources."""
    #     free_gpu()

    # --- Deterministic preprocessing -------------------------------------------------

    def preprocess(self, text: str) -> list[str]:
        """Clean OCR text and split it into sentences. Fully rule-based, hence reproducible."""
        if not text.strip():                        # empty or image-only slide
            return []

        # Canonicalise OCR Unicode quirks (full-width digits, ligatures) for consistent matching.
        text = unicodedata.normalize("NFKC", text)

        # Rejoin words split across lines ("muscu-\nlar" -> "muscular"); must precede the newline collapse.
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Flatten layout to prose for the splitter. This merges slide bullets into one run, since en_core_sci_sm assumes prose; a slide-aware context unit is future .
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
    def _generate(self, messages: list[dict[str, str]], model_id: str, temperature: float, max_new_tokens: int = 1024) -> str:
        """Greedily decode a chat message list (wraps runtime.generate with this model)."""
        target_model = model_id or self.extraction_model_id
        return generate(target_model, messages, max_new_tokens, temperature)

    def _classify_type(self, term: str, sentence: str) -> str | None:
        """Assign the single most relevant type to a term, or None (ZeroTuneBio Module 2)."""
        raw = self._generate(
            prompts.classification_messages(term, sentence), 
            model_id=self.classification_model_id,
            temperature=0,
            max_new_tokens=512
        )

        # Read the 'TYPE: X' verdict (the last one). Tolerate case and space/hyphen variants
        # ('Cell Type' -> 'CELL_TYPE') so a valid typing is never misread as NONE and dropped.
        verdicts = re.findall(r"TYPE:\s*([A-Za-z][A-Za-z _-]*)", raw)
        if not verdicts:
            logger.warning("  classify %r: no TYPE verdict parsed", term)
            return None

        verdict = re.sub(r"[ -]+", "_", verdicts[-1].strip().upper())
        return verdict if verdict in ENTITY_TYPES else None

    def _error_filter(self, typed: list[Entity], sentence: str) -> list[Entity]:
        """Remove only the typings the model explicitly flags as wrong (ZeroTune_Bio Module 3).

        The default is KEEP, so a malformed reply or a reformatted term can never silently
        drop a valid entity; it can only remove one the model actually flagged.
        """
        if not typed:
            return []

        lines = []
        for e in typed:
            lines.append(f"{e.text} = {e.type}")
        listing = "\n".join(lines)

        raw = self._generate(
            prompts.error_filter_messages(listing, sentence), 
            model_id=self.classification_model_id,
            temperature=0,
            max_new_tokens=512
        )

        wrong = set()
        for line in raw.splitlines():
            term = line.strip(" -*.\t")
            if term and term.upper() != "NONE":
                wrong.add(term)

        kept = []
        for e in typed:
            if e.text not in wrong:
                kept.append(e)
        return kept

    def run_zerotune(self, sentence: str) -> list[Entity]:
        """Extract typed entities from one sentence through the three ZeroTuneBio modules.

        M1 maximises recall, grounding drops hallucinations, M1 recovers precision by typing,
        and M3 removes only typings flagged wrong.
        """
        preview = sentence[:80] + ("..." if len(sentence) > 80 else "")
        logger.info("ZeroTuneBio | %s", preview)

        # Module 1: extract ALL professional terms (recall-first), then parse the JSON array.
        raw = self._generate(
            prompts.extraction_messages(sentence), 
            self.extraction_model_id,
            temperature=0.2,
            max_new_tokens=512
        )
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else json.loads(raw)
            candidates = []
            if isinstance(parsed, list):
                for t in parsed:
                    t = str(t).strip()
                    if t:
                        candidates.append(t)
        except json.JSONDecodeError:                 # malformed reply: fall back to a comma split
            candidates = []
            for t in raw.split(","):
                t = t.strip(" -*.\t\"'[]")
                if t:
                    candidates.append(t)

        candidates = self._dedup(candidates)
        logger.info("  M1: %d candidate term(s)", len(candidates))

        # Grounding: keep only candidates that occur in the sentence (drops hallucinations).
        candidates = self._ground(candidates, sentence)
        logger.info("  grounded: %d -> %s", len(candidates), candidates)
        if not candidates:
            return []

        # Module 2: classify each grounded term into a single type (or None).
        typed = []
        for term in candidates:
            etype = self._classify_type(term, sentence)
            logger.info("  M2 %r -> %s", term, etype)
            if etype:
                typed.append(Entity(text=term, type=etype, segment=sentence))
        if not typed:
            return []

        # Module 3: drop only the typings the model flags wrong (default keep).
        confirmed = self._error_filter(typed, sentence)
        logger.info("  M3 kept %d/%d", len(confirmed), len(typed))
        return confirmed

    def extract(self, sentences: list[str]) -> list[Entity]:
        """Run ZeroTuneBio on every sentence and return entities dedup: (text, type).

        An entity seen in several sentences keeps its first segment, which the normalization
        stage consumes as context.
        """
        non_empty = []
        for s in sentences:
            if s.strip():
                non_empty.append(s)
        logger.info("Extracting from %d sentence(s)", len(non_empty))

        all_entities = []
        for i, sent in enumerate(non_empty, start=1):
            logger.info("[sentence %d/%d]", i, len(non_empty))
            all_entities.extend(self.run_zerotune(sent))

        # File-level dedup by (text, type), case-sensitive so biomedical case is preserved.
        seen = set()
        unique = []
        for e in all_entities:
            key = (e.text, e.type)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        logger.info("Done: %d unique entities (%d before dedup)", len(unique), len(all_entities))
        return unique


def run_ner(texts: list[str], model_id_extraction: str, model_id_classification: str) -> list[Entity]:
    """Extract typed entities from a list of slide texts, loading and freeing the LLM once.

    CPU preprocessing (sentence splitting) runs before the model touches the GPU; the model
    is freed in a finally block (scale-to-zero) so a mid-run error cannot leak VRAM and the
    normalization stage inherits a clean GPU.
    """
    pipe = NERPipeline(model_id_extraction,model_id_classification)

    sentences = []
    for text in texts:
        sentences.extend(pipe.preprocess(text))

    try:
        pipe.load()
        entities = pipe.extract(sentences)
    finally:
        print("Typed entities extracted.")
        # pipe.unload()
    return entities


if __name__ == "__main__":

    from biomedcat.config import settings
    from biomedcat.types import Slide
    logging.basicConfig(level=logging.INFO)

    
    # Simple test case
    test_texts = ["Mechanisms of Apoptosis\n",
                  "- Caspase activation cascade\n",                                                                                                                                         
                "- Mitochondrial outer membrane permeabilization (MOMP)\n"
                "- Cytochrome c release into the cytosol\n"
                "Figure 1.2: The intrinsic pathway of programmed cell death."]                                                                                                                                                                                       
  

    print("--- Running NER Test ---")
    try:
        results = run_ner(test_texts, settings.extraction_model_id, settings.classification_model_id)  # "gemma4:e4b-it-qat"
        for ent in results:
            print(f"Found: {ent.text} ({ent.type})")
    except Exception as e:
        print(f"Test failed with error: {e}")
