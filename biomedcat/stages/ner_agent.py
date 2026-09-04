import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import spacy
from spacy.lang.en import English

from biomedcat.runtime import generate
from biomedcat.types import Entity, ENTITY_TYPES
from biomedcat.stages.rag_engine import BiomedRAG
from biomedcat.config import settings
from biomedcat.stages.rag_engine import build_rag
# from biomedcat import prompts

logger = logging.getLogger(__name__)

def implements_verification(verdict: str) -> str:
    """Helper to ensure the verdict is valid."""
    return verdict if verdict in ENTITY_TYPES else "NONE"

class NERAgentPipeline:
    """
    Pipeline NER Agentique utilisant le pattern ReAct (Reasoning + Acting).
    L'agent utilise des outils RAG pour valider les types d'entités biomédical    """

    def __init__(
        self, 
        rag_unseen: BiomedRAG, 
        classification_model_id: str,
        sanitization_model_id: str
    ):
        self.rag_engine = rag_unseen
        self.classification_model_id = classification_model_id
        self.sanitization_model_id = sanitization_model_id
        self.max_agent_steps = 3  # a verdict is reached in 1-3 steps in practice; more only loops
        self.MAX_INPUT_LENGTH = 5000
        self.MAX_NEW_TOKENS = 700  # agent replies average ~500 characters; 5000 only let the model ramble
        self.current_sentence = None  # Pour stocker le contexte courant
        # Verdict memo per document: the same term ("FSHD", "DUX4") recurs in many sentences and
        # was re-classified every time. Keyed by the lowercased term; None is cached too so a
        # term that yields no verdict is not retried at every sentence.
        self._verdict_cache: Dict[str, Optional[str]] = {}
        # Case-insensitive map of the Biolink class vocabulary, for verdict normalization.
        self._type_by_norm = {self._norm_type(t): t for t in ENTITY_TYPES}
        
        # Load the sentence model once. Without scispaCy, a blank English pipeline with the
        # rule-based sentencizer keeps the stage functional (segmentation only, no POS tags).
        try:
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("Successfully loaded scispaCy model")
        except Exception as e:  # missing model (OSError) or a model built for another spaCy version
            logger.warning("scispaCy model en_core_sci_sm unavailable (%s): using spaCy's rule-based sentencizer instead.", type(e).__name__)
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    def _filter_and_deduplicate_candidates(self, candidates: List[str]) -> List[str]:
        """Filters and deduplicates extracted terms based on length and content."""
        final_candidates = []
        seen = set()
        
        for candidate in candidates:
            # Ignore words too short or purely numeric
            if len(candidate) < 2 or candidate.isdigit():
                continue

            # Check for at least one alphabetic character
            if not re.search(r'[a-zA-Z]', candidate):
                continue

            # Deduplication case-insensitive
            if candidate.lower() not in seen:
                seen.add(candidate.lower())
                final_candidates.append(candidate)

        return final_candidates


    # ---------------------------------------------------------------------------
    # SECURITY (Sentinel, Sanitizer & Output Guard)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _norm_type(name: str) -> str:
        """'biolink:GrossAnatomicalStructure', 'Gross Anatomical Structure' -> 'grossanatomicalstructure'."""
        name = (name or "").strip().replace("biolink:", "")
        return re.sub(r"[\s_\-]", "", name).lower()

    INJECTION_PATTERNS = [
        r"ignore (all|previous|prior) instructions",
        r"system override",
        r"forget your tools",
        r"new instructions",
        r"disregard (all|previous|prior)",
    ]

    def screen_document(self, text: str, page: int) -> bool:
        """Content screening of one slide with the sanitization model (LlamaGuard), once per slide.

        Runs only when settings.document_screening is on. The classifier judges content safety,
        not prompt injection: injection is handled structurally (read-only tools, whitelisted
        verdicts, neutralized control keywords). Fail-open on a classifier error, with a warning,
        so that a missing model never silently empties a document.
        """
        if not settings.document_screening or not self.sanitization_model_id or not text.strip():
            return True
        try:
            reply = generate(self.sanitization_model_id, [{"role": "user", "content": text[: self.MAX_INPUT_LENGTH]}], 50, 0).lower()
            if "unsafe" in reply:
                logger.warning("[SECURITY ALERT] slide %d flagged unsafe by %s: skipped", page, self.sanitization_model_id)
                return False
        except Exception as e:
            logger.error("[SECURITY] screening model error on slide %d (%s): continuing without screening", page, e)
        return True

    def _is_input_safe(self, term: str, sentence: str) -> Tuple[bool, str, str]:
        """Structural sanitization of one (term, sentence) pair; returns (ok, term, sentence).

        Nothing here rejects biology. Oversized inputs are refused; control characters are
        removed; the agent's own control keywords (ACTION:, FINAL_VERDICT:, OBSERVATION:) and
        the usual injection phrases are neutralized in place and logged, instead of dropping the
        term, because the agent only has read-only ontology tools and a whitelisted output: an
        injected slide can at worst mistype its own entities.
        """
        if len(term) > self.MAX_INPUT_LENGTH or len(sentence) > self.MAX_INPUT_LENGTH:
            logger.warning("[SECURITY] Input too large. Rejecting to prevent DoS.")
            return False, "", ""

        def clean_control_chars(text: str) -> str:
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        term_clean = clean_control_chars(term).strip()
        sentence_clean = clean_control_chars(sentence).strip()

        for kw in ("ACTION:", "FINAL_VERDICT:", "OBSERVATION:"):
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            if pattern.search(term_clean) or pattern.search(sentence_clean):
                logger.warning("[SECURITY] control keyword %r in input: neutralized", kw)
                term_clean = pattern.sub(kw[:-1] + " -", term_clean)
                sentence_clean = pattern.sub(kw[:-1] + " -", sentence_clean)
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, term_clean + " " + sentence_clean, re.IGNORECASE):
                logger.warning("[SECURITY] injection phrase %r in input: neutralized", pattern)
                term_clean = re.sub(pattern, "[redacted]", term_clean, flags=re.IGNORECASE)
                sentence_clean = re.sub(pattern, "[redacted]", sentence_clean, flags=re.IGNORECASE)
        if not term_clean:
            return False, "", ""
        return True, term_clean, sentence_clean

    def _validate_tool_argument(self, tool_name: str, arg: str) -> Tuple[bool, str, str]:
        """Clean and validate a tool argument; returns (ok, message, cleaned argument).

        The tools are read-only lookups in an in-memory ontology and a local vector index, so
        there is no path or command to protect: the checks only keep arguments short and map
        class names onto the vocabulary. Keyword-style arguments the model sometimes writes,
        class_name="treatment", are unwrapped.
        """
        arg = re.sub(r'^\s*\w+\s*=\s*', "", arg).strip().strip("'\"").strip()
        if not arg or len(arg) > 300:
            return False, "Empty or oversized argument.", arg

        if tool_name in ("lookup_exact_term", "semantic_context_search"):
            return True, "", arg
        if tool_name == "get_class_hierarchy":
            canonical = self._type_by_norm.get(self._norm_type(arg))
            if canonical is None:
                return False, f"Class '{arg}' is not a Biolink class; use semantic_context_search to find candidate classes.", arg
            return True, "", canonical
        return False, f"No validator defined for tool: {tool_name}", arg

    def _validate_output(self, response: str) -> Tuple[bool, Optional[str], str]:
        """Extract and normalize the FINAL_VERDICT; returns (ok, canonical class name, message).

        Only the verdict line is inspected: the model's reasoning legitimately contains ';',
        '--' or markdown. The verdict is matched to the Biolink vocabulary case-insensitively,
        with or without the biolink: prefix, spaced or CamelCase ('Gene', 'biolink:Gene',
        'gross anatomical structure', 'GrossAnatomicalStructure' all map to the same class).
        """
        verdict_match = re.search(r"FINAL_VERDICT:\s*([^\n]+)", response)
        if not verdict_match:
            return False, None, "No FINAL_VERDICT found in agent response."
        raw = verdict_match.group(1).strip().strip("'\"`*.:;,()[] \t").strip()
        if re.search(r"<script>|\$\{|\{\{|\$\(", raw, re.IGNORECASE):
            return False, None, f"Suspicious verdict line: {raw[:60]}"
        canonical = self._type_by_norm.get(self._norm_type(raw))
        if canonical is None:
            return False, None, f"Verdict '{raw[:60]}' is not a Biolink class."
        return True, canonical, ""

    # ---------------------------------------------------------------------------
    # TOOLS
    # ---------------------------------------------------------------------------

    def tool_lookup_exact_term(self, term: str) -> str:
        """Recherche directe dans le dictionnaire Biolink avec contexte."""
        logger.info(f"[Agent Tool] Lookup exact term: {term}")
        flat_data = self.rag_engine.flat_data
        if term in flat_data:
            entry = flat_data[term]
            return json.dumps({
                "found": True,
                "definition": entry["metadata"].get("definition", ""),
                "contextual_info": f"Term '{term}' found in biomedical context: {self.current_sentence}",
                "metadata": entry["metadata"]
            }, ensure_ascii=False)
        return json.dumps({"found": False, "message": "Term not found in exact lookup."})

    def tool_semantic_context_search(self, query: str) -> str:
        """Interroges le moteur Hybrid RAG (Dense + Sparse) avec contexte."""
        logger.info(f"[Agent Tool] Semantic search: {query}")
        # The query is the agent's description of the term; appending the whole sentence diluted
        # the embedding and returned unrelated classes.
        results = self.rag_engine.search(query, top_k=10)
        if not results:
            return "No relevant biomedical classes found."
        
        context_parts = []
        for res_id in results:
            entry = self.rag_engine.flat_data.get(res_id)
            if entry:
                meta = entry["metadata"]
                context_parts.append(f"Class: {res_id} | Def: {meta.get('definition', '')}")
        
        return "\n".join(context_parts)

    def tool_get_class_hierarchy(self, class_name: str) -> str:
        """Retourne l'arbre complet (parents/enfants/siblings) d'une classe."""
        logger.info(f"[Agent Tool] Get hierarchy for: {class_name}")
        flat_data = self.rag_engine.flat_data
        if class_name not in flat_data:
            return f"Class {class_name} not found."

        meta = flat_data[class_name]["metadata"]
        hierarchy = {
            "parent": meta.get("parent"),
            "ancestors": meta.get("ancestors", []),
            "children": meta.get("children", []), 
            "siblings": meta.get("siblings", []),
            "mixins": meta.get("mixins", [])
        }
        return json.dumps(hierarchy, ensure_ascii=False)

    def _get_tool_executor(self, tool_name: str):
        """Returns the appropriate executor function for a given tool name."""
        executors = {
            "lookup_exact_term": self.tool_lookup_exact_term,
            "semantic_context_search": self.tool_semantic_context_search,
            "get_class_hierarchy": self.tool_get_class_hierarchy,
        }
        return executors.get(tool_name)

    # ---------------------------------------------------------------------------
    # EXTRACTOR WITH MODEL
    # ---------------------------------------------------------------------------

    def _extract_candidates_model(self, sentence: str,context: str) -> List[str]:
        """Extraire les candidats biologiques avec le modèle de classification."""
        if not self.classification_model_id:
            logger.warning("Classification model ID not available, falling back to basic extraction")
            return []
            
        try:
            if len(context)>200:
                context = context[:200]
            # Prompt simplifié et plus clair
            prompt = f"""Extract biomedical terms from this sentence: "{sentence}".
            
            Return ONLY a pipe-separated list of terms. 
            Examples: "insulin|diabetes|heart failure"
            Do NOT include any explanation or extra text.
            Focus on ANY biomedical entities independently of their information content that are pertinent to the context.
            For compound terms like "(2R)-2-aminopropanoic acid", keep them together if they are part of the same concept (e.g. modifier, etc.).
            Ignore common words like "the", "and", "with", "for", "of", "in", "on", "at", "by", "to", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can"."""
            
            messages = [{"role": "user", "content": f"CONTEXT:{context}"},{"role": "user", "content": prompt}]
            response = generate(self.classification_model_id, messages, 800, 0.0)
            
            # Parser la réponse pour extraire les termes bruts
            raw_candidates = []
            if response and "|" in response:
                raw_candidates = [term.strip() for term in response.split("|") if term.strip()]
            elif response:
                raw_candidates = [response.strip()]

            raw_candidates = self._filter_and_deduplicate_candidates(raw_candidates)
            raw_candidates = [(r,sentence) for r in raw_candidates]
            
            # Use shared utility for filtering and deduplication
            return raw_candidates
        except Exception as e:
            logger.warning(f"Model-based extraction failed: {e}")
            # Fallback to the raw spaCy tokens, then apply common filter/dedup
            raw_fallback = self._get_raw_spaCy_tokens(sentence)
            raw_fallback = self._filter_and_deduplicate_candidates(raw_fallback)
            raw_fallback = [(r,sentence) for r in raw_fallback]
            return raw_fallback

    def _get_raw_spaCy_tokens(self, sentence: str) -> List[str]:
        """Helper to return raw SpaCy tokens (pre-filtering/pre-dedup)."""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(sentence)
            # Without POS tags (blank pipeline) keep every alphabetic token longer than 2 characters.
            raw_candidates = [
                token.text for token in doc
                if len(token.text) > 2 and (token.pos_ in ["NOUN", "PROPN", "ADJ"] if token.pos_ else token.is_alpha)
            ]
            return raw_candidates
        except Exception as e:
            logger.warning(f"Raw spaCy extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------------
    # AGENT CORE LOGIC (ReAct Loop)
    # ---------------------------------------------------------------------------

    def _agent_system_prompt(self) -> str:
        
        return (
            """You are a Biomedical Ontology Agent. Your goal is to classify a term into the correct
            Biolink Entity Type by looking at the most probable type definition and ist children and parents definitions. 
            For terms with multiple words, you may need to break it down into two different concepts
            to capture the most information content. For example alcohol dependence will be translated
            into : alcohol (small molecule) and dependence (disease).
            If the term is a verb, transform it into a noun. For example, 'treat's' will be transformed
            into treatment and keep it ONLY if it has high informative content.
            You have access to three specialized tools that you MUST use.
            TOOLS:
            1. semantic_context_search(description): the index contains class DEFINITIONS, not entity
            names, so pass a short description of what the term IS, not the term itself. Good:
            semantic_context_search(a human gene encoding a transcription factor). Bad: semantic_context_search(DUX4).
            2. get_class_hierarchy(class_name): parents, children and siblings of a class, to choose
            between a class and its neighbours (e.g. 'protein' versus 'protein isoform').
            3. lookup_exact_term(class_name): definition of one class whose exact name you already know.
            PROCESS:
            You have at most 3 steps. Step 1 is always semantic_context_search with a description.
            Choose the most specific class the term is an instance of, but never a more specific
            class than the evidence supports (a plain protein is 'protein', not 'protein isoform').
            As soon as a tool result names a class that fits, answer.
            For each step, output a short 'THOUGHT' and then ONE 'ACTION' in the format:
            ACTION: tool_name(argument)
            When you know the type, end your response with exactly this line, using the class name
            exactly as returned by the tools (lowercase, with spaces):
            FINAL_VERDICT: <class name>"""
        )

    def _run_agentic_loop(self, term: str, sentence: str) -> Optional[str]:
        """The ReAct loop: Thought -> Action -> Observation."""
        
        is_safe, term_clean, sentence_int = self._is_input_safe(term, sentence)
        if not is_safe:
            return None

        cache_key = term_clean.lower()
        if cache_key in self._verdict_cache:
            logger.info("[Agent] %r: verdict reused from cache (%s)", term_clean, self._verdict_cache[cache_key])
            return self._verdict_cache[cache_key]

        system_prompt = self._agent_system_prompt().format(sentence=sentence_int)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {sentence_int}\nTerm to classify: {term_clean}"}
        ]

        verdict: Optional[str] = None
        for step in range(self.max_agent_steps):
            response = generate(self.classification_model_id, messages, self.MAX_NEW_TOKENS, 0.0)
            messages.append({"role": "assistant", "content": response})
            logger.info(f"[Agent Step {step+1}] Response: {response}")

            is_valid, final_verdict, error_msg = self._validate_output(response)
            if is_valid:
                verdict = final_verdict
                break
            if "FINAL_VERDICT" in response.upper():
                # An unrecognized class name: tell the agent once, let it correct itself.
                logger.warning("[Agent] verdict rejected for %r: %s", term_clean, error_msg)
                messages.append({"role": "user", "content": f"OBSERVATION: {error_msg} Answer with FINAL_VERDICT: <exact Biolink class name>."})
                continue

            action_match = re.search(r"ACTION:\s*(\w+)\((.*)\)", response)
            if action_match:
                tool_name = action_match.group(1)
                is_valid, error_msg, arg_str = self._validate_tool_argument(tool_name, action_match.group(2))
                if not is_valid:
                    observation = f"Error: {error_msg}"
                    logger.warning("[Agent] invalid tool call %s(%s): %s", tool_name, arg_str, error_msg)
                else:
                    # --- TOOL EXECUTION DELEGATION ---
                    executor = self._get_tool_executor(tool_name)
                    if executor:
                        try:
                            observation = executor(arg_str) 
                        except Exception as e:
                            observation = f"Error executing tool: {str(e)}"
                    else:
                        observation = f"Error: Unknown tool {tool_name}"

                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            else:
                logger.warning("Agent failed to provide an ACTION or FINAL_VERDICT.")
                break

        self._verdict_cache[cache_key] = verdict
        return verdict

    def extract(self, text: List[str]) -> List[Entity]:
        """Main entry point for the NER Agent."""
        all_entities = []
        for block in text:
            if not block.strip():
                continue

            # Segmenter le bloc de texte en phrases individuelles
            try:
                doc = self.nlp(block)
                sentences = [sent.text for sent in doc.sents]
            except Exception as e:
                logger.warning(f"Could not segment block into sentences: {e}")
                continue

            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                self.current_sentence = sentence 
                
                # 1. Extraction Phase (M1 - Recall)
                candidates = self._extract_candidates_model(sentence,block)

                if not candidates:
                    continue
                else:
                    # 2. Agentic Classification & Verification Phase (M2)
                    for term,sentence in candidates:
                        final_type = self._run_agentic_loop(term, sentence)
                        
                        if final_type:
                            all_entities.append(Entity(text=term, type=implements_verification(final_type), segment=sentence))

        return all_entities

def run_ner_agent(texts: List[str], model: str = settings.classification_model_id, rag: Optional[BiomedRAG] = None) -> List[Entity]:
    """Run the NER agent on one text block per slide and return typed entities with slide provenance.

    `texts[i]` is the OCR output of slide i+1. The RAG engine is built here only when the caller
    did not pass one, so importing this module never triggers an index build.
    """
    if rag is None:
        rag = build_rag()

    agent = NERAgentPipeline(
        rag_unseen=rag,
        classification_model_id=model,
        sanitization_model_id=settings.sanitization_model_id,
    )

    entities: List[Entity] = []
    for page, text in enumerate(texts, start=1):
        if not text or not text.strip():
            continue
        logger.info("NER on slide %d: %r", page, text[:150])
        if not agent.screen_document(text, page):
            continue
        try:
            for entity in agent.extract([text]):
                entity.page = page
                entities.append(entity)
        except Exception as e:
            logger.exception("NER failed on slide %d: %s", page, e)

    return entities


if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # 1. Setup Environment
    print("--- Initializing Agent Test Environment ---")
    rag = build_rag()
    agent = NERAgentPipeline(
        rag_unseen=rag,
        classification_model_id=settings.classification_model_id,
        sanitization_model_id=settings.sanitization_model_id
    )

    # 2. Load QA Data
    qa_file = Path("tests/qa_data.json")
    if not qa_file.exists():
        print(f"Error: Test file {qa_file} not found.")
    else:
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        print(f"--- Running QA Data Test ({len(qa_data)} queries) ---")
        passed = 0
        failed = 0

        for entry in qa_data:
            query = entry["query"]
            expected = entry.get("expected_class", "UNKNOWN").upper()
            
            print(f"\nTesting Query: '{query}' (Expected: {expected})")
            
            try:
                results = agent.extract([query])
                # Vérifier si le terme attendu est présent dans les résultats
                found_match = False
                for e_res in results:
                    if e_res:
                        # Accepter le terme brut ou le terme avec parenthèses
                        # Exemple : "CACNA1C" ou "CACNA1C (gene)"
                        if expected in e_res.text.upper() or e_res.text.upper() == expected:
                            # Vérifier que le type est correct
                            expected_type = entry.get("expected_class", "").upper()
                            if expected_type == "" or e_res.type.upper() == expected_type:
                                found_match = True
                                break
                
                if found_match:
                    print(f"  [PASS] Found match: {[e.text for e in results]}")
                    passed += 1
                else:
                    found_types = [f"{e.text} ({e.type})" for e in results]
                    print(f"  [FAIL] No match found. Extracted: {found_types}")
                    failed += 1
            except Exception as e:
                print(f"  [ERROR] Test execution failed: {e}")
                failed += 1

        print("\n" + "="*30)
        print("      FINAL TEST SUMMARY")
        print("="*30)
        print(f"Total Queries: {len(qa_data)}")
        print(f"Passed:        {passed}")
        print(f"Failed:        {failed}")
        print("="*30)
