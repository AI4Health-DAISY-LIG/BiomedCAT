import os
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

AGENT_TERSE=1 # asks for telegraphic THOUGHT lines (fewer generated tokens per step).

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

        # Second pass for hard terms (AGENT_SECOND_PASS=1): when the first episode ends without a
        # verdict, or only through the closing turn, or after using every step, the term is run
        # again with a wider candidate list (AGENT_SECOND_PASS_TOP_K), more steps
        # (AGENT_SECOND_PASS_STEPS) and, optionally, the model's hidden reasoning switched on with
        # its own token budget (AGENT_THINK_BUDGET, added to MAX_NEW_TOKENS). Measured 12 Sept 2026.
        self.second_pass = os.getenv("AGENT_SECOND_PASS", "0") in ("1", "true", "yes")
        self.second_pass_top_k = int(os.getenv("AGENT_SECOND_PASS_TOP_K", "20"))
        self.second_pass_steps = int(os.getenv("AGENT_SECOND_PASS_STEPS", "5"))
        self.think_budget = int(os.getenv("AGENT_THINK_BUDGET", "0"))      # > 0: think on in the second pass
        self.think_first_pass = os.getenv("AGENT_THINK_FIRST_PASS", "0") in ("1", "true", "yes")

        self.terse = os.getenv("AGENT_TERSE", "0") in ("1", "true", "yes")
        self._episode_top_k: Optional[int] = None

        # "Search then decide" mode (AGENT_MODE=decide, measured 14-16 Sept 2026: 0.467 against
        # 0.407 for the ReAct loop on the 300 reference terms). No tool loop: one retrieval of the
        # raw term, the top candidates plus the is_a neighbourhood (children, siblings, parent) of
        # the first ones, and ONE decision call with the model's hidden reasoning on. The
        # candidate list is capped because selection degrades above ~30 entries (D20N 0.430).
        self.mode = os.getenv("AGENT_MODE", "react").strip().lower()
        self.decide_top_k = int(os.getenv("AGENT_DECIDE_TOP_K", "10"))
        self.decide_neighbours_of = int(os.getenv("AGENT_DECIDE_NEIGHBOURS_OF", "3"))
        self.decide_cap = int(os.getenv("AGENT_DECIDE_CAP", "30"))
        # Experiment (20 Sept 2026): drop the Biolink grouping classes ("X or Y") from the
        # neighbourhood; the decide mode was measured to fall back on them (biological process
        # or activity absorbed 10/65 verdicts of the FSHD deck, physiological process 0.40 -> 0.05
        # on the real benchmark). Retrieved candidates are never dropped, only added neighbours.
        self.decide_no_grouping = os.getenv("AGENT_DECIDE_NO_GROUPING", "0").strip().lower() in ("1", "true", "yes")
        self.decide_search_depth = int(os.getenv("AGENT_DECIDE_SEARCH_DEPTH", "20"))   # fusion depth of the retrieval
        self.decide_num_predict = int(os.getenv("AGENT_DECIDE_NUM_PREDICT", "1500"))   # hidden reasoning + one verdict line
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
        # SCISPACY=0 reproduces the published configuration (benchmark and document runs of
        # 5-10 Sept 2026, before the model was installed): rule-based sentencizer here, basic
        # tokenizer and no BM25 leg in the retriever.
        try:
            if os.getenv("SCISPACY", "1") in ("0", "off", "false"):
                raise OSError("disabled by SCISPACY=0")
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
        # Tolerate the variants small models produce: **FINAL_VERDICT:**, FINAL VERDICT -, Final_Verdict.
        # The LAST occurrence is the verdict: the reasoning may quote the rule ("...answer with
        # the FINAL_VERDICT line") before the actual verdict line.
        verdict_matches = re.findall(r"FINAL[_ ]VERDICT\**\s*[:\-]?\s*\**\s*([^\n]+)", response, re.IGNORECASE)
        if not verdict_matches:
            return False, None, "No FINAL_VERDICT found in agent response."
        raw = verdict_matches[-1].strip().strip("'\"`*.:;,()[] \t").strip()
        if re.search(r"<script>|\$\{|\{\{|\$\(", raw, re.IGNORECASE):
            return False, None, f"Suspicious verdict line: {raw[:60]}"
        if raw.lower() in ("none", "null", "n/a", "not applicable"):
            return True, "NONE", ""
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
        # Candidate list size shown to the agent (AGENT_SEARCH_TOP_K, default 10).
        top_k = self._episode_top_k or int(os.getenv("AGENT_SEARCH_TOP_K", "10"))
        results = self.rag_engine.search(query, top_k=top_k)
        # The raw term is searched too: since the exemplar index holds mentions, the surface
        # form often retrieves the class directly; both rankings are fused by reciprocal rank.
        term = getattr(self, "current_term", None)
        if term and term.strip().lower() != query.strip().lower():
            fused: Dict[str, float] = {}
            for ranking in (results, self.rag_engine.search(term, top_k=top_k)):
                for rank, c in enumerate(ranking):
                    fused[c] = fused.get(c, 0.0) + 1.0 / (60 + rank)
            results = [c for c, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:top_k]
        if not results:
            return "No relevant biomedical classes found."

        assignable = set(self._type_by_norm.values())
        # Classes shown to the agent during this term; the closing turn is constrained to them.
        self._seen_classes.extend(c for c in results if c in assignable and c not in self._seen_classes)
        context_parts = []
        for i, res_id in enumerate(results):
            entry = self.rag_engine.flat_data.get(res_id)
            if entry:
                meta = entry["metadata"]
                parent = meta.get("parent") or "named thing"
                definition = (meta.get("definition") or "").strip().replace("\n", " ")[:220]
                line = f"Class: {res_id} (a kind of {parent}) | Def: {definition}"
                # Neighbourhood of the top hits, so the decision rule (most specific class whose
                # definition holds, else the parent) can be applied without another tool call.
                children = [c for c in (meta.get("children") or []) if c in assignable]
                if i < 3 and children:
                    line += " | Children: " + ", ".join(children[:12])
                context_parts.append(line)

        return "\n".join(context_parts)

    def _describe(self, name: str) -> str:
        """One line per class: name, short definition, flags for nodes that cannot be a verdict."""
        entry = self.rag_engine.flat_data.get(name)
        if not entry:
            return f"- {name}"
        meta = entry["metadata"]
        definition = (meta.get("definition") or "").strip().replace("\n", " ")
        flag = " [abstract, not assignable]" if meta.get("abstract") else (" [deprecated, not assignable]" if meta.get("deprecated") else "")
        return f"- {name}{flag}: {definition[:220]}" if definition else f"- {name}{flag}"

    def tool_get_class_hierarchy(self, class_name: str) -> str:
        """The comparable neighbourhood of a class: parent, children and siblings, each with its definition.

        This is the tool for the navigation step: after a candidate class is found, the agent
        compares it with its parent (is the candidate too specific?), its children (is there a
        more specific class that still holds?) and its siblings (is a neighbour a better fit?).
        """
        logger.info(f"[Agent Tool] Get hierarchy for: {class_name}")
        flat_data = self.rag_engine.flat_data
        if class_name not in flat_data:
            return f"Class {class_name} not found."

        meta = flat_data[class_name]["metadata"]
        parent = meta.get("parent")
        lines = [f"CLASS {self._describe(class_name)[2:]}"]
        lines.append("PARENT (more general):")
        lines.append(self._describe(parent) if parent else "- none")
        children = meta.get("children") or []
        lines.append(f"CHILDREN (more specific, {len(children)}):")
        lines.extend(self._describe(c) for c in children[:15]) if children else lines.append("- none")
        siblings = meta.get("siblings") or []
        lines.append(f"SIBLINGS (alternatives under the same parent, {len(siblings)}):")
        lines.extend(self._describe(s) for s in siblings[:15]) if siblings else lines.append("- none")
        mixins = meta.get("mixins")
        if mixins:
            lines.append(f"MIXINS: {mixins if isinstance(mixins, str) else ', '.join(mixins)}")
        assignable = set(self._type_by_norm.values())
        for c in [class_name, parent] + list(children[:15]) + list(siblings[:15]):
            if c and c in assignable and c not in self._seen_classes:
                self._seen_classes.append(c)
        return "\n".join(lines)

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
            1. semantic_context_search(description): pass a short description of what the term IS
            (the term itself is searched as well). Good: semantic_context_search(a human gene encoding
            a transcription factor). Each result comes with its parent and its child classes.
            2. get_class_hierarchy(class_name): parents, children and siblings of a class, to choose
            between a class and its neighbours (e.g. 'protein' versus 'protein isoform').
            3. lookup_exact_term(class_name): definition of one class whose exact name you already know.
            PRIVACY: never classify information about identifiable people: person names, patient or
            sample identifiers, dates of birth, ages, addresses, hospitals, contact details. For such a
            term answer immediately with FINAL_VERDICT: none.
            UNKNOWN TERMS: if you do not recognise the term as a real biomedical entity, gene, disease,
            molecule, process, structure, from your own knowledge, do not guess a class from its
            spelling; answer FINAL_VERDICT: none.
            PROCESS:
            You have at most 3 steps. Step 1 is always semantic_context_search with a description.
            DECISION RULE: among the classes returned, take the most specific class whose definition
            holds for the term; if none of the child classes holds, answer the parent class. Never a
            more specific class than the evidence supports (a plain protein is 'protein', not
            'protein isoform'); never a class only because it sounds like the term.
            As soon as a tool result names a class that fits, STOP calling tools and answer with the
            FINAL_VERDICT line: do not verify a class you already recognised.
            Otherwise output a short 'THOUGHT' and then ONE 'ACTION' in the format:
            ACTION: tool_name(argument)
            When you know the type, end your response with exactly this line, using the class name
            exactly as returned by the tools (lowercase, with spaces):
            FINAL_VERDICT: <class name>"""
            + self._tree_skeleton_block()
        )

    def _tree_skeleton_block(self) -> str:
        """The whole Biolink class tree, names only, indented, mixins in brackets (~600 tokens).

        Enabled with AGENT_TREE_SKELETON=1: the agent then sees every parent, sibling and child
        without a navigation call, so its steps go to searching and deciding. Abstract and
        deprecated classes are shown with a marker, since they cannot be a verdict.
        """
        if os.getenv("AGENT_TREE_SKELETON", "0") not in ("1", "true", "yes"):
            return ""
        if getattr(self, "_skeleton_cache", None) is None:
            flat = self.rag_engine.flat_data
            roots = [c for c, v in flat.items() if not v["metadata"].get("parent") or v["metadata"]["parent"] not in flat]
            lines: List[str] = []

            def walk(name: str, depth: int) -> None:
                meta = flat[name]["metadata"]
                mark = "" if name in self._type_by_norm.values() else " (not assignable)"
                mixins = meta.get("mixins")
                if mixins:
                    mixins = mixins if isinstance(mixins, str) else ", ".join(mixins)
                    mark += f" [{mixins}]"
                lines.append("  " * depth + name + mark)
                for child in sorted(meta.get("children") or []):
                    if child in flat:
                        walk(child, depth + 1)

            for r in sorted(roots):
                walk(r, 0)
            self._skeleton_cache = ("\n\nCLASS TREE (every Biolink class, indented by parent; mixins in brackets; "
                                    "definitions come from the tools):\n" + "\n".join(lines))
        return self._skeleton_cache

    # ---------------------------------------------------------------------------
    # "Search then decide" mode
    # ---------------------------------------------------------------------------

    def _neighbours(self, class_name: str) -> List[str]:
        """is_a neighbourhood of a class: its children, its siblings (the other children of its
        is_a parent) and the parent, assignable classes only. Mixin groups are deliberately not
        included (no effect measured, 16 Sept 2026)."""
        flat = self.rag_engine.flat_data
        meta = lambda c: flat.get(c, {}).get("metadata", {})
        ok = lambda c: c in flat and self.rag_engine._indexable(meta(c))
        out = [x for x in (meta(class_name).get("children") or []) if ok(x)]
        parent = meta(class_name).get("parent")
        if parent:
            out += [x for x in (meta(parent).get("children") or []) if x != class_name and ok(x)]
            if ok(parent):
                out.append(parent)
        if self.decide_no_grouping:
            out = [x for x in out if " or " not in x]
        return out

    def _decide_candidates(self, term: str) -> List[str]:
        """Top-k retrieval of the raw term (fused at `decide_search_depth`), then the neighbourhood
        of the first `decide_neighbours_of` candidates, deduplicated, capped at `decide_cap`."""
        top = self.rag_engine.search(term, top_k=max(self.decide_search_depth, self.decide_top_k)) or []
        cands = list(top[:self.decide_top_k])
        for c in top[:self.decide_neighbours_of]:
            for x in self._neighbours(c):
                if x not in cands:
                    cands.append(x)
        return cands[:self.decide_cap]

    # Not used by the decision menu (see _decide_prompt): a sentence / word-boundary cut was tried
    # on 20 Sept 2026 and measured worse than the hard 150-character cut (0.417 vs 0.470 on the
    # 300 reference terms). Kept for experiments (output/decide_prompt_v2_test.py).
    DECIDE_DEFINITION_CHARS = 250

    @classmethod
    def _short_definition(cls, definition: str, limit: Optional[int] = None) -> str:
        """First sentence of a Biolink definition when it fits in `limit` characters, else the
        longest prefix that ends on a word boundary, marked with an ellipsis. Whitespace is folded.
        Experimental helper, not used by the published decide prompt."""
        limit = cls.DECIDE_DEFINITION_CHARS if limit is None else limit
        text = " ".join(definition.split())
        if len(text) <= limit:
            return text
        head = text[:limit]
        end = head.rfind(". ")
        if end > 0:
            return head[:end + 1]
        cut = head.rfind(" ")
        return (head[:cut] if cut > 0 else head).rstrip(",;:") + "…"

    def _decide_prompt(self, term: str, sentence: str, candidates: List[str]) -> str:
        """One decision prompt: the candidate list with definitions, the mention, its sentence when
        one exists (document context; on the benchmark the sentence is the term itself), the
        privacy rule of the ReAct prompt, the decision rule, and the verdict line format."""
        flat = self.rag_engine.flat_data
        lines = []
        for c in candidates:
            # Hard cut at 150 characters, exactly as measured (300 terms 0.470, 1,356 terms 0.486 x2,
            # FSHD deck 0.517, real benchmark 0.371). Do not "improve" it without re-measuring: the
            # word-boundary cut at 250 characters (_short_definition) alone cost 5.3 points on the
            # 300 reference terms (0.417, 20 Sept 2026), the menu being 9% longer.
            definition = (flat.get(c, {}).get("metadata", {}).get("definition") or "")[:150]
            lines.append(f"- {c}: {definition}")
        menu = "\n".join(lines)
        has_context = bool(sentence) and sentence.strip().lower() != term.strip().lower()
        context = f"Sentence: {sentence}\n" if has_context else ""
        privacy = ("Never classify information about identifiable people (person names, patient or sample "
                   "identifiers, dates, ages, addresses, contact details): answer none for such a mention.\n") if has_context else ""
        return (f"Classify the biomedical mention into exactly one Biolink class from this list, or answer none if it is "
                f"not a biomedical entity.\n{menu}\n\n{context}Mention: {term}\n{privacy}"
                "Decision rule: choose the most specific class the evidence supports; if none of the child classes holds, "
                "answer the parent class; never a more specific class than the evidence supports.\n"
                "Reply with one line: FINAL_VERDICT: <class name>")

    def _run_decide(self, term_clean: str, sentence_int: str) -> Optional[str]:
        """One call, no tools. The verdict must be one of the candidates (or none); a verdict naming
        another class is recovered from the candidate names cited in the reply, else no verdict."""
        candidates = self._decide_candidates(term_clean)
        self._seen_classes = list(candidates)
        if not candidates:
            logger.warning("[Agent Decide] %r: no candidate from the retriever", term_clean)
            return None
        messages = [{"role": "user", "content": self._decide_prompt(term_clean, sentence_int, candidates)}]
        response = generate(self.classification_model_id, messages, self.decide_num_predict, 0.0, think=True, raw=True)
        logger.info(f"[Agent Decide] {term_clean!r}: {len(candidates)} candidates | Response: {response}")
        is_valid, verdict, error_msg = self._validate_output(response)
        if is_valid and (verdict == "NONE" or verdict in candidates):
            return verdict
        # Recovery: the reply names one of the candidates (longest match wins), as in the closing turn.
        lowered = response.lower()
        named = [c for c in candidates if c.lower() in lowered]
        if named:
            verdict = max(named, key=len)
            logger.info("[Agent Decide] verdict recovered from the reply for %r: %s", term_clean, verdict)
            return verdict
        logger.warning("[Agent Decide] no admissible verdict for %r: %s", term_clean, error_msg or f"'{verdict}' is not a candidate")
        return None

    def _run_agentic_loop(self, term: str, sentence: str) -> Optional[str]:
        """The ReAct loop: Thought -> Action -> Observation."""
        
        is_safe, term_clean, sentence_int = self._is_input_safe(term, sentence)
        if not is_safe:
            return None

        cache_key = term_clean.lower()
        if cache_key in self._verdict_cache:
            logger.info("[Agent] %r: verdict reused from cache (%s)", term_clean, self._verdict_cache[cache_key])
            return self._verdict_cache[cache_key]

        self.current_term = term_clean
        if self.mode == "decide":
            verdict = self._run_decide(term_clean, sentence_int)
            self._verdict_cache[cache_key] = verdict
            return verdict
        verdict, hard = self._run_episode(term_clean, sentence_int, self.max_agent_steps, None,
                                          self.think_first_pass and self.think_budget > 0)
        if self.second_pass and hard:
            logger.info("[Agent] second pass for %r (first pass: %s)", term_clean, verdict)
            second, _ = self._run_episode(term_clean, sentence_int, self.second_pass_steps, self.second_pass_top_k,
                                          self.think_budget > 0)
            if second is not None:
                verdict = second
        self._verdict_cache[cache_key] = verdict
        return verdict

    def _run_episode(self, term_clean: str, sentence_int: str, max_steps: int, top_k: Optional[int],
                     think: bool) -> tuple[Optional[str], bool]:
        """One ReAct episode. Returns (verdict, hard): `hard` is True when the episode ended without
        a verdict, through the closing turn only, or after using every step."""
        self._episode_top_k = top_k
        budget = self.MAX_NEW_TOKENS + (self.think_budget if think else 0)
        self._seen_classes: List[str] = []
        system_prompt = self._agent_system_prompt().format(sentence=sentence_int)
        if self.terse:
            system_prompt += "\nKeep every THOUGHT to one short telegraphic line (at most 25 words)."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {sentence_int}\nTerm to classify: {term_clean}"}
        ]

        verdict: Optional[str] = None
        steps_used = 0
        for step in range(max_steps):
            steps_used = step + 1
            response = generate(self.classification_model_id, messages, budget, 0.0, think=think or None)
            messages.append({"role": "assistant", "content": response})
            logger.info(f"[Agent Step {step+1}] Response: {response}")

            is_valid, final_verdict, error_msg = self._validate_output(response)
            if is_valid:
                verdict = final_verdict
                break
            if re.search(r"FINAL[_ ]VERDICT", response, re.IGNORECASE):
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

        closing_turn = False
        if verdict is None and len(messages) > 2:
            # The model often knows the class after one or two observations but keeps calling
            # tools; one forced closing turn, no tools allowed, recovers those verdicts cheaply.
            # Constrained decision: only the classes seen during this term are eligible, and the
            # reply is one line with no reasoning (a THOUGHT here used to eat the whole budget).
            options = ", ".join(self._seen_classes[:40]) or "none"
            messages.append({"role": "user", "content": (
                "No more tool calls and no THOUGHT. Apply the decision rule to the classes you have seen: "
                f"{options}. Reply with exactly one line and nothing else: FINAL_VERDICT: <one of these class names>, "
                "or FINAL_VERDICT: none if the term is not a biomedical entity.")})
            response = generate(self.classification_model_id, messages, 60, 0.0)
            logger.info(f"[Agent Final] Response: {response}")
            closing_turn = True
            is_valid, final_verdict, error_msg = self._validate_output(response)
            if is_valid:
                verdict = final_verdict
            else:
                # Last resort: the reply names a seen class without the verdict line.
                lowered = response.lower()
                named = [c for c in self._seen_classes if c.lower() in lowered]
                if named:
                    verdict = max(named, key=len)
                    logger.info("[Agent] verdict recovered from the closing reply for %r: %s", term_clean, verdict)
                else:
                    logger.warning("[Agent] no verdict for %r after the closing turn: %s", term_clean, error_msg)

        self._episode_top_k = None
        hard = verdict is None or closing_turn or steps_used >= max_steps
        return verdict, hard

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
                        # "NONE" is the agent's explicit abstention (not biomedical, or personal data).
                        if final_type and final_type != "NONE":
                            all_entities.append(Entity(text=term, type=implements_verification(final_type), segment=sentence))

        # The typing model has finished its work for this document: release it before the
        # existence gate loads the second model family (llama3:8b), so the two never coexist.
        from biomedcat.runtime import unload_models
        unload_models(self.classification_model_id, self.sanitization_model_id)
        keep = self.existence_gate([e.text for e in all_entities])
        return [e for e in all_entities if keep.get(e.text.strip().lower(), True)]

    # ---------------------------------------------------------------------------
    # EXISTENCE GATE (anti-invention), once per document
    # ---------------------------------------------------------------------------

    def existence_gate(self, terms: List[str]) -> Dict[str, bool]:
        """Lowercased term -> keep? The classifier invents entities from spelling; a term is
        dropped when the Name Resolver (production instance) has no candidate at all ("nameres",
        legacy alias "es") and/or when a model of another family does not recognise it ("llm"),
        per settings.existence_gate.
        Batched so the second model is loaded once per document, not once per term."""
        mode = "nameres" if settings.existence_gate == "es" else settings.existence_gate
        if settings.resolvers_offline and mode in ("nameres", "union"):
            # Offline mode: no term may leave the machine, so the resolver leg is dropped and
            # only the local second-family model remains (nothing at all for "nameres").
            mode = "llm" if mode == "union" else "off"
            logger.warning("[Gate] offline mode: the name-resolver check is skipped, gate mode is now %r", mode)
        # Keyed by the lowercased term, looked up with its original surface form (case matters
        # to the resolver: DUX4, not dux4).
        surface = {t.strip().lower(): t.strip() for t in terms if t.strip()}
        uniq = sorted(surface)
        keep = {t: True for t in uniq}
        if mode == "off" or not uniq:
            return keep
        if mode in ("nameres", "union"):
            for t in uniq:
                if not self._nameres_has_candidate(surface[t]):
                    keep[t] = False
        if mode in ("llm", "union"):
            for t in self._llm_unknown_terms([surface[t] for t in uniq if keep[t]]):
                keep[t] = False
        rejected = [t for t in uniq if not keep[t]]
        logger.info("[Gate %s] %d/%d terms rejected: %s", mode, len(rejected), len(uniq), rejected)
        return keep

    def _nameres_has_candidate(self, term: str) -> bool:
        """True when the Name Resolver returns at least one candidate (any type, no type filter:
        the question is whether the string names anything at all).
        A failed request keeps the term: the gate must never reject on a network error."""
        from biomedcat.retrieval import _fetch_json
        data = _fetch_json(settings.gate_nameres_url, {"string": term, "limit": 3})
        return True if data is None else bool(data)

    def _llm_unknown_terms(self, terms: List[str], batch: int = 40) -> List[str]:
        """Terms the existence model does not recognise as real biomedical entities."""
        unknown: List[str] = []
        for start in range(0, len(terms), batch):
            chunk = terms[start:start + batch]
            prompt = (
                "You are a strict biomedical fact checker. For each term below, say whether it names a "
                "real, existing biomedical entity (gene, protein, drug, chemical, disease, phenotype, "
                "anatomical structure, organism, process, procedure, device, measurement, or a standard "
                "biomedical concept) that you know from the literature or from databases. Common medical "
                "or biological words count as real. Invented, garbled or non-biomedical strings are not.\n"
                'Answer with a JSON object only: {"unknown": [<terms that are NOT real>]}.\n'
                "Terms:\n" + "\n".join(f"- {t}" for t in chunk)
            )
            reply = generate(settings.existence_model_id, [{"role": "user", "content": prompt}], 600, 0.0)
            m = re.search(r"\{.*\}", reply, re.DOTALL)
            try:
                listed = json.loads(m.group(0)).get("unknown", []) if m else []
            except (json.JSONDecodeError, AttributeError):
                logger.warning("[Gate llm] unparsable reply, batch kept: %s", reply[:120])
                continue
            lower = {t.lower() for t in chunk}
            unknown.extend(str(u).strip().lower() for u in listed if str(u).strip().lower() in lower)
        return unknown

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
