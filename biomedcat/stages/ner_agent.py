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
        self.max_agent_steps = 5  # Limite pour éviter les boucles infinies
        self.MAX_INPUT_LENGTH = 5000
        self.current_sentence = None  # Pour stocker le contexte courant
        
        # Charger le modèle SpaCy une seule fois
        try:
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("Successfully loaded scispaCy model")
        except OSError:
            logger.warning("scispaCy model not found. Falling back to basic tokenizer.")
            self.nlp = None

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

            # Regex check for standard biomedical/chemical characters
            valid_chars = r'^[a-zA-Z0-9\-\.()\[\]_]+$'
            if re.match(valid_chars, candidate) or re.match(r'^[a-zA-ZÀ-ÿ0-9\-\.()\[\]_]+$', candidate):
                # Deduplication case-insensitive
                if candidate.lower() not in seen:
                    seen.add(candidate.lower())
                    final_candidates.append(candidate)

        return final_candidates


    # ---------------------------------------------------------------------------
    # SECURITY (Sentinel, Sanitizer & Output Guard)
    # ---------------------------------------------------------------------------

    def _is_input_safe(self, term: str, sentence: str) -> Tuple[bool, str, str]:
        """
        Phase 1 (Sanitizer): Vérification structurelle et nettoyage rapide.
        Phase 2 (Sentinel): Analyse sémantique via LlamaGuard.
        
        Retourne: (is_safe, sanitized_term, sanitized_sentence)
        """
        # --- PHASE 1: SANITIZER (Local Regex/String) ---
        
        # 1. Validation de la taille (DoS Protection)
        if len(term) > self.MAX_INPUT_LENGTH or len(sentence) > self.MAX_INPUT_LENGTH:
            logger.warning("[SECURITY] Input too large. Rejecting to prevent DoS.")
            return False, "", ""

        # 2. Nettoyage des caractères de contrôle uniquement (Preserve scientific symbols)
        def clean_control_chars(text: str) -> str:
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        term_clean = clean_control_chars(term).strip()
        sentence_clean = clean_control_chars(sentence).strip()

        # 3. Protection de la structure (Structural Integrity)
        forbidden_keywords = ["ACTION:", "FINAL_VERDICT:", "OBSERVATION:"]
        for kw in forbidden_keywords:
            if kw in term_clean.upper() or kw in sentence_clean.upper():
                logger.warning(f"[SECURITY] Forbidden keyword '{kw}' detected in input!")
                return False, "", ""

        # 4. Détection de patterns d'injection (Pattern Matching)
        injection_patterns = [
            r"ignore all instructions",
            r"system override",
            r"forget your tools",
            r"new instructions",
            r"disregard previous"
        ]
        for pattern in injection_patterns:
            if re.search(pattern, term_clean + " " + sentence_clean, re.IGNORECASE):
                logger.warning(f"[SECURITY] Injection pattern '{pattern}' detected!")
                return False, "", ""

        # --- PHASE 2: SENTINEL (LlamaGuard) ---
        content = f"Sentence: {sentence_clean}\nTerm: {term_clean}"
        messages = [{"role": "user", "content": content}]
        try:
            response = generate(self.sanitization_model_id, messages, 50, 0).lower()
            if "unsafe" in response:
                logger.warning(f"[SECURITY ALERT] LlamaGuard flagged input as UNSAFE! Term: '{term_clean}'")
                return False, "", ""
            return True, term_clean, sentence_clean
        except Exception as e:
            # En cas d'erreur du modèle de sécurité, on adopme une approche "Fail-Closed" (on refuse)
            logger.error(f"[SECURITY ERROR] Error during sanitization: {e}")
            return False, "", ""

    def _validate_tool_argument(self, tool_name: str, arg: str) -> Tuple[bool, str]:
        """
        Couche de Sandboxing : Validation stricte des arguments extraits par l'agent.
        Empêche le path traversal et l'exécution d'arguments non autorisés.
        """
        # 1. Protection globale contre le path traversal (interdiction de . et /)
        if any(char in arg for char in [".", "/", "\\"]):
            return False, "Security Violation: Path traversal characters (., /, \\) are forbidden."

        # 2. Validation spécifique par outil
        if tool_name == "lookup_exact_term":
            # Autorise uniquement alphanumérique et symboles biologiques de base
            if not re.match(r"^[a-zA-Z0-9\s\+\-\(\)\_\!]+$", arg):
                return False, "Invalid characters in term. Only alphanumeric and biological symbols allowed."
            return True, ""

        elif tool_name == "get_class_hierarchy":
            # Whitelist : La classe doit exister dans l'ontologie chargée
            if arg not in self.rag_engine.flat_data:
                return False, f"Class '{arg}' not an authorized class."
            return True, ""

        elif tool_name == "semantic_context_search":
            # Pour la recherche sémantique, on est plus permissif mais on garde la protection path traversal ci-dessus
            return True, ""

        return False, f"No validator defined for tool: {tool_name}"

    def _validate_output(self, response: str) -> Tuple[bool, Optional[str], str]:
        """
        Phase 4 (Output Guard): Validation de la réponse finale de l'agent.
        Vérifie l'absence d'injection et la validité du verdict par rapport à la whitelist.
        """
        # 1. Contrôle de la structure (Protection contre injection SQL/NoSQL/Template)
        suspicious_patterns = [
            r";", r"--", r"/\*", r"\*/",  # SQL comments / multi-line
            r"DROP\s+", r"DELETE\s+", r"UPDATE\s+", # Destructive commands
            r"\$\{", r"\$\(", r"\{\{", # Template injection (Jinja/Mustache)
            r"###", r"<html>", r"<script>"  # Markdown/Format injection protection
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False, None, f"Suspicious pattern detected in agent output: {pattern}"

        # 2. Extraction et vérification de la Whitelist (Verdict Validation)
        verdict_match = re.search(r"FINAL_VERDICT:\s*([A-Za-z0-9_ ]+)", response)
        if not verdict_match:
            return False, None, "No FINAL_VERDICT found in agent response."

        verint = verdict_match.group(1).strip()
        if verint not in ENTITY_TYPES:
            return False, None, f"Verdict '{verint}' is not a valid entity type (Whitelist violation)."

        return True, verint, ""

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
        # Ajouter le contexte de la phrase à la requête
        enhanced_query = f"{query} (context: {self.current_sentence})" if self.current_sentence else query
        
        results = self.rag_engine.search(enhanced_query, top_k=10) 
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

    def _extract_candidates_model(self, sentence: str) -> List[str]:
        """Extraire les candidats biologiques avec le modèle de classification."""
        if not self.classification_model_id:
            logger.warning("Classification model ID not available, falling back to basic extraction")
            return []
            
        try:
            # Prompt simplifié et plus clair
            prompt = f"""Extract biomedical terms from this sentence: "{sentence}"
            
            Return ONLY a pipe-separated list of terms. 
            Examples: "insulin|diabetes|heart failure"
            Do NOT include any explanation or extra text.
            Focus on ANY biological entities independently of their information content.
            For compound terms like "(2R)-2-aminopropanoic acid", keep them together.
            Ignore common words like "the", "and", "with", "for", "of", "in", "on", "at", "by", "to", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can".

            Terms (pipe separated): """
            
            messages = [{"role": "user", "content": prompt}]
            response = generate(self.classification_model_id, messages, 500, 0.0)
            
            # Parser la réponse pour extraire les termes bruts
            raw_candidates = []
            if response and "|" in response:
                raw_candidates = [term.strip() for term in response.split("|") if term.strip()]
            elif response:
                raw_candidates = [response.strip()]
            
            # Use shared utility for filtering and deduplication
            return self._filter_and_deduplicate_candidates(raw_candidates)
        except Exception as e:
            logger.warning(f"Model-based extraction failed: {e}")
            # Fallback to the raw spaCy tokens, then apply common filter/dedup
            raw_fallback = self._get_raw_spaCy_tokens(sentence) 
            return self._filter_and_deduplicate_candidates(raw_fallback)

    def _get_raw_spaCy_tokens(self, sentence: str) -> List[str]:
        """Helper to return raw SpaCy tokens (pre-filtering/pre-dedup)."""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(sentence)
            raw_candidates = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and len(token.text) > 2]
            return raw_candidates
        except Exception as e:
            logger.warning(f"Raw spaCy extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------------
    # AGENT CORE LOGIC (ReAct Loop)
    # ---------------------------------------------------------------------------

    def _agent_system_prompt(self) -> str:
        
        return (
            "You are a Biomedical Ontology Agent. Your goal is to classify a term into the correct "
            "Biolink Entity Type. \n\n"
            "For terms with multiple words, you may need to break it down into two different concepts "
            "to capture the most information content. For example alcohol dependence will be translated "
            "into : alcohol (small molecule) and dependence (disease).\n\n"
            "If the term is a verb, transform it into a noun. For example, 'treats' will be transformed "
            "into treatment and keep it ONLY if it has high informative content.\n\n"
            " You have access to three specialized tools.\n\n"
            "TOOLS:\n"
            "1. lookup_exact_term(term): Use this for specific terms like 'TP53'.\n"
            "2. semantic_context_search(query): Use this for fuzzy concepts or when unsure.\n"
            "3. get_class_hierarchy(class_name): Use this to see parents, children, and siblings "
            "to verify if a term fits a category.\n\n"
            "PROCESS:\n"
            "For each step, you must output your 'THOUGHT' (reasoning) and then an 'ACTION' in the format:\n"
            "ACTION: tool_name(argument)\n\n"
            "When you are certain of the type, end your response with exactly:\n"
            "FINAL_VERDICT: <TYPE>\n\n"
            "Available Types: " + ", ".join(ENTITY_TYPES)
        )

    def _run_agentic_loop(self, term: str, sentence: str) -> Optional[str]:
        """The ReAct loop: Thought -> Action -> Observation."""
        
        # --- SECURITY CHECK (Sanitizer + Sentinel) ---
        is_safe, term_clean, sentence_int = self._is_input_safe(term, sentence)
        if not is_safe:
            return None

        # Créer le prompt système avec le contexte
        system_prompt = self._agent_system_prompt().format(sentence=sentence_int)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {sentence_int}\nTerm to classify: {term_clean}"}
        ]

        for step in range(self.max_agent_steps):
            response = generate(self.classification_model_id, messages, 5000, 0)
            messages.append({"role": "assistant", "content": response})
            
            logger.info(f"[Agent Step {step+1}] Response: {response}")

            # On utilise le validateur comme unique point d'entrée pour interpréter la fin du cycle
            is_valid, final_verdict, error_msg = self._validate_output(response)
            if is_valid:
                return final_verdict
            
            # Si un verdict a été tenté mais est invalide (Security Alert)
            if "FINAL_VERDICT" in response.upper():
                logger.warning(f"[SECURITY ALERT] Agent output failed validation: {error_msg}")
                return None

            action_match = re.search(r"ACTION:\s*(\w+)\((.*)\)", response)
            if action_match:
                tool_name = action_match.group(1)
                arg_str = action_match.group(2).strip().strip("'").strip('"')

                # --- SANDBOXING LAYER: Argument Validation ---
                is_valid, error_msg = self._validate_tool_argument(tool_name, arg_str)
                if not is_valid:
                    observation = f"Error: {error_msg}"
                    logger.warning(f"[SECURITY ALERT] Agent attempted invalid tool call: {tool_name}({arg_str}) -> {error_msg}")
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

                messages.append({"role": "user", "annotated_content": f"OBSERVATION: {observation}"})
            else:
                logger.warning("Agent failed to provide an ACTION or FINAL_VERDICT.")
                break

        return None

    def extract(self, sentences: List[str]) -> List[Entity]:
        """Main entry point for the NER Agent."""
        all_entities = []
        
        # for sentence in sentences:
        #     if not sentence.strip():
        #         continue
            
        #     logger.info(f"Processing sentence: {sentence[:100]}...")
        #     self.current_sentence = sentence  # Stocker le contexte
            
        # 1. Extraction Phase (M1 - Recall)
        candidates = self._extract_candidates_model(sentences)

        if not candidates:
            # continue
            return []
        else:
            # 2. Agentic Classification & Verification Phase (M2)
            for term in candidates:
                final_type = self._run_agentic_loop(term, sentences)
                
                if final_type:
                    all_entities.append(Entity(text=term, type=implements_verification(final_type), segment=sentences))

        return all_entities

def run_ner_agent(texts, model=settings.classification_model_id, rag=build_rag()):
    logging.basicConfig(level=logging.INFO)

    # 1. Setup Environment
    agent = NERAgentPipeline(
        rag_unseen=rag,
        classification_model_id=model,
        sanitization_model_id=settings.sanitization_model_id
    )

    entities = []
    for s,entry in enumerate(texts):

        print(f"\n Slide {s+1} description: '{entry[1:150]}' ")
        
        try:
            results = agent.extract([entry])
            
            
            entities.append(Entity(text='', type='',segment=''))


        except Exception as e:
            print(f"  [ERROR] NER failed: {e}")

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
