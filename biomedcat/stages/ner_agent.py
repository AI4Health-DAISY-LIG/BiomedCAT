import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import spacy
from spacy.lang.en import English

from biomedcat.runtime import generate
from biomedcat.types import Entity, ENTITY_TYPES
from biomedcat.stages.rag_engine import BiomedRAG
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
        # On supprime les caractères non-imprimables (0x00-0x1F sauf \n, \r, \t et 0x7F)
        def clean_control_chars(text: str) -> str:
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        term_clean = clean_control_chars(term).strip()
        sentence_clean = clean_control_chars(sentence).strip()

        # 3. Protection de la structure (Structural Integrity)
        # Interdiction d'injecter les délimiteurs du prompt ReAct
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
            # On interdit tout ce qui pourrait être interprété comme un chemin ou une commande
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
        
        results = self.rag_engine.search(enhanced_query, top_k=10) #### TO BE REVIEWED BASED ON TESTING
        if not results:
            return "No relevant biological classes found."
        
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

    # ---------------------------------------------------------------------------
    # EXTRACTOR WITH SPACY
    # ---------------------------------------------------------------------------

    def _extract_candidates_spacy(self, sentence: str) -> List[str]:
        """Extraire les candidats avec en_core_sci_sm en conservant les termes composés."""
        if not self.nlp:
            logger.warning("SpaCy model not available, falling back to basic extraction")
            return []
            
        try:
            doc = self.nlp(sentence)
            candidates = []
            
            # Extraire les tokens avec des règles plus souples
            for token in doc:
                # Garder les tokens significatifs (noms, noms propres, adjectifs)
                if token.pos_ in ["NOUN", "PROPN", "ADJ"] and len(token.text) > 2:
                    # Ne pas inclure les mots courants qui ne sont pas des termes biologiques
                    if token.text.lower() not in ["the", "and", "with", "for", "of", "in", "on", "at", "by", "to"]:
                        candidates.append(token.text)
            
            # Pour les termes composés, on va essayer de les regrouper intelligemment
            # Mais garder une approche simple : ne pas diviser les termes qui sont dans le dictionnaire
            final_candidates = []
            i = 0
            while i < len(candidates):
                # Essayer de construire des termes composés à partir de plusieurs tokens
                # Si on peut trouver un terme composé dans le dictionnaire, on le prend
                found_compound = False
                # Tester les combinaisons de 1 à 3 tokens
                for length in range(min(3, len(candidates) - i), 0, -1):
                    combined = " ".join(candidates[i:i+length])
                    # Vérifier si le terme combiné est dans le dictionnaire ou est un terme commun
                    if combined.lower() in self.rag_engine.flat_data or \
                       combined.lower() in ["heart failure", "diabetes mellitus", "blood pressure"]:
                        final_candidates.append(combined)
                        i += length
                        found_compound = True
                        break
                
                if not found_compound:
                    final_candidates.append(candidates[i])
                    i += 1
                    
            return list(set(final_candidates))  # Remove duplicates
        except Exception as e:
            logger.warning(f"SpaCy extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------------
    # AGENT CORE LOGIC (ReAct Loop)
    # ---------------------------------------------------------------------------

    def _agent_system_prompt(self) -> str:
        return (
            "You are a Biomedical Ontology Agent. Your goal is to classify a term into the correct "
            "Biolink Entity Type. You have access to three specialized tools.\n\n"
            "CONTEXT: The term must be classified based on its usage within the full sentence context: '{{sentence}}'.\n\n"
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
            response = generate(self.classification_model_id, messages, 1024, 0)
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

            # Check for Action call
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
                    # Proceed with execution only if valid
                    try:
                        if tool_name == "lookup_exact_term":
                            observation = self.tool_lookup_exact_term(arg_str)
                        elif tool_name == "semantic_context_search":
                            observation = self.tool_semantic_context_search(arg_str)
                        elif tool_name == "get_class_hierarchy":
                            observation = self.tool_get_class_hierarchy(arg_str)
                        else:
                            observation = f"Error: Unknown tool {tool_name}"
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"

                messages.append({"role": "user", "annotated_content": f"OBSERVATION: {observation}"})
            else:
                logger.warning("Agent failed to provide an ACTION or FINAL_VERDICT.")
                break

        return None

    def extract(self, sentences: List[str]) -> List[Entity]:
        """Main entry point for the NER Agent."""
        all_entities = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            logger.info(f"Processing sentence: {sentence[:50]}...")
            self.current_sentence = sentence  # Stocker le contexte
            
            # 1. Extraction Phase (M1 - Recall) - Utiliser SpaCy au lieu de LLM
            candidates = self._extract_candidates_spacy(sentence)

            if not candidates:
                continue

            # 2. Agentic Classification & Verification Phase (M2)
            for term in candidates:
                # Vérifier si le terme est trop court ou invalide
                if len(term.strip()) < 2:
                    continue
                    
                final_type = self._run_agentic_loop(term, sentence)
                
                if final_type:
                    all_entities.append(Entity(text=term, type=implements_verification(final_type), segment=sentence))

        return all_entities

if __name__ == "__main__":
    import json
    from pathlib import Path
    from biomedcat.config import settings
    from biomedcat.stages.rag_engine import build_rag

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
