import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import json
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
import en_core_sci_sm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

logger = logging.getLogger(__name__)


def free_gpu():
    # release Python garbage, then return PyTorch's cached VRAM to the driver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# The 7 biomedical entity types the pipeline can assign.
ENTITY_TYPES = [
    "GENE",
    "DISEASE",
    "CHEMICAL",
    "CELL_TYPE",
    "ANATOMY",
    "CHROMOSOMAL_LOCUS",
    "EPIGENETIC_MODIFICATION",
]


# System (role) prompts for the LLM stages.
EXTRACTOR_SYSTEM = (  # M1: broad term extraction
    "You are a biomedical text analyst. "
    "Extract professional biomedical terms from the given text."
)

JUDGE_SYSTEM = "You are a strict biomedical annotation expert."  # M2 + classifier

VERIFIER_SYSTEM = "You are a strict biomedical annotation verifier."  # M3


class NERPipeline:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None  # heavy LLM loaded lazily in load_model()
        self.model = None
        self.nlp = en_core_sci_sm.load()  # scispaCy (CPU, light): sentence splitting only

    def load_model(self):
        logger.info(f"Loading tokenizer and model: {self.model_id}")



        # Hugging Face login via token file to avoid rate limits
        hf_token_path = Path("hf_token.txt")
        if hf_token_path.exists():
            login(token=hf_token_path.read_text().strip())


        # Load the model with 4-bit quantization using bitsandbytes, in which the model weights are stored in 4-bit precision to reduce memory usage and speed up inference.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Enable 4-bit quantization
            bnb_4bit_quant_type="nf4", # Use NormalFloat4 (NF4) quantization for better accuracy
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation to balance speed and precision
            bnb_4bit_use_double_quant=True, # Use double quantization to further reduce memory usage while maintaining accuracy
        )

        free_gpu() # Clear GPU memory before loading the model to avoid out-of-memory errors


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id) # Load the tokenizer associated with the model for text preprocessing and tokenization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config, # apply the 4-bit quantization configuration to the model
            dtype=torch.bfloat16, # Use bfloat16 for model weights to reduce memory usage while maintaining precision
            device_map="auto", # Automatically distribute the model across available devices
        ).eval() # Set the model to evaluation mode, which disables dropout and other training-specific behaviors for inference




        logger.info(f"Model loaded on {next(self.model.parameters()).device}") # Log the device on which the model is loaded (e.g., CPU or GPU) for debugging and monitoring purposes


        if torch.cuda.is_available():
            logger.info(f"VRAM allocated after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB") # Log the amount of GPU memory allocated after loading the model to monitor resource usage and detect potential memory issues

    def unload(self): # Unload the model and tokenizer from memory to free up resources, especially GPU memory, when they are no longer needed.
        self.model = None
        self.tokenizer = None
        free_gpu()

    def preprocess(self, text: str) -> List[str]:
        """Clean OCR text and return a list of sentences via scispaCy.

        Fully deterministic (string ops + rule-based splitter), unlike the LLM stages.
        Each returned sentence is the per-unit context fed to run_zerotune().
        """
        if not text.strip():  # image-only / empty slide -> nothing to process
            return []
    

        # NFKC: canonicalise OCR Unicode quirks (full-width digits, ligatures) for consistent matching.
        text = unicodedata.normalize("NFKC", text)

        # Rejoin words split across lines ("muscu-\nlar" -> "muscular"). Must precede the newline collapse.
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Flatten layout to prose for the splitter.
        # TODO(refine): merges slide bullets into one sentence; make slide-aware once we have real slide text + a gold set.

        text = re.sub(r"\n+", " ", text)
        text = re.sub(r" {2,}", " ", text)  # collapse repeated spaces
        text = text.strip()
        doc = self.nlp(text)  # scispaCy en_core_sci_sm: used ONLY for sentence segmentation
        # keep each non-empty sentence, stripped of surrounding whitespace


        return [sent.text.strip() for sent in doc.sents if sent.text.strip()] # Return a list of non-empty sentences extracted from the input text, each stripped of leading and trailing whitespace, to be used as context for further processing in the NER pipeline.

    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024) -> str:


        """Run greedy decoding on a chat message list and return the generated text."""

        
        # Format messages into Llama's chat token sequence; return_dict -> input_ids + attention_mask.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # append the "assistant's turn" tokens so the model replies
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)  # move inputs to the GPU where the model lives

        input_len = inputs["input_ids"].shape[1]  # prompt length, used to strip the prompt off the output
        with torch.inference_mode():  # no gradients: faster + less memory for inference
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy (temp 0): deterministic argmax decoding
                pad_token_id=self.tokenizer.eos_token_id,  # Llama has no pad token; reuse EOS
            )

        generated = output_ids[0, input_len:]  # keep only the newly generated tokens
        logger.debug(f"_generate: {input_len} prompt tokens -> {generated.shape[0]} new tokens")
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()  # tokens -> clean text
    


    def _ground(self, terms: List[str], sentence: str) -> List[str]:
        """ZeroTuneBio Module 3 Task 1 ('determine whether the terms exist in the
        text'), enforced deterministically. Keep only terms that occur in the sentence on
        alphanumeric word boundaries (NFKC-normalised, CASE-SENSITIVE to preserve biomedical
        case such as 'Co'/'CO'; aligns with M1's 'exactly as it appears' rule)."""
        # normalise the sentence once: NFKC + collapsed whitespace (case preserved)
        sent_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", sentence))
        grounded = []
        for t in terms:
            # normalise the candidate the same way so the comparison is consistent
            t_norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip()
            if not t_norm:
                continue
            # word-boundary match (no alphanumeric neighbour) so "DNA" won't match inside "DNase"
            pattern = r"(?<![A-Za-z0-9])" + re.escape(t_norm) + r"(?![A-Za-z0-9])"
            if re.search(pattern, sent_norm):  # keep only if it actually appears in the text
                grounded.append(t)  # keep the original surface form, not the normalised one
        return grounded

    def _dedup(self, terms: List[str]) -> List[str]:
        """Deterministically remove duplicate terms (NFKC + whitespace-normalised, CASE-SENSITIVE
        so biomedical case like 'Co'/'CO' is preserved), keeping the first surface form."""
        seen = set()
        out = []
        for t in terms:
            key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(t.strip())
        return out

    def _classify_type(self, entity: str, sentence: str) -> Optional[str]:
        """Module 2 (Rule Judgement): explain the entity in context, then assign its single most
        relevant type (or None) by assessing it against the type criteria."""
        type_defs = (  # short definition per type, handed to the model as the criteria
            "GENE (a gene or gene symbol), DISEASE (a disease or disorder), "
            "CHEMICAL (a chemical, drug, or compound), CELL_TYPE (a type of cell), "
            "ANATOMY (an anatomical structure, tissue, or organ), "
            "CHROMOSOMAL_LOCUS (a chromosomal location, locus, or genomic region), "
            "EPIGENETIC_MODIFICATION (an epigenetic modification such as methylation)"
        )
        raw = self._generate([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": (
                f"Sentence: {sentence}\n\n"
                f'Entity: "{entity}"\n\n'
                f'Step 1: Explain what "{entity}" means in the context of this sentence.\n'
                "Step 2: Based on that meaning, choose the SINGLE most relevant type, "
                f"or NONE if it does not clearly belong to any of:\n{type_defs}.\n"
                "End your answer with a final line in exactly this form:\n"
                "TYPE: <one type or NONE>"
            )},
        ], max_new_tokens=512)
        # parse the 'TYPE: X' verdict (take the last); keep only if it is a valid type
        verdicts = re.findall(r"TYPE:\s*([A-Z_]+)", raw)
        if not verdicts:
            logger.warning(f"  classify '{entity}': no TYPE verdict parsed")
            return None
        verdict = verdicts[-1].strip()
        return verdict if verdict in ENTITY_TYPES else None

    def _error_filter(self, typed: List[Dict[str, str]], sentence: str) -> List[Dict[str, str]]:
        """Module 3 (Error Filtering): re-examine Module 2's (term -> type) typings and remove ONLY
        the ones the model explicitly flags as wrong. Default is KEEP, so a malformed reply or a
        reformatted term can never silently drop a valid entity (it can only remove a flagged error)."""
        if not typed:
            return []
        listing = "\n".join(f"{e['text']} = {e['type']}" for e in typed)
        raw = self._generate([
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": (
                f"Sentence: {sentence}\n\n"
                f"Each line proposes a biomedical entity and its type:\n{listing}\n\n"
                "Re-examine each line in the context of the sentence. List ONLY the terms whose type "
                "is WRONG, one term per line, exactly as written. If every typing is correct, reply NONE."
            )},
        ])
        wrong = set()
        for line in raw.splitlines():
            term = line.strip(" -*.\t")
            if term and term.upper() != "NONE":
                wrong.add(term)
        return [e for e in typed if e["text"] not in wrong]  # default keep; drop only flagged errors

    def run_zerotune(self, sentence: str) -> List[Dict[str, str]]:
        preview = sentence[:80] + ("..." if len(sentence) > 80 else "")
        logger.info(f"ZeroTuneBio | {preview}")
        # Module 1 (report 5.4): identify ALL professional terms WITHOUT filtering -> maximise recall.
        # Precision is recovered by M2 (rule judgement) + M3 (error filtering).
        m1_raw = self._generate([
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": (
                "Identify ALL biomedical professional terms and concepts mentioned in the text below.\n"
                "Do NOT filter or judge them -- list every professional term to maximise recall.\n"
                "Ensure that multi-word concepts (composite terms) are extracted as a single complete phrase.\n"
                "Return ONLY a valid JSON array of strings, exactly as they appear in the text.\n\n"
                "Text: The TP53 gene mutation is common in non-small cell lung cancer.\nTerms:"
            )},
            {"role": "assistant", "content": "[\"TP53 gene mutation\", \"non-small cell lung cancer\"]"},
            {"role": "user", "content": f"Text: {sentence}\nTerms:"}
        ], max_new_tokens=512)
        
        try:
            match = re.search(r'\[.*\]', m1_raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(m1_raw)
            if isinstance(parsed, list):
                candidates = [str(t).strip() for t in parsed if str(t).strip()]
            else:
                candidates = []
        except json.JSONDecodeError:
            candidates = [t.strip(" -*.\t\"'[]") for t in m1_raw.split(",")]
            candidates = [t for t in candidates if t]


            
        candidates = self._dedup(candidates)
        logger.info(f"  M1: {len(candidates)} distinct candidate term(s)")
        # Module 3 Task 1 (applied early): keep only candidates that exist in the text.
        candidates = self._ground(candidates, sentence)
        logger.info(f"  grounded: {len(candidates)} term(s): {candidates}")
        if not candidates:
            logger.info("  no grounded candidates, skipping sentence")
            return []

        # Module 2 (Rule Judgement): explain each grounded term in context and assign its single
        # most relevant type, or NONE. Runs over EVERY grounded term, so the judge sees every entity.
        typed = []
        for cand in candidates:
            etype = self._classify_type(cand, sentence)
            logger.info(f"  M2 '{cand}' -> {etype}")
            if etype:
                typed.append({"text": cand, "type": etype})
        if not typed:
            return []

        # Module 3 (Error Filtering): re-examine M2's typings against the text, keep the ones that hold.
        confirmed = self._error_filter(typed, sentence)
        logger.info(f"  M3 kept {len(confirmed)}/{len(typed)} -> {[e['text'] for e in confirmed]}")
        return confirmed

    def extract(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Run ZeroTuneBio on every sentence and return deduplicated entities."""
        non_empty = [s for s in sentences if s.strip()]
        logger.info(f"Extracting entities from {len(non_empty)} sentence(s)")
        all_entities = []
        for i, sent in enumerate(non_empty, start=1):
            logger.info(f"[sentence {i}/{len(non_empty)}]")
            all_entities.extend(self.run_zerotune(sent))
        # deduplicate by (text, type) -- CASE-SENSITIVE to preserve biomedical case ('Co' != 'CO')
        seen = set()
        unique = []
        for e in all_entities:
            key = (e["text"], e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        logger.info(f"Done: {len(unique)} unique entities ({len(all_entities)} before dedup)")
        return unique


if __name__ == "__main__":
    # smoke test: run one FSHD sentence end-to-end (load -> preprocess -> extract -> print)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sentence = (
        "In FSHD, hypomethylation of the D4Z4 macrosatellite repeat on chromosome 4q35 "
        "permits aberrant expression of the DUX4 transcription factor in skeletal muscle."
    )
    ner = NERPipeline()
    ner.load_model()
    sents = ner.preprocess(sentence)
    logger.info(f"{len(sents)} sentence(s) after preprocessing")
    ents = ner.extract(sents)
    for e in ents:
        print(f"  {e['type']:<24} {e['text']}")
    ner.unload()
