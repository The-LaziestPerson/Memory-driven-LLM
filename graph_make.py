#!/usr/bin/env python3

import argparse
import json
import os
import requests

# HF cache defaults (adjust if you want)
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")

import re
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Set

# Optional HF imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    BitsAndBytesConfig = None

# PEFT (LoRA / QLoRA) support (optional)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    PEFT_AVAILABLE = False

# Policy imports
try:
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception:
    joblib = None
    np = None
    LogisticRegression = None

# -----------------------------
# Configurable thresholds / hyperparams (tweak carefully)
# -----------------------------
MIN_PREDICATE_OVERLAP = 0.35   # for fallback contradiction detection vs ancestor
PRESERVATION_TOKEN_OVERLAP = 0.35  # candidate must share at least this fraction of tokens with ancestor set
MAX_TOPIC_DRIFT_SIM_DROP = 0.4  # if candidate is much less similar to ancestor than loser, it's drift
NEW_NODE_SUPPORT_PRIOR = 0.08   # initial support weight for newly added nodes
POLICY_RETRAIN_EVERY = 5
RETRAIN_EVERY = POLICY_RETRAIN_EVERY
GROUND_TRUTH_SIM_THRESHOLD = 0.7  # Similarity threshold used when matching GT paragraphs to nodes
NLI_DEFAULT_THRESHOLD = 0.7
# -----------------------------

# -----------------------------
# Utilities (unchanged)
# -----------------------------
def clean(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip())

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and any stray <think> tokens."""
    if not text:
        return text
    # remove full blocks like <think>...</think> (dotall)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # remove any remaining tags
    text = re.sub(r"</?think\s*/?>", "", text, flags=re.IGNORECASE)
    # also remove common chain-of-thought markers e.g. "[THOUGHT]" if any (conservative)
    text = re.sub(r"\[think\].*?\[\/think\]", "", text, flags=re.DOTALL | re.IGNORECASE)
    return clean(text)


def toks_lower(s: str):
    return re.findall(r"\w+", (s or "").lower())

def tokens(text: str) -> List[str]:
    return re.findall(r"\w{3,}", (text or "").lower())

def token_set(text: str) -> Set[str]:
    return set(tokens(text))

def symmetric_similarity(a: str, b: str) -> float:
    """Symmetric token-overlap: average of overlap relative to a and b."""
    if not a or not b:
        return 0.0
    at = token_set(a)
    bt = token_set(b)
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    ra = inter / max(1, len(at))
    rb = inter / max(1, len(bt))
    return (ra + rb) / 2.0

def simple_similarity(a: str, b: str) -> float:
    """Backward-compatible asymmetric similarity (kept for some heuristics)."""
    if not a or not b:
        return 0.0
    at = token_set(a)
    bt = token_set(b)
    if not at:
        return 0.0
    return len(at & bt) / len(at)

def has_negation(text: str) -> bool:
    negs = [" not ", " not,", " can't ", " cannot ", " cant ", " don't ", " don't,", " doesn't ", " never ", " no "]
    t = " " + (text or "").lower() + " "
    return any(n in t for n in negs)

def predicate_tokens(text: str) -> Set[str]:
    # tokens with length >= 4 as proxy for predicates/entities
    return set(re.findall(r"\w{4,}", (text or "").lower()))

# -----------------------------
# Policy loader & features (unchanged)
# -----------------------------
POLICY_PATH = "policy_clf.joblib"
policy = None  # (clf, columns)
POLICY_META = "policy_meta.json"
RETRAIN_EVERY = POLICY_RETRAIN_EVERY

CAUSAL = set(["cause","causes","caused","lead","leads","led","result","results","enable","enables","make","makes","force","forces"])
HEDGE = set(["may","might","could","possible","likely","suggest","appear","seems","tends"])
ABSOL = set(["all","every","always","never","none","no","completely","entirely"])

def extract_features(text: str) -> Dict[str, float]:
    t = toks_lower(text or "")
    c = {}
    for w in t:
        c[w] = c.get(w, 0) + 1
    feats = {}
    feats["len"] = len(t)
    feats["uniq"] = len(set(t))
    feats["causal_cnt"] = sum(c.get(w, 0) for w in CAUSAL)
    feats["hedge_cnt"] = sum(c.get(w, 0) for w in HEDGE)
    feats["absol_cnt"] = sum(c.get(w, 0) for w in ABSOL)
    feats["neg"] = int(any(w in ["not","no","never","cannot","cant","dont","doesnt","isnt"] for w in t))
    # unigram features (limited to top 50 by order of occurrence)
    seen = []
    for w in t:
        if w not in seen:
            seen.append(w)
        if len(seen) >= 50:
            break
    for w in seen:
        feats["u_"+w] = c.get(w, 0)
    # bigram features (first 50 unique bigrams)
    bigrams = []
    for i in range(len(t)-1):
        bg = f"{t[i]}_{t[i+1]}"
        if bg not in bigrams:
            bigrams.append(bg)
        if len(bigrams) >= 50:
            break
    for b in bigrams:
        feats["b_"+b] = bigrams.count(b)
    return feats

def try_load_policy():
    global policy
    if joblib is None or np is None:
        print("[policy] joblib/numpy not available; skipping policy load.")
        policy = None
        return
    if not os.path.exists(POLICY_PATH):
        print(f"[policy] {POLICY_PATH} not found — running without policy.")
        policy = None
        return
    try:
        pkg = joblib.load(POLICY_PATH)
        clf = pkg.get("clf")
        cols = pkg.get("columns")
        if clf is None or cols is None:
            print("[policy] model file missing expected keys; ignoring policy.")
            policy = None
            return
        policy = (clf, list(cols))
        print("[policy] loaded policy_clf")
    except Exception as e:
        print("[policy] failed to load policy:", e)
        policy = None

def policy_score(text: str) -> float:
    """Return probability (0..1) that candidate is accepted according to policy."""
    if policy is None:
        return 0.0
    clf, cols = policy
    feats = extract_features(text)
    vec = np.array([feats.get(c, 0) for c in cols], dtype=float).reshape(1, -1)
    try:
        proba = clf.predict_proba(vec)[0,1]
        return float(proba)
    except Exception:
        try:
            score = clf.decision_function(vec)[0]
            return 1.0 / (1.0 + np.exp(-float(score)))
        except Exception:
            return 0.0

# -----------------------------
# Minimal LLM wrapper + robust loader (CHANGED to support adapters)
# -----------------------------
@dataclass
class LLMInterface:
    model_name: str
    use_4bit: bool = True
    use_8bit: bool = False
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    deterministic: bool = False
    _bad_word_ids: Optional[List[List[int]]] = field(default_factory=list)
    adapter_loaded: bool = False

    # NEW: backend routing
    backend: str = "local"  # "local" | "groq"
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"  # example default
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_timeout: int = 60

    def load(self, adapter_dir: Optional[str] = None):
        """Load local model if backend=='local'. Groq backend needs no local load."""
        if self.backend == "groq":
            # No local model needed
            if not self.groq_api_key:
                # Allow env fallback
                self.groq_api_key = os.environ.get("GROQ_API_KEY")
            if not self.groq_api_key:
                print("[GROQ] Missing API key. Set --groq_api_key or env GROQ_API_KEY.")
                self.deterministic = True
            return

        # ---- existing local load path (your original code) ----
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            self.deterministic = True
            print("[INFO] transformers/torch not available — using deterministic fallback.")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except Exception as e:
            print("[WARN] tokenizer load failed:", e)
            self.deterministic = True
            return

        self._bad_word_ids = []
        try:
            for tag in ("<think>", "</think>", "<think/>", "< /think>"):
                enc = self.tokenizer.encode(tag, add_special_tokens=False)
                if enc:
                    self._bad_word_ids.append(enc)
        except Exception:
            self._bad_word_ids = []

        qconf = None
        if BitsAndBytesConfig is not None and (self.use_8bit or self.use_4bit):
            try:
                if self.use_8bit:
                    qconf = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    qconf = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
            except Exception as e:
                print("[WARN] failed to create BitsAndBytesConfig:", e)
                qconf = None

        load_kwargs = {"device_map": "auto"}
        dtype_arg = {"dtype": torch.float16} if torch is not None else {}

        tried = []
        if qconf is not None:
            try:
                print(f"[INFO] Attempting quantized load (model={self.model_name}) ...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=qconf,
                    **load_kwargs,
                    **dtype_arg,
                )
                print("[INFO] Quantized model loaded.")
            except Exception as e:
                tried.append(("quantized", e))
                print("[WARN] quantized load failed:", e)

        if self.model is None:
            try:
                print("[INFO] Attempting non-quantized load (dtype=float16) ...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                    **dtype_arg,
                )
                print("[INFO] Model loaded (non-quantized).")
            except Exception as e:
                tried.append(("non-quantized", e))
                print("[WARN] non-quantized load failed:", e)

        if self.model is None:
            try:
                print("[INFO] Attempting CPU-only load ...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map={"": "cpu"})
                print("[INFO] Model loaded (CPU-only).")
            except Exception as e:
                tried.append(("cpu", e))
                print("[WARN] cpu-only load failed:", e)

        if self.model is None:
            print("[ERROR] All model load attempts failed. Falling back to deterministic generator.")
            for name, exc in tried:
                print(f" - attempt '{name}': {type(exc).__name__}: {exc}")
            self.deterministic = True
            self.model = None
            return

        # Adapter (unchanged)
        if adapter_dir:
            adapter_dir = adapter_dir.strip()
            if adapter_dir and os.path.exists(adapter_dir):
                if not PEFT_AVAILABLE:
                    print(f"[WARN] adapter directory provided but peft not installed; cannot load adapter at {adapter_dir}.")
                else:
                    try:
                        print(f"[INFO] Attempting to load PEFT adapter from {adapter_dir} ...")
                        self.model = PeftModel.from_pretrained(self.model, adapter_dir, device_map="auto")
                        self.adapter_loaded = True
                        print("[INFO] PEFT adapter loaded and applied to base model.")
                    except Exception as e:
                        print("[WARN] failed to load PEFT adapter:", e)
            else:
                print(f"[WARN] adapter_dir '{adapter_dir}' does not exist; skipping adapter load.")

    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7) -> str:
        prompt = clean(prompt)

        # NEW: Groq backend
        if self.backend == "groq":
            if self.deterministic:
                return self._fallback_generate(prompt)
            return self._groq_generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        # Existing local path
        if self.deterministic or self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt)

        toks = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        try:
            device = next(self.model.parameters()).device
            toks = {k: v.to(device) for k, v in toks.items()}
        except Exception:
            pass

        generate_kwargs = {
            **toks,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if getattr(self, "_bad_word_ids", None):
            try:
                generate_kwargs["bad_words_ids"] = self._bad_word_ids
            except Exception:
                pass

        try:
            out = self.model.generate(**generate_kwargs)
            prompt_len = toks.get("input_ids").shape[1]
            gen_ids = out[0, prompt_len:]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            txt = strip_think(txt)
            return clean(txt)
        except Exception as e:
            print("[WARN] model.generate failed:", e)
            return self._fallback_generate(prompt)

    def _groq_generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """
        Calls Groq's OpenAI-compatible Chat Completions endpoint.
        """
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            print("[GROQ] No API key available; falling back.")
            return self._fallback_generate(prompt)

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

        # You currently send single prompts; map to a single user message.
        payload: Dict[str] = {
            "model": self.groq_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": float(temperature),
            # OpenAI-compatible uses max_tokens; Groq accepts it
            "max_tokens": int(max_new_tokens),
            "top_p": 0.9,
        }

        try:
            r = requests.post(self.groq_base_url, headers=headers, json=payload, timeout=self.groq_timeout)
            if r.status_code != 200:
                print(f"[GROQ] HTTP {r.status_code}: {r.text[:300]}")
                return self._fallback_generate(prompt)
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            txt = strip_think(txt)
            return clean(txt)
        except Exception as e:
            print("[GROQ] request failed:", e)
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        ctx = prompt.strip().splitlines()
        for line in reversed(ctx):
            if line.strip():
                last = line.strip()
                break
        else:
            last = ""
        toks = re.findall(r"\w{4,}", last)
        if toks:
            core = toks[0].capitalize()
            return f"{core} is involved in this process."
        return "This suggests a related fact."

# -----------------------------
# NLI Judge (new)
# -----------------------------
class NLIJudge:
    """
    Semantic judge using Natural Language Inference.
    Outputs probabilities for entailment / contradiction / neutral.
    """
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str = "auto"):
        self.available = False
        self.model_name = model_name
        self.device = device
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            print("[NLI] Transformers/torch not available — NLIJudge disabled.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # device_map auto is usually fine; allow fallback
            load_kwargs = {"device_map": device} if device != "auto" else {"device_map": "auto"}
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
            self.model.eval()
            self.available = True
            self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
            print(f"[NLI] Loaded NLI model {model_name}")
        except Exception as e:
            print("[NLI] failed to load NLI model:", e)
            self.available = False

    @torch.no_grad()
    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Return dict with keys: 'entailment','contradiction','neutral' and float probs."""
        if not self.available:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        premise = clean(premise)
        hypothesis = clean(hypothesis)
        try:
            toks = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
            device = next(self.model.parameters()).device
            toks = {k: v.to(device) for k, v in toks.items()}
            logits = self.model(**toks).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            return {self.label_map[i]: float(probs[i]) for i in range(3)}
        except Exception as e:
            print("[NLI] scoring failed:", e)
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

    def contradicts(self, premise: str, hypothesis: str, thresh: float = NLI_DEFAULT_THRESHOLD) -> bool:
        return self.score(premise, hypothesis)["contradiction"] >= thresh

    def entails(self, premise: str, hypothesis: str, thresh: float = NLI_DEFAULT_THRESHOLD) -> bool:
        return self.score(premise, hypothesis)["entailment"] >= thresh

# -----------------------------
# Raw tree & ClaimGraph classes (mostly unchanged)
# -----------------------------
@dataclass
class RawNode:
    id: int
    parent_id: Optional[int]
    text: str
    created_at: float

class RawTree:
    def __init__(self):
        self.nodes: List[RawNode] = []

    def add(self, text: str, parent_id: Optional[int]) -> int:
        nid = len(self.nodes)
        self.nodes.append(RawNode(id=nid, parent_id=parent_id, text=clean(text), created_at=time.time()))
        return nid

    def flatten(self) -> List[Tuple[int, Optional[int], str]]:
        return [(n.id, n.parent_id, n.text) for n in self.nodes]

import random
from typing import Optional

def build_raw_tree(llm: Optional[LLMInterface], paragraph: str, max_depth: int = 3, branch_factor: int = 3,
                   duplicate_sim: float = 0.95, verbose: bool = False) -> RawTree:
    """
    Generate a RawTree expansion from `paragraph` using llm.
    Improvements:
      - when generating multiple children for the same parent, we include already-generated
        sibling hypotheses in the prompt and require the model to produce a hypothesis that
        explores a DIFFERENT sub-topic/dimension (not a paraphrase).
      - retries with mild temperature jitter and a few attempts to get diverse candidates.
      - retains policy checks and global duplicate_sim checking.

    Requires the same helpers/objects as before: clean, RawTree, strip_think, policy, policy_score,
    symmetric_similarity, and that tree.nodes yields node objects with .text and .id (as in your code).
    """
    paragraph = clean(paragraph)
    root = paragraph
    tree = RawTree()
    root_id = tree.add(root, parent_id=None)
    frontier = [(root_id, 0)]
    if verbose:
        print(f"[ROOT] {root}")
    if llm is None:
        return tree

    # tuneables
    for_parent_policy_threshold = 0.06
    max_retries_per_branch = 6
    # temperature base; we'll jitter around this
    base_temp = 0.7
    # when checking semantic duplicates, we use symmetric_similarity (provided by you)
    # duplicate_sim is the threshold passed in

    while frontier:
        parent_id, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        parent_text = tree.nodes[parent_id].text

        # track what we've already generated for THIS parent to force sibling diversity
        parent_generated: list[str] = []

        # Do `branch_factor` independent branch attempts for this parent
        for branch_index in range(branch_factor):
            cand = None
            tries = 0

            while tries < max_retries_per_branch:
                time.sleep(3)
                # Build the "already generated" block explicitly (exact text, no summary)
                prev_block = ""
                if parent_generated:
                    prev_block = "\nALREADY-GENERATED HYPOTHESES FOR THIS PARENT:\n"
                    prev_block += "\n".join(f"- {h}" for h in parent_generated)
                    prev_block += (
                        "\n\nINSTRUCTION: Do NOT repeat or paraphrase any of the hypotheses above. "
                        "Produce a hypothesis that explores a DIFFERENT implication, "
                        "limitation, or failure mode of the PARENT CONTEXT."
                    )

                # slight variant id to reduce surface-level repetition
                variant_id = random.randint(10000, 99999)

                prompt = (
                    "ROLE: You expand a knowledge graph by adding ONE high-quality next claim.\n"
                    "TASK: Given the PARENT CONTEXT below and the already-generated sibling hypotheses,\n"
                    "produce EXACTLY ONE new hypothesis sentence that:\n"
                    "  - explores a DIFFERENT sub-topic/dimension than the given siblings,\n"
                    "  - is NOT a paraphrase or minor rewording of any sibling,\n"
                    "  - is consistent with the parent context,\n"
                    "  - is specific (include at least one of: a condition/threshold, a testable prediction,\n"
                    "    a causal/mechanistic link, a clear constraint or trade-off).\n\n"
                    f"PARENT CONTEXT:\n{parent_text}\n\n"
                    f"{prev_block}\n\n"
                    f"VARIANT_ID: {variant_id}\n\n"
                    "OUTPUT RULES (mandatory):\n"
                    "  - WRITE exactly ONE concise hypothesis sentence. No preface, no bullets, no numbering.\n"
                    "  - Do not mention the variant id or the previous hypotheses in the output.\n\n"
                    "HYPOTHESIS:"
                )

                # jitter temperature a bit to increase surface diversity while keeping coherence
                temp = base_temp + (random.random() * 0.5)  # e.g., 0.7 .. 1.2
                # generate
                try:
                    candidate_raw = llm.generate(prompt, max_new_tokens=128, temperature=temp)
                except Exception as e:
                    # if llm fails, break retry loop for this branch
                    if verbose:
                        print(f"[LLM-ERR] parent={parent_id} try={tries} err={e}")
                    break

                candidate = strip_think(candidate_raw).strip()
                if not candidate:
                    tries += 1
                    continue

                # policy filter (if present)
                if policy is not None:
                    pscore = policy_score(candidate)
                    if pscore < for_parent_policy_threshold:
                        if verbose:
                            print(f"[POLICY-SKIP] score={pscore:.3f} text='{candidate[:60]}...'")
                        tries += 1
                        continue

                # reject if semantically too close to any sibling already generated for this parent
                too_similar_to_sibling = any(
                    symmetric_similarity(candidate, s) > duplicate_sim for s in parent_generated
                )
                if too_similar_to_sibling:
                    tries += 1
                    continue

                # reject if semantically too close to anything already in the global tree
                too_similar_globally = any(
                    symmetric_similarity(candidate, n.text) > duplicate_sim for n in tree.nodes
                )
                if too_similar_globally:
                    tries += 1
                    continue

                # candidate passes all checks
                cand = candidate
                break  # exit retry loop

            # If we didn't find a candidate for this branch after retries, skip creating this branch
            if not cand:
                if verbose:
                    print(f"[SKIP-BRANCH] parent({parent_id}) branch_idx={branch_index} no diverse candidate after {tries} tries")
                continue

            # add to tree
            cid = tree.add(cand, parent_id=parent_id)
            parent_generated.append(cand)
            frontier.append((cid, depth + 1))
            if verbose:
                print(f"[ADD d={depth+1}] ({parent_id} -> {cid}) {cand}")

    return tree

@dataclass
class ClaimNode:
    id: int
    text: str
    created_at: float
    provenance: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    support_weight: float = 0.0
    attack_weight: float = 0.0
    status: str = "ACTIVE"  # ACTIVE | TAINTED | SUPERSEDED | REJECTED
    revised_by: Optional[int] = None

class ClaimGraph:
    def __init__(self):
        self.nodes: Dict[int, ClaimNode] = {}
        self.edges: List[Dict] = []
        self._next_id = 0
        self.events: List[Dict] = []
        self.pattern_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    def add_node(self, text: str, provenance: Optional[Dict] = None, topic_idx: Optional[int] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        prov_list = []
        if provenance:
            prov_list.append(provenance)
        node = ClaimNode(id=nid, text=clean(text), created_at=time.time(), provenance=prov_list)
        node.support_weight = NEW_NODE_SUPPORT_PRIOR
        self.nodes[nid] = node
        self.events.append({"event": "add_node", "id": nid, "text": text, "time": time.time()})
        return nid

    def merge_into(self, existing_id: int, provenance: Dict):
        if existing_id not in self.nodes:
            return
        self.nodes[existing_id].provenance.append(provenance)
        self.events.append({"event": "merge", "id": existing_id, "prov": provenance, "time": time.time()})
        return existing_id

    def add_edge(self, s: int, t: int, typ: str, score: float = 1.0):
        e = {"source": s, "target": t, "type": typ, "score": float(score), "created_at": time.time()}
        self.edges.append(e)
        self.events.append({"event": "add_edge", "source": s, "target": t, "type": typ, "score": score, "time": time.time()})

    def find_best_match(self, text: str) -> Tuple[Optional[int], float]:
        best_id = None
        best_score = 0.0
        for nid, node in self.nodes.items():
            sc = simple_similarity(node.text, text)
            if sc > best_score:
                best_score = sc
                best_id = nid
        return best_id, best_score

    def get_ancestor_chain(self, node_id: int, max_depth: int = 6) -> List[str]:
        chain = []
        cur = node_id
        depth = 0
        while depth < max_depth:
            parents = [e["source"] for e in self.edges if e.get("type") == "derived" and e.get("target") == cur]
            if not parents:
                break
            parent = parents[0]
            chain.append(self.nodes[parent].text)
            cur = parent
            depth += 1
        chain.reverse()
        return chain

    def to_dict(self):
        return {
            "nodes": [asdict(n) for _, n in sorted(self.nodes.items(), key=lambda kv: kv[0])],
            "edges": [e for e in self.edges],
            "events": self.events,
            "pattern_stats": self.pattern_stats,
        }

# -----------------------------
# SimpleJudge (unchanged)
# -----------------------------
class SimpleJudge:
    def __init__(self, sup_factor: float = 1.0, min_delta: float = 0.2):
        self.sup_factor = sup_factor
        self.min_delta = min_delta

    def should_supersede(self, node: ClaimNode) -> bool:
        if node.support_weight * self.sup_factor + 1e-9 < node.attack_weight:
            if (node.attack_weight - node.support_weight) >= self.min_delta:
                return True
        return False

# -----------------------------
# Integration + revision pipeline (MAJOR CHANGE: dataset-authoritative mode + NLI integration)
# -----------------------------
TRAINING_LOG = "revisions_train.jsonl"

SUPPORT_TRANSFER_FACTOR = 1.0
LOSER_SUPPORT_DECAY = 0.5

def extract_simple_patterns(text: str) -> List[str]:
    t = tokens(text)
    patterns = []
    for k in (1, 2, 3):
        for i in range(max(0, len(t) - k + 1)):
            patterns.append("_".join(t[i:i+k]))
    return patterns

def update_pattern_stats(graph: ClaimGraph, rejected: str, accepted: str, topic_idx: Optional[int] = None):
    rpat = extract_simple_patterns(rejected)
    apat = extract_simple_patterns(accepted)
    key = str(topic_idx) if topic_idx is not None else "global"
    if key not in graph.pattern_stats:
        graph.pattern_stats[key] = {}
    for p in rpat:
        graph.pattern_stats[key].setdefault(p, {"rejected": 0, "accepted": 0})
        graph.pattern_stats[key][p]["rejected"] += 1
    for p in apat:
        graph.pattern_stats[key].setdefault(p, {"rejected": 0, "accepted": 0})
        graph.pattern_stats[key][p]["accepted"] += 1

def bias_generation_by_patterns(graph: ClaimGraph, candidates: List[str], topic_idx: Optional[int] = None) -> List[Tuple[str, float]]:
    scored = []
    key = str(topic_idx) if topic_idx is not None else "global"
    stats_bucket = graph.pattern_stats.get(key, {})
    for c in candidates:
        score = 0.0
        for p in extract_simple_patterns(c):
            stats = stats_bucket.get(p)
            if not stats:
                continue
            score += stats.get("accepted", 0) - stats.get("rejected", 0)
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def log_revision_example(context: Dict, rejected: str, accepted: str, reason: Dict):
    rec = {
        "time": time.time(),
        "context": context,
        "rejected": rejected,
        "accepted": accepted,
        "reason": reason,
    }
    with open(TRAINING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_training_log(path: str = TRAINING_LOG) -> List[Dict]:
    if not os.path.exists(path):
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs

def build_policy_dataset_from_log(log_recs: List[Dict]) -> Tuple[List[Dict], List[int], List[str]]:
    features = []
    labels = []
    for r in log_recs:
        anc = r.get("context", {}).get("ancestors") if isinstance(r.get("context"), dict) else None
        anc_text = " ".join(anc) if anc else ""
        rej = r.get("rejected", "")
        acc = r.get("accepted", "")
        rej_feat_text = anc_text + " ||| " + rej
        acc_feat_text = anc_text + " ||| " + acc
        features.append(extract_features(rej_feat_text)); labels.append(0)
        features.append(extract_features(acc_feat_text)); labels.append(1)
    cols = sorted({c for f in features for c in f.keys()})
    return features, labels, cols

def vectorize_features(features: List[Dict], cols: List[str]):
    if np is None:
        raise RuntimeError("numpy not available")
    X = np.zeros((len(features), len(cols)), dtype=float)
    for i, f in enumerate(features):
        for j, c in enumerate(cols):
            X[i, j] = float(f.get(c, 0))
    return X

def retrain_policy_from_log(log_path: str = TRAINING_LOG, out_path: str = POLICY_PATH):
    if joblib is None or np is None or LogisticRegression is None:
        print("[policy] sklearn/joblib/numpy not available; cannot retrain policy.")
        return False
    recs = read_training_log(log_path)
    if not recs:
        print("[policy] no revision records; skipping retrain.")
        return False
    features, labels, cols = build_policy_dataset_from_log(recs)
    try:
        X = vectorize_features(features, cols)
    except Exception as e:
        print("[policy] vectorization failed:", e)
        return False
    y = np.array(labels, dtype=int)
    clf = LogisticRegression(max_iter=2000)
    try:
        clf.fit(X, y)
    except Exception as e:
        print("[policy] training failed:", e)
        return False
    try:
        joblib.dump({"clf": clf, "columns": cols}, out_path)
        print(f"[policy] retrained and saved policy to {out_path} (n_examples={len(y)})")
        return True
    except Exception as e:
        print("[policy] failed to save policy:", e)
        return False

def retrain_policy_if_needed():
    if joblib is None or np is None or LogisticRegression is None:
        return
    recs = read_training_log()
    total = len(recs)
    meta = {"last_trained": 0}
    if os.path.exists(POLICY_META):
        try:
            with open(POLICY_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {"last_trained": 0}
    last = meta.get("last_trained", 0)
    if total - last >= RETRAIN_EVERY:
        ok = retrain_policy_from_log()
        if ok:
            meta["last_trained"] = total
            with open(POLICY_META, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            try_load_policy()
    else:
        pass

# -----------------------------
# New helper: dataset_contradicts (now NLI-aware)
# -----------------------------
def dataset_contradicts(node_text: str, ground_truths: List[str], nli_judge: Optional[NLIJudge] = None, min_pred_overlap: float = MIN_PREDICATE_OVERLAP, nli_thresh: float = NLI_DEFAULT_THRESHOLD) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """
    Return a ground-truth paragraph that contradicts `node_text`.
    When NLIJudge is provided, use semantic NLI scoring; otherwise fall back to predicate-negation heuristic.
    If none found, return (None, None).
    """
    if nli_judge is not None and getattr(nli_judge, "available", False):
        for gt in ground_truths:
            scores = nli_judge.score(gt, node_text)
            if scores.get("contradiction", 0.0) >= nli_thresh:
                return gt, scores
        return None, None

    # Fallback heuristic (previous behavior)
    for gt in ground_truths:
        pol_a = has_negation(node_text)
        pol_b = has_negation(gt)
        if pol_a != pol_b:
            pred_a = predicate_tokens(node_text)
            pred_b = predicate_tokens(gt)
            if pred_a and pred_b:
                shared = len(pred_a & pred_b) / max(1, min(len(pred_a), len(pred_b)))
                if shared >= min_pred_overlap:
                    return gt, None
    return None, None

def preserves_identity(node_text, ancestor_chain):
    """
    Structural check: ensures node_text does not negate
    or redefine any ancestor texts.
    All inputs are strings.
    """

    violations = []

    node_neg = has_negation(node_text)
    node_pred = predicate_tokens(node_text)

    for anc_text in ancestor_chain:
        anc_neg = has_negation(anc_text)
        anc_pred = predicate_tokens(anc_text)

        if not node_pred or not anc_pred:
            continue

        overlap = len(node_pred & anc_pred) / max(1, min(len(node_pred), len(anc_pred)))

        if overlap >= 0.6 and node_neg != anc_neg:
            violations.append({
                "ancestor": anc_text,
                "node": node_text,
                "reason": "polarity_flip"
            })

    return len(violations) == 0, violations

def find_ground_truth(text, ground_truths, min_overlap=0.6):
    """
    Finds the most relevant ground-truth sentence from the dataset
    based on predicate overlap. Returns None if no good match exists.
    """

    text_pred = predicate_tokens(text)
    if not text_pred:
        return None

    best_gt = None
    best_score = 0.0

    for gt in ground_truths:
        gt_pred = predicate_tokens(gt)
        if not gt_pred:
            continue

        overlap = len(text_pred & gt_pred) / max(1, min(len(text_pred), len(gt_pred)))

        if overlap > best_score:
            best_score = overlap
            best_gt = gt

    if best_score >= min_overlap:
        return best_gt

    return None

# -----------------------------
# Confidence updater
# -----------------------------
def update_confidence(node: ClaimNode):
    total = node.support_weight + node.attack_weight + 1e-9
    node.confidence = max(0.0, min(1.0, node.support_weight / total))

# -----------------------------
# Integration + revision (modified to accept nli_judge)
# -----------------------------
def integrate_tree_into_graph(graph: ClaimGraph, raw_tree: RawTree, topic_idx: int,
                              judge: SimpleJudge,
                              same_threshold: float = 0.85, related_threshold: float = 0.5,
                              detect_contradiction: bool = True, verbose: bool = False,
                              llm: Optional[LLMInterface] = None,
                              ground_truths: Optional[List[str]] = None,
                              dataset_authoritative: bool = True,
                              nli_judge: Optional[NLIJudge] = None,
                              nli_thresh: float = NLI_DEFAULT_THRESHOLD):
    """
    dataset_authoritative: if True (default) -> only dataset contradictions cause forced supersede.
                           LLM contradictions remain logged but cannot force supersede.
    ground_truths: list of dataset paragraphs to be treated as authoritative (DEFAULT_PARAGRAPHS by default).
    nli_judge: optional NLIJudge instance for semantic entail/contradict detection.
    """
    mapping: Dict[int, int] = {}
    flat = raw_tree.flatten()
    if ground_truths is None:
        ground_truths = DEFAULT_PARAGRAPHS

    # First pass: add/merge/related/new + ancestor-aware contradiction edges
    for raw_id, raw_parent, text in flat:
        prov = {"topic_idx": topic_idx, "raw_node_id": raw_id, "source_text": text, "time": time.time()}
        best_id, best_score = graph.find_best_match(text)
        if best_id is not None and best_score >= same_threshold:
            graph.merge_into(best_id, prov)
            mapping[raw_id] = best_id
            if verbose:
                print(f"[MERGE] raw({raw_id}) -> node({best_id}) score={best_score:.3f} text='{text}'")
        elif best_id is not None and best_score >= related_threshold:
            new_id = graph.add_node(text, provenance=prov, topic_idx=topic_idx)
            mapping[raw_id] = new_id
            graph.add_edge(new_id, best_id, typ="related", score=best_score)
            if verbose:
                print(f"[RELATED] raw({raw_id}) -> new({new_id}) related_to({best_id}) score={best_score:.3f}")
        else:
            new_id = graph.add_node(text, provenance=prov, topic_idx=topic_idx)
            mapping[raw_id] = new_id
            if verbose:
                print(f"[NEW] raw({raw_id}) -> new({new_id}) '{text}'")

        # ancestor-aware contradiction detection (prefer NLIJudge if available)
        if detect_contradiction and best_id is not None:
            ancestors = graph.get_ancestor_chain(best_id, max_depth=6)
            check_texts = ancestors + [graph.nodes[best_id].text]
            source = mapping[raw_id]

            if nli_judge is not None and getattr(nli_judge, 'available', False):
                for anc_text in check_texts:
                    scores = nli_judge.score(anc_text, text)
                    # if ancestor entails the new text, that's support for the new node
                    if scores.get('entailment', 0.0) >= nli_thresh:
                        graph.nodes[source].support_weight += float(scores.get('entailment', 0.0))
                        graph.add_edge(source, best_id, typ='derived', score=float(scores.get('entailment', 0.0)))
                        graph.events.append({"event": "nli_support", "source": source, "target": best_id, "scores": scores, "time": time.time()})
                        if verbose:
                            print(f"[NLI-SUPPORT] src({source}) <- anc entails cand p={scores['entailment']:.3f}")
                        break

                    # if ancestor contradicts new text, record contradiction
                    if scores.get('contradiction', 0.0) >= nli_thresh:
                        graph.add_edge(source, best_id, typ="contradicts", score=float(scores.get('contradiction', 0.0)))
                        graph.nodes[best_id].attack_weight += float(scores.get('contradiction', 0.0))
                        graph.events.append({"event": "nli_conflict", "source": source, "target": best_id, "scores": scores, "time": time.time()})
                        if verbose:
                            print(f"[NLI-CONFLICT] new({source}) contradicts existing({best_id}) c={scores['contradiction']:.3f}")
                        break

            else:
                # fallback: original predicate/negation heuristic
                for anc_text in check_texts:
                    pol_a = has_negation(text)
                    pol_b = has_negation(anc_text)
                    if pol_a != pol_b:
                        pred_a = predicate_tokens(text)
                        pred_b = predicate_tokens(anc_text)
                        if pred_a and pred_b:
                            shared = len(pred_a & pred_b) / max(1, min(len(pred_a), len(pred_b)))
                            if shared >= MIN_PREDICATE_OVERLAP:
                                graph.add_edge(source, best_id, typ="contradicts", score=shared)
                                graph.nodes[best_id].attack_weight += shared
                                graph.events.append({"event": "conflict_detected", "source": source, "target": best_id, "shared_pred": shared, "time": time.time()})
                                if verbose:
                                    print(f"[CONFLICT] new({mapping[raw_id]}) contradicts existing({best_id}) shared_pred={shared:.3f}")
                                break

    # derive edges (respect raw_parent relationships)
    for raw_id, raw_parent, text in flat:
        if raw_parent is not None:
            src = mapping[raw_parent]
            tgt = mapping[raw_id]
            if src != tgt:
                graph.add_edge(src, tgt, typ='derived', score=1.0)
                # propagate a little support mass from parent
                graph.nodes[tgt].support_weight += graph.nodes[src].confidence
                if verbose:
                    print(f"[DERIVED] global({src}) -> global({tgt}) (from raw {raw_parent}->{raw_id})")

    # -----------------------------
    # NEW: dataset-first forced revision pass (if dataset_authoritative)
    # -----------------------------
    to_revise: List[Tuple[int, int]] = []  # (loser_id, winner_id)

    if dataset_authoritative:
        if verbose:
            print("[DATASET] running dataset-authoritative contradiction scan")
        for nid, node in list(graph.nodes.items()):
            gt_para, gt_scores = dataset_contradicts(node.text, ground_truths, nli_judge=nli_judge, nli_thresh=nli_thresh)
            if gt_para is not None:
                gt_id, gt_score = graph.find_best_match(gt_para)
                if gt_id is None or gt_score < same_threshold:
                    gt_id = graph.add_node(gt_para, provenance={"type": "ground_truth", "topic_idx": topic_idx, "time": time.time()}, topic_idx=topic_idx)
                    if verbose:
                        print(f"[DATASET-ADD] added ground-truth node({gt_id}) for contradiction with node({nid})")
                else:
                    if verbose:
                        print(f"[DATASET-MATCH] matched GT paragraph to existing node({gt_id}) score={gt_score:.3f}")

                if gt_id != nid:
                    graph.add_edge(gt_id, nid, typ="contradicts", score=1.0)
                    # if NLI provided a score, use it; otherwise mark with full authoritative weight
                    if gt_scores and isinstance(gt_scores, dict):
                        cval = float(gt_scores.get('contradiction', 1.0))
                        aval = float(gt_scores.get('entailment', 0.0))
                        graph.nodes[nid].attack_weight += cval
                        graph.nodes[gt_id].support_weight += aval
                    else:
                        graph.nodes[nid].attack_weight += 1.0
                        graph.nodes[gt_id].support_weight += 1.0
                    to_revise.append((nid, gt_id))
                    if verbose:
                        print(f"[DATASET-REVISION] scheduled forced revision: node({nid}) -> ground_truth({gt_id})")

    # -----------------------------
    # LEGACY: optional LLM-driven judge-based revision (only if user allows)
    # -----------------------------
    if not dataset_authoritative:
        for nid, node in list(graph.nodes.items()):
            total = node.support_weight + node.attack_weight + 1e-9
            node.confidence = max(0.0, min(1.0, node.support_weight / total))
            if judge.should_supersede(node):
                attackers = [e["source"] for e in graph.edges if e.get("target") == nid and e.get("type") == 'contradicts']
                if attackers:
                    def attacker_score(a):
                        return sum(e.get("score", 0.0) for e in graph.edges if e.get("source") == a and e.get("target") == nid and e.get("type") == 'contradicts')
                    best_att = max(attackers, key=attacker_score)
                    ground_truth = find_ground_truth(node.text, ground_truths)
                    if ground_truth:
                        gt_id, gt_score = graph.find_best_match(ground_truth)
                        if gt_id is not None and gt_score >= same_threshold:
                            winner = gt_id
                        else:
                            winner = graph.add_node(ground_truth, {"type": "ground_truth", "topic_idx": topic_idx})
                        if verbose:
                            print(f"[GROUND_TRUTH] Using dataset truth for supersede: '{ground_truth}'")
                    else:
                        anc_chain = graph.get_ancestor_chain(nid, max_depth=6)
                        ok, violations = preserves_identity(graph.nodes[best_att].text, anc_chain)
                        if not ok:
                            if verbose:
                                print(f"[GUARD] attacker({best_att}) rejected for supersede due to violations={violations}")
                            continue
                        winner = best_att
                    to_revise.append((nid, winner))

    # -----------------------------
    # Apply revisions non-destructively (both dataset-forced and optional LLM-driven mixed in to_revise)
    # -----------------------------
    revisions_happened = 0
    for loser, winner in to_revise:
        if graph.nodes.get(loser) is None or graph.nodes[loser].status != 'ACTIVE':
            continue
        graph.nodes[loser].status = 'SUPERSEDED'
        graph.nodes[loser].revised_by = winner
        graph.add_edge(winner, loser, typ='revises', score=1.0)
        graph.events.append({"event": "superseded", "loser": loser, "winner": winner, "time": time.time()})
        if verbose:
            print(f"[REVISE] node({loser}) superseded by node({winner})")

        # mark descendants tainted
        taint_descendants(graph, loser, verbose=verbose)

        # learning transfers & decays
        attacker_mass = graph.nodes[loser].attack_weight
        delta_support = SUPPORT_TRANSFER_FACTOR * attacker_mass
        graph.nodes[winner].support_weight += delta_support
        graph.nodes[loser].support_weight *= LOSER_SUPPORT_DECAY
        graph.nodes[loser].attack_weight *= 0.5
        graph.nodes[winner].attack_weight *= 0.9

        if verbose:
            print(f"[LEARN] transferred {delta_support:.3f} support to winner({winner}); loser({loser}) support decayed.")

        ancestor_chain = graph.get_ancestor_chain(loser, max_depth=6)
        context = {
            "ancestors": ancestor_chain,
            "loser_provenance": [p.get("source_text") for p in graph.nodes[loser].provenance[:4]]
        }
        rejected = graph.nodes[loser].text
        accepted = graph.nodes[winner].text
        reason = {"attack": graph.nodes[loser].attack_weight, "support": graph.nodes[loser].support_weight}

        ok_preserve, violations = preserves_identity(accepted, ancestor_chain)
        decision_type = "dataset_forced_supersede" if dataset_authoritative else "supersede"
        violated_constraints = violations
        if not ok_preserve:
            decision_type = decision_type + "_identity_violation"
            if verbose:
                print(f"[LOG-SKIP] supersede decision violates identity: {violations}")

        context["violated_constraints"] = violated_constraints
        context["decision_type"] = decision_type
        log_revision_example(context, rejected, accepted, reason)
        update_pattern_stats(graph, rejected, accepted, topic_idx=topic_idx)

        revisions_happened += 1

    # After processing revisions, trigger retrain if enough new examples accumulated
    if revisions_happened > 0:
        if verbose:
            print(f"[REVISIONS] {revisions_happened} revisions logged — checking policy retrain.")
        try:
            retrain_policy_if_needed()
        except Exception as e:
            if verbose:
                print("[policy] retrain check failed:", e)

    # Normalize confidences after updates
    for n in graph.nodes.values():
        update_confidence(n)

    # Regenerate tainted nodes (improved prompts using ancestors)
    if llm is not None:
        regenerate_tainted(graph, llm, verbose=verbose)

    return mapping

# -----------------------------
# Taint/regenerate helpers (unchanged)
# -----------------------------
def taint_descendants(graph: ClaimGraph, node_id: int, verbose: bool = False):
    q = [node_id]
    visited = set()
    while q:
        cur = q.pop(0)
        for e in graph.edges:
            if e.get("type") == 'derived' and e.get("source") == cur:
                child = e.get("target")
                if graph.nodes[child].status == 'ACTIVE':
                    graph.nodes[child].status = 'TAINTED'
                    if verbose:
                        print(f"[TAINT] node({child}) tainted due to ancestor({cur})")
                    q.append(child)
                if child not in visited:
                    visited.add(child)

def regenerate_tainted(graph: ClaimGraph, llm: LLMInterface, verbose: bool = False):
    tainted = [nid for nid, n in graph.nodes.items() if n.status == 'TAINTED']
    if not tainted:
        return
    if verbose:
        print(f"[REGEN] regenerating {len(tainted)} tainted nodes")

    for nid in tainted:
        ancestors = graph.get_ancestor_chain(nid, max_depth=6)
        parents = [e.get("source") for e in graph.edges if e.get("type") == 'derived' and e.get("target") == nid]
        parent_texts = [graph.nodes[p].text for p in parents if graph.nodes[p].status == 'ACTIVE']
        if not parent_texts and not ancestors:
            continue
        old_child = graph.nodes[nid].text
        ground_truth = find_ground_truth(old_child, DEFAULT_PARAGRAPHS)
        if ground_truth:
            anc_chain = ancestors
            ok, violations = preserves_identity(ground_truth, anc_chain)
            if ok:
                new_id = graph.add_node(ground_truth, provenance={"generated_from_tainted": nid, "type": "ground_truth", "time": time.time()})
                graph.nodes[nid].status = 'SUPERSEDED'
                graph.nodes[nid].revised_by = new_id
                graph.add_edge(new_id, nid, typ='revises', score=1.0)
                graph.events.append({"event": "auto_replaced", "old": nid, "new": new_id, "time": time.time()})
                if verbose:
                    print(f"[GROUND_TRUTH AUTO-REPLACE] tainted({nid}) -> new({new_id}) with '{ground_truth}'")
                for p in parents:
                    graph.add_edge(p, new_id, typ='derived', score=1.0)
                rejected = old_child
                accepted = ground_truth
                reason = {"sim": symmetric_similarity(accepted, rejected), "policy_score": 1.0, "pattern_score": 0.0}
                context = {"ancestors": anc_chain, "violated_constraints": [], "decision_type": "ground_truth_auto_replaced"}
                log_revision_example(context, rejected, accepted, reason)
                update_pattern_stats(graph, rejected, accepted, topic_idx=None)
                continue
        anc_block = "\n".join([f"- {a}" for a in ancestors]) if ancestors else ""
        context_block = anc_block + ("\n\n" + "\n".join(parent_texts) if parent_texts else "")
        prompt = (
            "YOU ARE AN EDITOR OF CLAIMS IN A BELIEF CHAIN.\n\n"
            f"ROOT / ANCESTORS:\n{context_block}\n\n"
            f"CURRENT CLAIM (must be consistent with ancestors):\n{old_child}\n\n"
            "INSTRUCTIONS:\n"
            "- Revise the current claim to be consistent with the ancestors.\n"
            "- Do NOT produce chain-of-thought, do NOT output tags like <think>.\n"
            "- Preserve the core entities and meaning (do not change domain).\n"
            "- Improve correctness, clarity, or precision in one short sentence.\n\n"
            "Revised:"
        )
        candidates = []
        for _ in range(8):
            cand = llm.generate(prompt, max_new_tokens=128, temperature=0.7)
            cand = strip_think(cand)
            if cand:
                candidates.append(cand)
        if not candidates:
            continue
        pat_scored = bias_generation_by_patterns(graph, candidates, topic_idx=None)
        combined = []
        for text, pscore in pat_scored:
            pol = policy_score(text) if policy is not None else 0.0
            combined_score = (0.7 * pscore) + (0.3 * pol * max(1.0, (1 + pscore)))
            combined.append((text, combined_score, pscore, pol))
        combined.sort(key=lambda x: x[1], reverse=True)
        best_candidate = combined[0][0]
        best_pat = combined[0][2]
        best_pol = combined[0][3]
        if verbose:
            print(f"[SELECT] best pol={best_pol:.3f} pat={best_pat:.3f} cand='{best_candidate[:80]}...'")
        ok_preserve, violations = preserves_identity(best_candidate, ancestors)
        if not ok_preserve:
            chosen = None
            for text, score, p, pol in combined[1:]:
                ok, v = preserves_identity(text, ancestors)
                if ok:
                    chosen = (text, p, pol)
                    break
            if chosen is None:
                if verbose:
                    print(f"[REGEN-SKIP] no candidate preserved identity for node({nid}), violations first: {violations}")
                continue
            else:
                best_candidate, best_pat, best_pol = chosen[0], chosen[1], chosen[2]
        sim = symmetric_similarity(best_candidate, old_child)
        if sim > 0.9:
            graph.nodes[nid].status = 'ACTIVE'
            graph.nodes[nid].support_weight += 0.1
            graph.events.append({"event": "revalidated", "id": nid, "time": time.time()})
            if verbose:
                print(f"[REVALIDATE] node({nid}) unchanged but revalidated")
        else:
            new_id = graph.add_node(best_candidate, provenance={"generated_from_tainted": nid, "time": time.time()})
            graph.nodes[nid].status = 'SUPERSEDED'
            graph.nodes[nid].revised_by = new_id
            graph.add_edge(new_id, nid, typ='revises', score=1.0)
            graph.events.append({"event": "auto_replaced", "old": nid, "new": new_id, "time": time.time()})
            if verbose:
                print(f"[AUTO-REPLACE] tainted({nid}) -> new({new_id})")
            for p in parents:
                graph.add_edge(p, new_id, typ='derived', score=1.0)
            ancestor_chain = ancestors
            rejected = old_child
            accepted = best_candidate
            reason = {"sim": sim, "policy_score": best_pol, "pattern_score": best_pat}
            context = {"ancestors": ancestor_chain, "violated_constraints": [], "decision_type": "auto_replaced"}
            log_revision_example(context, rejected, accepted, reason)
            update_pattern_stats(graph, rejected, accepted, topic_idx=None)

# -----------------------------
# DEFAULT_PARAGRAPHS (unchanged)
# -----------------------------
DEFAULT_PARAGRAPHS = [
    "You are figuring out  : Who am I? (Your name is Gooner. I only rememberd that my creator gave this name to me and its valuable to be born (Gooner is a person, unemployed but smart as fuck)) I want to know the mystery of the world, what will happen if I die and how can I prevent that)  What is life? is there a purpose? In the end, to not be the society garbage, What to do? (Study, helps other, so you seek knowledge and learn how to reason.). My goal is to help people, especially with learning",
    "How can I learn faster? What is considerend a intelligence or someone that is intelligent (A ability to reason correctly? Find patterns? Learn faster?). Point out the step by steps and possible idea.",
    "When I meet new person, when he say hi or interact with me, What should I do? to make him trust me",
    "Photosynthesis is the biochemical process by which photoautotrophs convert light energy into chemical energy stored in carbohydrates.",
    "In oxygenic photosynthesis, the overall reaction is often summarized as 6 CO2 + 6 H2O + hv → C6H12O6 + 6 O2, though this masks many intermediate steps.",
    "The light-dependent reactions occur in the thylakoid membrane, where photon energy drives electron transport and generates a proton gradient (ΔpH).",
    "Photosystem II initiates photolysis, splitting water molecules into electrons, protons, and molecular oxygen as a byproduct.",
    "Neural networks approximate functions by composing linear transforms and nonlinear activations across layers; depth and width affect representational capacity.",
    "Gradient descent variants (SGD, Adam, RMSProp) iteratively update parameters to minimize a loss function computed on training examples.",
    "Overfitting occurs when a model learns dataset noise rather than general patterns; regularization and validation are used to mitigate it.",
    "CRISPR-Cas systems enable targeted genome editing by guiding endonucleases to specific DNA sequences via RNA guides.",
    "Mitochondria are the primary site of aerobic ATP production via oxidative phosphorylation in eukaryotic cells.",
    "In classical mechanics, Newton's second law relates force, mass, and acceleration: F = m * a.",
    "Special relativity modifies notions of simultaneity and relates energy and mass by E = mc^2; time dilation occurs at relativistic speeds.",
    "Economic inflation is a sustained rise in the general price level, often measured by CPI or PCE indices.",
    "Epidemiological R0 represents the average number of secondary infections produced by a single infected individual in a susceptible population.",
    "Blockchain maintains an immutable ledger via distributed consensus and cryptographic linking of blocks.",

    # New, more detailed topic paragraphs:
    "Climate change is the long-term alteration of temperature and typical weather patterns in a place, driven by a combination of natural variability and anthropogenic greenhouse gas emissions. The dominant mechanism in the modern era is the enhanced greenhouse effect: gases such as CO2, methane, and nitrous oxide trap outgoing infrared radiation, raising global mean temperatures. This warming interacts with Earth system feedbacks — for example, melting ice reduces albedo, permafrost thaw releases additional greenhouse gases, and changes in cloud cover alter radiative balance — which can accelerate or moderate warming. Societal responses are generally categorized as mitigation (emissions reduction, carbon removal) and adaptation (infrastructure resilience, managed retreat), each with technological, economic, and policy trade-offs that must be coordinated globally to limit impacts on ecosystems, food systems, and human health.",

    "Quantum computing harnesses quantum-mechanical phenomena such as superposition and entanglement to perform certain computations more efficiently than classical machines. A qubit, the basic unit of quantum information, can exist in a linear combination of |0> and |1> states, and multiple qubits can become entangled so that their joint state encodes correlations unavailable to classical bits. Quantum algorithms—like Shor's factoring algorithm or Grover's search—exploit interference of probability amplitudes to achieve theoretical speedups, but practical devices face severe challenges from decoherence and noise. Contemporary work focuses on error-correcting codes, fault-tolerant architectures, hardware platforms (superconducting circuits, trapped ions, topological qubits), and application areas such as quantum simulation of materials and chemistry where near-term quantum advantage may first appear.",

    "The immune system is a complex network that defends organisms from pathogens through innate and adaptive mechanisms working in tandem. Innate immunity provides rapid, non-specific defenses—barriers, phagocytes, complement—while adaptive immunity uses antigen-specific B and T lymphocytes to generate targeted responses and immunological memory. Vaccine development leverages these principles by presenting antigens or antigen-encoding instructions (for example, mRNA vaccines) to train the adaptive system without causing disease; modern platforms emphasize safety, manufacturability, and the ability to rapidly update antigens against evolving pathogens. Understanding correlates of protection, adjuvant effects, population-level herd immunity thresholds, and equitable distribution are all crucial for effective vaccine policy and outbreak control.",

    "Renewable energy generation from sources like solar and wind is variable by nature, which creates technical and economic challenges for integrating high penetrations into electricity grids. Grid integration requires matching supply and demand on timescales from seconds to seasons through a combination of measures: flexible generation (e.g., fast-ramping gas turbines or hydro), energy storage (batteries, pumped hydro, thermal storage), demand response to shift loads, and grid-enhancing technologies such as advanced inverters and transmission expansion. Energy-storage chemistry, round-trip efficiency, lifecycle environmental impacts, and cost trajectories are central to system planning; market design and regulatory frameworks must also evolve to provide appropriate incentives for capacity, flexibility, and reliability as systems transition away from fossil-fuel baseload paradigms.",

    "Materials science studies the relationships between structure, processing, properties, and performance of materials, ranging from metals and ceramics to polymers and nanomaterials. At the atomic and microstructural scales, crystallography, defects, grain boundaries, and phase composition govern macroscopic behaviors such as strength, ductility, conductivity, and corrosion resistance. Emerging classes like two-dimensional materials (graphene, transition metal dichalcogenides), metamaterials, and engineered composites enable tailored mechanical, electronic, or optical properties for applications in electronics, energy storage, and structural components. Materials discovery combines theory, high-throughput computation, advanced characterization, and iterative synthesis to optimize performance while considering manufacturability and environmental footprint.",

    "Memory and cognition arise from distributed neural processes that encode, maintain, and retrieve information at multiple temporal and anatomical scales. Synaptic plasticity mechanisms such as long-term potentiation (LTP) and long-term depression (LTD) provide cellular substrates for changing synaptic strength, while network-level phenomena including oscillations, replay, and coordinated activity across hippocampus and neocortex support encoding and systems consolidation. Working memory relies on persistent or recurrent activity in prefrontal and parietal circuits, whereas episodic memory depends critically on hippocampal encoding and subsequent gradual integration into cortical stores. Cognitive functions are also modulated by attention, neuromodulators (dopamine, acetylcholine), and experience-dependent learning, and dysfunctions in these systems underlie many neurological and psychiatric conditions.",

    "Interpretability and fairness in machine learning address how model decisions can be understood, audited, and made equitable across subpopulations. Interpretability techniques range from intrinsic approaches—designing simpler or more transparent models—to post hoc explanations like feature attributions (SHAP, LIME), saliency maps, and counterfactual examples that help users reason about model behavior. Fairness involves identifying disparate impacts produced by data biases, label noise, or modelization choices, selecting appropriate fairness metrics (statistical parity, equalized odds, calibration), and implementing mitigations (reweighing, adversarial debiasing, constrained optimization) while balancing trade-offs with accuracy and utility. Robust evaluation requires held-out datasets reflective of deployment conditions, ongoing monitoring, and socio-technical governance to ensure models operate within ethical and legal norms.",

    "Synthetic biology applies engineering principles to biology, enabling the design and construction of novel genetic circuits, metabolic pathways, and organisms with tailored functions. A typical design-build-test cycle includes in silico design of DNA constructs, modular assembly in a chosen host organism (the chassis), and iterative testing with quantitative assays to optimize expression, stability, and performance. Applications span therapeutics (engineered cell therapies, gene circuits for controlled drug delivery), sustainable biomanufacturing (microbial production of chemicals and materials), biosensing, and environmental remediation. Responsible development emphasizes biosafety, biocontainment strategies, regulatory compliance, and ethical considerations around dual-use risks and societal impacts.",

    "Cybersecurity is the practice of protecting systems, networks, and data from unauthorized access, disruption, or damage by employing a layered defense strategy informed by a clear threat model. Core elements include strong authentication and authorization, cryptographic protections for data at rest and in transit, network segmentation and monitoring, timely patching and vulnerability management, and an incident response capability to detect, contain, and recover from breaches. Modern paradigms such as zero-trust architectures assume that no network location is inherently safe and instead require continuous verification of users and devices. Effective cybersecurity combines technical controls, threat intelligence, secure development lifecycle practices, and human-centered measures such as training and phishing-resistant authentication to mitigate risk.",

    "Oceanography studies the physical, chemical, biological, and geological processes of the world's oceans, which play a central role in climate regulation, biogeochemical cycles, and biodiversity. Circulation patterns—driven by wind, Earth's rotation, and density gradients due to temperature and salinity—transport heat and influence regional climates; phenomena like the thermohaline circulation and mesoscale eddies shape nutrient distribution and marine productivity. Marine ecosystems, from plankton communities to coral reefs and deep-sea habitats, are sensitive to changes in temperature, acidification from increased CO2, deoxygenation, and human activities such as overfishing and pollution. Conservation and sustainable management of marine resources require integrated observation systems, ecosystem-based fisheries management, and international cooperation to address transboundary challenges.",

    "Behavioral economics integrates insights from psychology into economic models to better describe how people actually make decisions under bounded rationality, limited attention, and often systematic biases. Heuristics and biases such as loss aversion, anchoring, present bias, and framing effects lead to predictable departures from the utility-maximizing agent of classical economics, with consequences for savings, health behaviors, and market outcomes. Prospect theory formalizes some of these observations by modeling value functions that are concave for gains, convex for losses, and steeper for losses than gains, explaining risk preferences in different domains. Policymakers and designers use these insights to craft nudges, choice architecture, and incentive designs that can improve welfare while preserving freedom of choice, but they must also consider ethical and distributional implications.",

    "Space exploration and exoplanet science combine astrophysics, planetary science, and advanced engineering to study worlds beyond Earth and to enable human and robotic exploration of the Solar System and beyond. Detection methods such as transit photometry, radial velocity, direct imaging, and microlensing have revealed thousands of exoplanets, and follow-up spectroscopy probes their atmospheres for composition and potential biosignatures. Mission architectures range from low-thrust electric propulsion and gravity assists to proposed high-Δv concepts for interstellar probes; technologies such as in-situ resource utilization, autonomous robotics, and sample-return systems are pivotal for sustainable exploration. Scientific goals include understanding planetary formation and evolution, characterizing habitability factors (liquid water, atmosphere, magnetic field), and preparing for long-term human presence through life-support and radiation-mitigation strategies.",
]

# -----------------------------
# initialize_graph_from_file (unchanged)
# -----------------------------
def initialize_graph_from_file(graph: ClaimGraph, path: str, verbose: bool = False) -> Dict:
    if not path or not os.path.exists(path):
        if verbose:
            print(f"[INIT] init graph path '{path}' not found; skipping initialization.")
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[INIT] failed to load init file {path}: {e}")
        return {}

    added = []
    text_to_id = {}
    idx_to_id = {}

    in_nodes = data.get('nodes', []) if isinstance(data.get('nodes', []), list) else []
    for i, n in enumerate(in_nodes):
        txt = clean(n.get('text') if isinstance(n, dict) else str(n))
        prov = n.get('provenance') if isinstance(n, dict) else None
        nid = graph.add_node(txt, provenance=prov)
        # allow overriding some metadata if provided
        try:
            if isinstance(n, dict):
                if 'support_weight' in n:
                    graph.nodes[nid].support_weight = float(n.get('support_weight', graph.nodes[nid].support_weight))
                if 'attack_weight' in n:
                    graph.nodes[nid].attack_weight = float(n.get('attack_weight', graph.nodes[nid].attack_weight))
                if 'confidence' in n:
                    graph.nodes[nid].confidence = float(n.get('confidence', graph.nodes[nid].confidence))
                if 'status' in n:
                    graph.nodes[nid].status = n.get('status', graph.nodes[nid].status)
                if 'revised_by' in n:
                    graph.nodes[nid].revised_by = n.get('revised_by')
        except Exception:
            pass
        added.append(nid)
        text_to_id[txt] = nid
        idx_to_id[i] = nid
        if verbose:
            print(f"[INIT-ADD] created node({nid}) '{txt[:80]}'")

    # Add edges
    in_edges = data.get('edges', []) if isinstance(data.get('edges', []), list) else []
    for e in in_edges:
        s = None
        t = None
        typ = 'related'
        score = 1.0
        if isinstance(e, dict):
            typ = e.get('type', typ)
            score = float(e.get('score', score)) if 'score' in e else score
            # try indices first
            if 'source_idx' in e and e.get('source_idx') in idx_to_id:
                s = idx_to_id[e.get('source_idx')]
            if 'target_idx' in e and e.get('target_idx') in idx_to_id:
                t = idx_to_id[e.get('target_idx')]
            # try text matching
            if s is None and 'source_text' in e:
                s = text_to_id.get(clean(e.get('source_text')))
                if s is None:
                    # fallback: best match
                    bid, bsc = graph.find_best_match(e.get('source_text'))
                    if bid is not None:
                        s = bid
            if t is None and 'target_text' in e:
                t = text_to_id.get(clean(e.get('target_text')))
                if t is None:
                    bid, bsc = graph.find_best_match(e.get('target_text'))
                    if bid is not None:
                        t = bid
        # if both found, create edge
        if s is not None and t is not None and s != t:
            graph.add_edge(s, t, typ=typ, score=score)
            if verbose:
                print(f"[INIT-EDGE] {s} -[{typ},{score}]-> {t}")
        else:
            if verbose:
                print(f"[INIT-EDGE-SKIP] could not resolve edge {e}")

    return {'added_node_ids': added, 'text_to_id': text_to_id, 'idx_to_id': idx_to_id}

# -----------------------------
# CLI and orchestration (added flag --adapter_dir, --init_graph, --preload_model_memory, --nli_model)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=3, help="max depth per raw subtree")
    p.add_argument("--branch", type=int, default=2, help="branching factor per node")
    p.add_argument("--same_threshold", type=float, default=0.7)
    p.add_argument("--related_threshold", type=float, default=0.5)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--use_4bit", action="store_true", default=True)
    p.add_argument("--use_8bit", action="store_true", default=False)
    p.add_argument("--adapter_dir", type=str, default="./out_trainer3_old/dpo_epoch1", help="Path to a folder containing a LoRA/PEFT adapter (saved with save_pretrained). If provided, the adapter will be loaded on top of the base model.")
    p.add_argument("--output", type=str, default="integrated_graph_revision_with_memory.json")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--allow_llm_supersede", action="store_true", default=False,
                   help="If set, allow LLM-driven judge supersede in addition to dataset-driven revisions (DEFAULT: disabled).")
    p.add_argument("--init_graph", type=str, default=None, help="Path to a JSON file describing initial nodes/edges to preload into the graph (model memory etc).")
    p.add_argument("--preload_model_memory", action='store_true', default=False, help="If set, create memory nodes describing the model name and basic characteristics in the graph at init.")
    p.add_argument("--nli_model", type=str, default="facebook/bart-large-mnli", help="HuggingFace NLI model to use (e.g. facebook/bart-large-mnli)")
    p.add_argument("--nli_device", type=str, default="auto", help="Device map for NLI model (e.g. 'auto' or '')")
    p.add_argument("--nli_thresh", type=float, default=NLI_DEFAULT_THRESHOLD, help="Threshold for NLI entail/contradict decisions (0-1)")
    p.add_argument("--backend", type=str, default="groq", choices=["local", "groq"],
                   help="LLM backend: 'local' (transformers) or 'groq' (API).")
    p.add_argument("--groq_api_key", type=str, default="[HIDDEN]",
                   help="Groq API key. If omitted, uses env GROQ_API_KEY.")
    p.add_argument("--groq_model", type=str, default="llama-3.1-8b-instant",
                   help="Groq model id (e.g., llama-3.1-8b-instant).")

    return p.parse_args()


def main():
    args = parse_args()
    try_load_policy()

    llm = LLMInterface(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        backend=args.backend,
        groq_api_key=args.groq_api_key,
        groq_model=args.groq_model,
    )
    try:
        llm.load(adapter_dir=args.adapter_dir if args.adapter_dir else None)
    except Exception as e:
        print(f"[WARN] LLM load failed unexpectedly: {e} -> using deterministic fallback.")
        llm.deterministic = True

    if args.backend == "local" and args.adapter_dir:
        if llm.adapter_loaded:
            print(f"[INFO] Adapter from {args.adapter_dir} applied to model.")
        else:
            print(f"[INFO] Adapter requested at {args.adapter_dir} but not applied (see warnings above).")


    # initialize NLI Judge
    nli_judge = NLIJudge(model_name=args.nli_model, device=args.nli_device)

    graph = ClaimGraph()

    # Initialize graph from provided JSON (memory)
    if args.init_graph:
        init_info = initialize_graph_from_file(graph, args.init_graph, verbose=args.verbose)
        if args.verbose:
            print(f"[INIT] populated graph with {len(init_info.get('added_node_ids', []))} nodes from {args.init_graph}")

    # Optionally preload model memory nodes (model name, quantization, adapter_loaded etc)
    if args.preload_model_memory:
        mem_texts = [
            f"Model: {args.model_name}",
            f"Quantized: use_4bit={args.use_4bit}, use_8bit={args.use_8bit}",
            f"Adapter path: {args.adapter_dir if args.adapter_dir else 'None'}",
            f"Adapter loaded: {llm.adapter_loaded}",
            "Capability note: This node records the model and adapter used to generate candidate claims."
        ]
        for t in mem_texts:
            nid = graph.add_node(t, provenance={"type": "memory", "time": time.time(), "source": "preload_model_memory"})
            if args.verbose:
                print(f"[MEMORY] added memory node({nid}) '{t[:80]}'")

    judge = SimpleJudge(sup_factor=1.0, min_delta=0.25)

    for t_idx, paragraph in enumerate(DEFAULT_PARAGRAPHS):
        if args.verbose:
            print("\n" + "="*40)
            print(f"[TOPIC {t_idx}] {paragraph}")
        raw_tree = build_raw_tree(llm, paragraph, max_depth=args.depth, branch_factor=args.branch,
                                  duplicate_sim=0.95, verbose=args.verbose)
        mapping = integrate_tree_into_graph(
            graph, raw_tree, topic_idx=t_idx, judge=judge,
            same_threshold=args.same_threshold,
            related_threshold=args.related_threshold,
            detect_contradiction=True, verbose=args.verbose,
            llm=llm, ground_truths=DEFAULT_PARAGRAPHS,
            dataset_authoritative=not args.allow_llm_supersede,
            nli_judge=nli_judge,
            nli_thresh=args.nli_thresh
        )
        if args.verbose:
            print(f"[INTEGRATED Topic {t_idx}] mapped {len(mapping)} raw nodes -> global nodes total={len(graph.nodes)}")

    out = {
        "params": {
            "depth": args.depth,
            "branch": args.branch,
            "same_threshold": args.same_threshold,
            "related_threshold": args.related_threshold,
            "model_name": args.model_name,
            "adapter_dir": args.adapter_dir if args.adapter_dir else None,
            "dataset_authoritative": True,
            "nli_model": args.nli_model,
        },
        "graph": graph.to_dict(),
        "generated_at": time.time(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {args.output} nodes={len(graph.nodes)} edges={len(graph.edges)} events={len(graph.events)}")
    print(f"[TRAIN_LOG] appended records to {TRAINING_LOG} (if any revisions occurred)")

if __name__ == "__main__":
    main()
