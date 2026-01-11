#!/usr/bin/env python3
"""
inference simulation (LoRA-only mode, GNN removed)

Two-phase pipeline:
1) ingest all topic chains (derive/merge/related) into a global ClaimGraph
2) expand each topic starting *strictly* from the deepest pre-existing node and
   generate `--depth` steps forward (chain mode). Optionally create siblings if
   --branch > 1 (they are not expanded further).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Set, Union

# HF cache defaults (adjust if you want)
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")

# Optional HF imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

# optional PEFT (LoRA) imports
try:
    from peft import PeftModel
except Exception:
    PeftModel = None

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
# Configurable thresholds / hyperparams
# -----------------------------
MIN_PREDICATE_OVERLAP = 0.35
PRESERVATION_TOKEN_OVERLAP = 0.35
MAX_TOPIC_DRIFT_SIM_DROP = 0.4
NEW_NODE_SUPPORT_PRIOR = 0.08
POLICY_RETRAIN_EVERY = 5
RETRAIN_EVERY = POLICY_RETRAIN_EVERY
GROUND_TRUTH_SIM_THRESHOLD = 0.7

# -----------------------------
# Utilities
# -----------------------------
def clean(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip())

def toks_lower(s: str):
    return re.findall(r"\w+", (s or "").lower())

def tokens(text: str) -> List[str]:
    return re.findall(r"\w{3,}", (text or "").lower())

def token_set(text: str) -> Set[str]:
    return set(tokens(text))

def symmetric_similarity(a: str, b: str) -> float:
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
    return set(re.findall(r"\w{4,}", (text or "").lower()))

# -----------------------------
# Policy loader & features
# -----------------------------
POLICY_PATH = "policy_clf.joblib"
policy = None
POLICY_META = "policy_meta.json"

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
    seen = []
    for w in t:
        if w not in seen:
            seen.append(w)
        if len(seen) >= 50:
            break
    for w in seen:
        feats["u_"+w] = c.get(w, 0)
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
# LLM wrapper (kept from prior)
# -----------------------------
@dataclass
class LLMInterface:
    model_name: Optional[str]
    use_4bit: bool = False
    use_8bit: bool = False
    use_lora: bool = False
    lora_path: Optional[str] = None
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    deterministic: bool = False
    _bad_word_ids: Optional[List[List[int]]] = field(default_factory=list)

    def load(self):
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            self.deterministic = True
            print("[INFO] transformers/torch not available — using deterministic fallback.")
            return

        tokenizer_source = None
        if self.use_lora and self.lora_path and os.path.isdir(self.lora_path):
            tok_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"}
            has_tok = any(os.path.exists(os.path.join(self.lora_path, f)) for f in tok_files)
            if has_tok:
                tokenizer_source = self.lora_path
        if tokenizer_source is None:
            tokenizer_source = self.model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[INFO] Loaded tokenizer from {tokenizer_source} (vocab_size={len(self.tokenizer)})")
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
                    qconf = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
                else:
                    qconf = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        llm_int8_enable_fp32_cpu_offload=False,
                    )
            except Exception as e:
                print("[WARN] failed to create BitsAndBytesConfig:", e)
                qconf = None

        load_kwargs = {"device_map": "auto"}
        dtype_arg = {"torch_dtype": torch.float16} if torch is not None and torch.cuda.is_available() else {}

        tried = []
        if qconf is not None and self.model_name:
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
                self.model = None

        if self.model is None and self.model_name:
            try:
                print("[INFO] Attempting non-quantized load (dtype=float16 if available) ...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                    **dtype_arg,
                )
                print("[INFO] Model loaded (non-quantized).")
            except Exception as e:
                tried.append(("non-quantized", e))
                print("[WARN] non-quantized load failed:", e)
                self.model = None

        if self.model is None and self.model_name:
            try:
                print("[INFO] Attempting CPU-only load ...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map={"": "cpu"})
                print("[INFO] Model loaded (CPU-only).")
            except Exception as e:
                tried.append(("cpu", e))
                print("[WARN] cpu-only load failed:", e)
                self.model = None

        if self.model is None:
            print("[ERROR] All model load attempts failed. Falling back to deterministic generator.")
            for name, exc in tried:
                print(f" - attempt '{name}': {type(exc).__name__}: {exc}")
            self.deterministic = True
            return

        if self.use_lora and self.lora_path:
            if PeftModel is None:
                print("[WARN] PEFT (peft.PeftModel) not available — cannot load LoRA adapters.")
            else:
                try:
                    print(f"[INFO] Attempting to load LoRA adapters from: {self.lora_path}")
                    self.model = PeftModel.from_pretrained(self.model, self.lora_path, device_map="auto")
                    print("[INFO] LoRA adapters successfully applied.")
                except Exception as e:
                    print("[WARN] failed to load/apply LoRA adapters:", e)
                    print("[WARN] continuing with base model (LoRA not applied).")

        try:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception:
            pass

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 48,
                 temperature: float = 0.7,
                 return_metrics: bool = False,
                 metrics_device: Optional[str] = None,
                 scorer_kwargs: Optional[Dict] = None) -> Union[str, Tuple[str, Dict]]:
        """
        Generate text. By default returns str. If return_metrics=True and
        the scorer (score_dataset.OfflineScorer) is available, returns (text, metrics_dict).

        scorer_kwargs can contain keys forwarded to OfflineScorer initializer:
          e.g. {'nli_model_name': 'roberta-large-mnli', 'embed_model_name': 'sentence-transformers/all-MiniLM-L6-v2'}
        """
        prompt = clean(prompt)
        if self.deterministic or self.model is None or self.tokenizer is None:
            out = self._fallback_generate(prompt)
            if not return_metrics:
                return out
            else:
                return out, {'Q_acc': None, 'flag_reject': False, 'flag_low_reasoning': False, 'flag_hallucination': False}

        toks = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        try:
            device = next(self.model.parameters()).device
            toks = {k: v.to(device) for k, v in toks.items()}
        except Exception:
            device = 'cpu'

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        generate_kwargs = {
            **toks,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": float(temperature),
            "repetition_penalty": 1.1,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "return_dict_in_generate": True,
            "output_scores": False,
        }

        if getattr(self, "_bad_word_ids", None):
            generate_kwargs["bad_words_ids"] = self._bad_word_ids

        try:
            # generation
            with torch.inference_mode():
                out = self.model.generate(**generate_kwargs)
            # if out is a ModelOutput (dict-like) produced by return_dict_in_generate,
            # handle both cases (HF older/newer)
            if hasattr(out, "sequences"):
                seq = out.sequences[0]
            else:
                # older .generate returns tensor
                seq = out[0]
            prompt_len = toks["input_ids"].shape[1]
            gen_ids = seq[prompt_len:].tolist()
            if eos_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos_id)]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        except Exception as e:
            print("[WARN] model.generate failed:", e)
            txt = self._fallback_generate(prompt)

        if not return_metrics:
            return txt

        # ---- lazy scorer import/instantiate ----
        # we avoid importing the scorer at module import time because it loads heavy models.
        scorer = None
        try:
            from score_dataset import OfflineScorer  # your scorer file
            # determine device for scorer: pass through metrics_device or choose GPU if available
            mdev = metrics_device or ('cuda' if torch.cuda.is_available() else 'cpu')
            sargs = scorer_kwargs or {}
            scorer = OfflineScorer(nli_model_name=sargs.get('nli_model_name', 'roberta-large-mnli'),
                                   embed_model_name=sargs.get('embed_model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                                   device=mdev,
                                   cache_dir=sargs.get('cache_dir', None),
                                   batch_size=sargs.get('batch_size', 16))
        except Exception as e:
            # scorer not available / dependencies missing -> warn and return None metrics
            if return_metrics:
                print("[WARN] OfflineScorer not available or failed to init; returning empty metrics. Error:", e)
                return txt, {'Q_acc': None, 'E_acc': None, 'C_acc': None, 'G_acc': None, 'D_acc': None, 'H_acc': None,
                             'flag_reject': False, 'flag_low_reasoning': False, 'flag_hallucination': False}

        # build example dict expected by scorer: context must be passed by caller in prompt,
        # but inference.py often builds prompt from ancestors+topic. Here we attempt to reconstruct a minimal context:
        # NOTE: caller can instead call llm.generate(..., return_metrics=False) and call scorer.score_example themselves.
        # We'll try a heuristic: split prompt header to extract 'Prior hypotheses' block if present.
        ctx_obj = {'topic': None, 'ancestors': []}
        try:
            # naive parse: look for 'Prior hypotheses:' marker and extract preceding lines as topic
            if 'Prior hypotheses:' in prompt:
                # topic block: before 'Prior hypotheses:'
                pre, post = prompt.split('Prior hypotheses:', 1)
                # try to find 'Topic:' in pre
                m = re.search(r'Topic:\\s*(.*)\\n', pre)
                if m:
                    ctx_obj['topic'] = m.group(1).strip()
                # attempt to extract ancestor bullets in `post` until 'Task:' or 'The next hypothesis:'
                anc_block = re.split(r'Task:|The next hypothesis:', post, maxsplit=1)[0]
                # split ancestor lines for scorer format
                anc_lines = [line.strip('- ').strip() for line in anc_block.splitlines() if line.strip()]
                ctx_obj['ancestors'] = [a for a in anc_lines if a]
        except Exception:
            ctx_obj = {'topic': None, 'ancestors': []}

        # call scorer on a single example
        try:
            example = {'context': ctx_obj, 'accepted': txt, 'rejected': None}
            res = scorer.score_example(example)
        except Exception as e:
            print("[WARN] scorer.score_example failed:", e)
            res = {'Q_acc': None, 'E_acc': None, 'C_acc': None, 'G_acc': None, 'D_acc': None, 'H_acc': None,
                   'flag_reject': False, 'flag_low_reasoning': False, 'flag_hallucination': False}

        return txt, res

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
# Prompt builder (full ancestor chain supported)
# -----------------------------
def build_context_prompt(ancestors: Union[str, List[str]], topic: Optional[str] = None) -> str:
    topic_block = f"Topic: {clean(topic)}\n\n" if topic else ""
    if isinstance(ancestors, (list, tuple)):
        if len(ancestors) == 0:
            ancestors_block = ""
        else:
            ancestors_block = "\n".join([f"- {clean(a)}" for a in ancestors])
    else:
        ancestors_block = clean(ancestors or "")

    prompt = f"""
Given the context below, evaluate the hypothesis.

{topic_block}
    
Prior hypotheses:
{ancestors_block}

Hypothesis:
"""
    return prompt

# -----------------------------
# RawTree, ClaimGraph, integration, and other helpers
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

@dataclass
class ClaimNode:
    id: int
    text: str
    created_at: float
    provenance: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    support_weight: float = 0.0
    attack_weight: float = 0.0
    status: str = "ACTIVE"
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

# DEFAULT_TREE and DEFAULT_PARAGRAPHS - keep full content in your repo
DEFAULT_TREE = [
    [
        "Light propagates through heterogeneous media, undergoing wavelength-dependent absorption and scattering. Given optical transport models, consider what hypotheses can be justified about spatial color shifts.",
        "[Definition] Light is a physical phenomenon enabling visual perception and energy transport.",
        "[Axiom] Light travels in straight lines in homogeneous media (geometric optics).",
      "[Interaction] Light–matter interactions include reflection, transmission, absorption, and scattering.",
      "[Observation] White light separates into wavelengths when passed through a prism (Newton).",
      "[Model] White light is a superposition of monochromatic waves with wavelength-dependent properties.",
      "[Law] Snell’s law relates angles of incidence and refraction via n₁ sinθᵢ = n₂ sinθᵣ.",
      "[Dispersion] The refractive index satisfies n = n(λ) and dn/dλ ≠ 0 for real materials.",
      "[Complex Index] Real materials require ñ(λ)=n(λ)+iκ(λ) to model attenuation.",
      "[Absorption] Beer–Lambert law: I(λ,x)=I₀(λ)exp[−α(λ)x], α(λ)=4πκ(λ)/λ.",
      "[Limit] Pure absorption fails for scattering-dominated media (fog, milk, tissue).",
      "[Scattering] Introduce σₐ(λ), σₛ(λ), and σₜ(λ)=σₐ+σₛ.",
      "[Radiative Transfer] Spectral radiance obeys the RTE with extinction, in-scattering, and sources.",
      "[Regime] σₛ(λ) scaling: Rayleigh (λ⁻⁴) vs. Mie (size-dependent).",
      "[Effect] Blue wavelengths scatter more strongly, altering apparent color.",
      "[Measurement] I_out(λ)=I_in(λ)exp[−σₜ(λ)d]T_sys(λ).",
      "[Derived] Spectral centroid λ_c characterizes color but is non-unique.",
      "[Strategy] Vary path length and concentration to separate absorption vs scattering.",
      "[Inverse Problem] Estimating σₐ and σₛ from radiance measurements is ill-posed and requires constraints."
    ],
    [   # Topic 1: glass of water / temperature / perception
        "Temperature",
        "A glass of water at temperature T = 25°C feels cool to hand H when touched.",
        "Human tactile perception of 'cool' depends on the temperature gradient, skin thermal conductivity, and prior thermal adaptation of H.",
        "If H were chronically adapted to colder climates, the threshold for perceived coolness would shift downward and transient impressions would be dominated by rate-of-change rather than absolute temperature; state an empirical test to measure that adaptation."
    ],
    [   # Topic 2: car fuel consumption
        "Kinematics",
        "Car C travels distance d = 100 km with fuel consumption f = 7 L/100km under baseline conditions.",
        "Fuel consumption f depends on aerodynamic drag, rolling resistance, engine efficiency, and average speed profile over the trip.",
        "When terrain or weather change (e.g., prolonged inclines or strong headwinds), the dominant alteration to f will be from increased rolling or aerodynamic work respectively; specify how to isolate these contributions from trip telemetry."
    ],
    [   # Topic 3: plant photosynthesis
        "Photosynthesis in plants",
        "Plant P absorbs sunlight S and carbon dioxide CO2 to produce oxygen O2 via photosynthesis.",
        "Photosynthetic rate is constrained by light intensity, CO2 concentration at stomata, and enzymatic limits of RuBisCO and downstream metabolism.",
        "Therefore, changing S or CO2 independently produces differing nonlinear responses in O2 output; design an experimental manipulation that separates light-limited from CO2-limited regimes."
    ],
    [   # Topic 4: falling ball / vacuum vs atmosphere
        "Ball B dropped from height h = 2 m hits the ground in time t ≈ 0.64 s in Earth's atmosphere.",
        "In vacuum, only gravity acts and air drag is absent, but in atmosphere drag depends on velocity, cross-sectional area, and fluid density.",
        "Thus increasing h substantially changes the terminal versus transient regime: in vacuum impact time scales purely with sqrt(h), whereas in atmosphere high h can push B into drag-dominated descent; compare predicted arrival times for both regimes."
    ],
    [   # Topic 5: sound perception across species
        "Sound wave W with frequency f = 440 Hz is heard by ear E as musical note A by human listeners.",
        "Different species have different cochlear tuning and auditory range, which change perceived pitch salience and harmonic resolution.",
        "If E belonged to a species with higher-frequency hearing sensitivity, the perceptual mapping of harmonic structure would change and the same 440 Hz signal may not correspond to their 'A' reference; propose a cross-species psychoacoustic comparison."
    ],
]
DEFAULT_PARAGRAPHS = [s for chain in DEFAULT_TREE for s in chain]

# -----------------------------
# Helper functions used in integration (kept)
# -----------------------------
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

def read_training_log(path: str = None) -> List[Dict]:
    if path is None:
        path = "revisions_train.jsonl"
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

# -----------------------------
# build_raw_tree (legacy use for creating initial chain only)
# -----------------------------
def build_raw_tree(llm: Optional[LLMInterface],
                   paragraph_or_chain: Union[str, List[str]],
                   max_depth: int = 3,
                   branch_factor: int = 3,
                   duplicate_sim: float = 0.95,
                   verbose: bool = False,
                   topic: Optional[str] = None) -> RawTree:
    """
    Creates a RawTree from a paragraph or a predefined chain.
    If llm is None, this function **only** creates the initial nodes (no generation).
    """
    tree = RawTree()
    chain_ids = []
    topic_text = None

    if isinstance(paragraph_or_chain, (list, tuple)):
        if len(paragraph_or_chain) == 0:
            return tree
        topic_text = clean(paragraph_or_chain[0])
        topic_nid = tree.add(topic_text, parent_id=None)
        chain_ids.append((topic_nid, 0))
        prev = topic_nid
        for idx, txt in enumerate(paragraph_or_chain[1:], start=1):
            node_text = clean(txt)
            nid = tree.add(node_text, parent_id=prev)
            chain_ids.append((nid, idx))
            prev = nid
        if verbose:
            for nid, idx in chain_ids:
                prefix = "[TOPIC]" if idx == 0 else f"[CHAIN d={idx}]"
                print(f"{prefix} ({nid}) {tree.nodes[nid].text}")
    else:
        root = clean(paragraph_or_chain)
        if topic:
            topic_text = clean(topic)
            topic_nid = tree.add(topic_text, parent_id=None)
            root_id = tree.add(root, parent_id=topic_nid)
            chain_ids = [(topic_nid, 0), (root_id, 1)]
            if verbose:
                print(f"[TOPIC] ({topic_nid}) {tree.nodes[topic_nid].text}")
                print(f"[ROOT under topic] ({root_id}) {tree.nodes[root_id].text}")
        else:
            root_id = tree.add(root, parent_id=None)
            chain_ids = [(root_id, 0)]
            if verbose:
                print(f"[ROOT] {root}")
    # if llm provided, you may want generation in other contexts; here we return the initial tree
    if llm is None:
        return tree

    # else fall back to the original behavior if user explicitly calls build_raw_tree with llm
    return tree

# -----------------------------
# New helper: compute depths for nodes in a RawTree
# -----------------------------
def compute_rawtree_depths(tree: RawTree) -> Dict[int, int]:
    depths = {}
    for node in tree.nodes:
        d = 0
        cur = node
        while cur.parent_id is not None:
            d += 1
            cur = tree.nodes[cur.parent_id]
        depths[node.id] = d
    return depths

# -----------------------------
# New helper: expand an existing RawTree with strict deepest-start chain expansion
# -----------------------------
def expand_existing_raw_tree(llm: LLMInterface,
                             tree: RawTree,
                             topic_text: Optional[str] = None,
                             max_depth: int = 3,
                             branch_factor: int = 1,
                             duplicate_sim: float = 0.95,
                             policy_threshold: float = 0.06,
                             verbose: bool = False) -> RawTree:
    """
    Expand the given RawTree in-place using the LLM.

    Strict rules implemented:
    - Identify the deepest pre-existing node (base deepest) computed from nodes already in `tree` at function entry.
    - Only start expansion from that deepest pre-existing node.
    - `max_depth` = number of forward generation steps to perform from the base deepest node.
    - The main chain creates exactly ONE main child per step (ensures strict forward derivation).
    - If branch_factor > 1, additional sibling children are created at each step but are NOT expanded further.
    """
    if llm is None:
        return tree

    def collect_ancestors(node_id: int) -> List[str]:
        anc = []
        cur = node_id
        while cur is not None:
            anc.append(tree.nodes[cur].text)
            cur = tree.nodes[cur].parent_id
        anc.reverse()
        return anc

    def is_leaf(node_id: int) -> bool:
        for n in tree.nodes:
            if n.parent_id == node_id:
                return False
        return True

    # Record original size and depths to identify pre-existing nodes
    original_count = len(tree.nodes)
    if original_count == 0:
        return tree

    depths = compute_rawtree_depths(tree)

    # Base depth and deepest base node (if multiple pick newest by created_at)
    base_nodes = list(range(original_count))
    base_depth = max(depths[nid] for nid in base_nodes)
    base_candidates = [nid for nid in base_nodes if depths[nid] == base_depth]
    base_parent = max(base_candidates, key=lambda nid: getattr(tree.nodes[nid], "created_at", 0.0))

    if verbose:
        print(f"[EXPAND] base_depth={base_depth} base_parent={base_parent} text='{tree.nodes[base_parent].text[:80]}...'")

    current_parent = base_parent

    # We'll run exactly `max_depth` forward steps (stop earlier if generation can't produce valid candidate)
    for step in range(max_depth):
        parent_depth = depths.get(current_parent, 0)
        child_depth = parent_depth + 1

        # build prompt using full ancestor chain up to current_parent
        ancestors_list = collect_ancestors(current_parent)
        prompt = build_context_prompt(ancestors=ancestors_list, topic=topic_text)
        if verbose:
            print("\n----------------\nEXPANSION STEP", step + 1, "OF", max_depth)
            print(f"Expanding parent_id={current_parent} (depth={parent_depth})")
            print(prompt)
            print("---------------------------")

        # Attempt to generate a valid main candidate (retry up to N times)
        main_cand = None
        MAIN_ATTEMPTS = 1
        for attempt in range(MAIN_ATTEMPTS):
            cand_raw = llm.generate(prompt, max_new_tokens=1024, temperature=0.7)
            if not cand_raw or not cand_raw.strip():
                if verbose:
                    print(f"[MAIN-TRY {attempt+1}] empty generation, retrying...")
                continue
            # policy check
            if policy is not None:
                pscore = policy_score(cand_raw)
                if pscore < policy_threshold:
                    if verbose:
                        print(f"[MAIN-TRY {attempt+1}] policy rejected (score={pscore:.3f}), retrying...")
                    continue
            # duplicate check vs entire tree
            if any(symmetric_similarity(cand_raw, n.text) > duplicate_sim for n in tree.nodes):
                if verbose:
                    print(f"[MAIN-TRY {attempt+1}] duplicate to existing node, retrying...")
                continue
            main_cand = cand_raw
            break

        if main_cand is None:
            if verbose:
                print(f"[STOP] failed to generate a valid main candidate at step {step+1}; stopping expansion.")
            break

        # Add main child (this is the canonical chain)
        main_id = tree.add(main_cand, parent_id=current_parent)
        depths[main_id] = child_depth
        if verbose:
            print(f"[ADD-MAIN] id={main_id} parent={current_parent} depth={child_depth} '{main_cand}")

        # Optionally create sibling children (they are not expanded further)
        if branch_factor and branch_factor > 1:
            # create siblings as additional children of the same parent (current_parent)
            SIB_ATTEMPTS = 2
            for bi in range(branch_factor - 1):
                sib_cand = None
                for attempt in range(SIB_ATTEMPTS):
                    sib_raw = llm.generate(prompt, max_new_tokens=512, temperature=0.9)
                    if not sib_raw or not sib_raw.strip():
                        continue
                    if policy is not None:
                        pscore = policy_score(sib_raw)
                        if pscore < policy_threshold:
                            continue
                    if any(symmetric_similarity(sib_raw, n.text) > duplicate_sim for n in tree.nodes):
                        continue
                    sib_cand = sib_raw
                    break
                if sib_cand:
                    sid = tree.add(sib_cand, parent_id=current_parent)
                    depths[sid] = child_depth
                    if verbose:
                        print(f"[ADD-SIB] id={sid} parent={current_parent} depth={child_depth} '{sib_cand[:120]}...'")
                else:
                    if verbose:
                        print(f"[SIB-TRY] could not generate sibling #{bi+1} for parent {current_parent}")

        # Move forward along the main chain
        current_parent = main_id

    return tree

# -----------------------------
# integrate_tree_into_graph (kept from previous implementation)
# -----------------------------
def integrate_tree_into_graph(graph: ClaimGraph, raw_tree: RawTree, topic_idx: int,
                              judge: 'SimpleJudge',
                              same_threshold: float = 0.85, related_threshold: float = 0.5,
                              detect_contradiction: bool = True, verbose: bool = False,
                              llm: Optional[LLMInterface] = None,
                              ground_truths: Optional[List[str]] = None,
                              dataset_authoritative: bool = True) -> Dict[int, int]:
    """
    Add/merge/related nodes from raw_tree into the global ClaimGraph.
    Returns mapping raw_node_id -> global_node_id.
    """
    mapping: Dict[int, int] = {}
    flat = raw_tree.flatten()
    if ground_truths is None:
        ground_truths = DEFAULT_PARAGRAPHS

    # First pass: add/merge/related
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

        # basic contradiction detection (kept)
        if detect_contradiction and mapping.get(raw_id) is not None and mapping.get(raw_parent) is not None:
            # optional domain-specific checks could be placed here
            pass

    # derive edges (respect raw_parent relationships)
    for raw_id, raw_parent, text in flat:
        if raw_parent is not None:
            src = mapping.get(raw_parent)
            tgt = mapping.get(raw_id)
            if src is None or tgt is None:
                continue
            if src != tgt:
                graph.add_edge(src, tgt, typ='derived', score=1.0)
                graph.nodes[tgt].support_weight += graph.nodes[src].confidence
                if verbose:
                    print(f"[DERIVED] global({src}) -> global({tgt}) (from raw {raw_parent}->{raw_id})")

    # Additional dataset-authoritative revision / LLM judge passes can be inserted as before if desired.
    return mapping

# -----------------------------
# SimpleJudge and other helpers
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
# CLI and orchestration (2-phase flow)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4, help="number of forward generation steps from the deepest pre-existing node")
    p.add_argument("--branch", type=int, default=1, help="branching factor per step (siblings per step). main chain = 1 child per step; branch-1 siblings optional.")
    p.add_argument("--same_threshold", type=float, default=0.7)
    p.add_argument("--related_threshold", type=float, default=0.5)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--use_4bit", action="store_true", default=True)
    p.add_argument("--use_8bit", action="store_true", default=False)
    p.add_argument("--output", type=str, default="integrated_graph_revision_fixed.json")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--allow_llm_supersede", action="store_true", default=False)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--lora_path", type=str, default="out_trainer3_old/dpo_epoch1")
    p.add_argument("--lora_only", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    try_load_policy()

    # infer base model if lora_only
    inferred_model_name = args.model_name
    if args.use_lora and args.lora_path and args.lora_only and (not args.model_name):
        found = False
        for fname in ("adapter_config.json", "peft_config.json", "peft_config.jsonc"):
            pth = os.path.join(args.lora_path, fname)
            if os.path.exists(pth):
                try:
                    with open(pth, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    bm = cfg.get("base_model_name_or_path") or cfg.get("base_model") or cfg.get("base_model_name")
                    if bm:
                        inferred_model_name = bm
                        print(f"[INFO] inferred base model name from {pth}: {inferred_model_name}")
                        found = True
                        break
                except Exception as e:
                    print(f"[WARN] failed to read {pth}: {e}")
        if not found and not args.model_name:
            print("[ERROR] --lora_only requested but no peft/adapter config with base model name found in lora_path. Provide --model_name or include adapter peft config.")
            return

    # instantiate LLM (but we will first do ingestion without calling LLM)
    llm = LLMInterface(
        model_name=inferred_model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_lora=args.use_lora,
        lora_path=args.lora_path if args.use_lora else None,
    )
    try:
        llm.load()
    except Exception as e:
        print(f"[WARN] LLM load failed unexpectedly: {e} -> using deterministic fallback.")
        llm.deterministic = True

    graph = ClaimGraph()
    judge = SimpleJudge(sup_factor=1.0, min_delta=0.25)

    # -------------------------
    # PHASE 1: ingest all topics (create RawTrees without LLM generation, then integrate them)
    # -------------------------
    raw_trees: List[RawTree] = []
    mappings_per_topic: List[Dict[int, int]] = []

    if args.verbose:
        print("\n[PHASE 1] Ingesting all topics (no LLM generation) and integrating into graph.")

    for t_idx, chain in enumerate(DEFAULT_TREE):
        if args.verbose:
            print("\n" + "="*40)
            print(f"[INGEST TOPIC {t_idx}] {chain[0] if isinstance(chain, list) else str(chain)[:120]}")
        # create raw tree nodes only (no LLM calls)
        raw_tree = build_raw_tree(None, chain, max_depth=args.depth, branch_factor=args.branch, verbose=args.verbose)
        raw_trees.append(raw_tree)

        # integrate initial raw tree into graph now (so all topic nodes are in graph before generation)
        mapping = integrate_tree_into_graph(
            graph, raw_tree, topic_idx=t_idx, judge=judge,
            same_threshold=args.same_threshold,
            related_threshold=args.related_threshold,
            detect_contradiction=True, verbose=args.verbose,
            llm=None,  # no regenerate at ingest time
            ground_truths=DEFAULT_PARAGRAPHS,
            dataset_authoritative=not args.allow_llm_supersede
        )
        mappings_per_topic.append(mapping)
        if args.verbose:
            print(f"[INGESTED Topic {t_idx}] mapped {len(mapping)} raw nodes -> global nodes total={len(graph.nodes)}")

    # -------------------------
    # PHASE 2: expand each topic at its deepest pre-existing node using LLM, then integrate newly generated nodes
    # -------------------------
    if args.verbose:
        print("\n[PHASE 2] Expanding each topic starting from deepest pre-existing node using the LLM and integrating new nodes.")

    for t_idx, raw_tree in enumerate(raw_trees):
        if args.verbose:
            print("\n" + "-"*30)
            print(f"[EXPAND TOPIC {t_idx}] starting expansion (steps={args.depth}, branch={args.branch})")
        # compute topic_text (first node is topic root)
        topic_text = raw_tree.nodes[0].text if raw_tree.nodes else None

        # expand in-place using LLM (strict deepest-start chain expansion)
        expand_existing_raw_tree(
            llm=llm,
            tree=raw_tree,
            topic_text=topic_text,
            max_depth=args.depth,
            branch_factor=args.branch,
            duplicate_sim=0.95,
            policy_threshold=0.06,
            verbose=args.verbose
        )

        # integrate newly expanded raw nodes back into the graph
        mapping = integrate_tree_into_graph(
            graph, raw_tree, topic_idx=t_idx, judge=judge,
            same_threshold=args.same_threshold,
            related_threshold=args.related_threshold,
            detect_contradiction=True, verbose=args.verbose,
            llm=llm,  # allow regenerate/taint handling if you want
            ground_truths=DEFAULT_PARAGRAPHS,
            dataset_authoritative=not args.allow_llm_supersede
        )
        if args.verbose:
            print(f"[EXPANDED & INTEGRATED Topic {t_idx}] mapped {len(mapping)} raw nodes -> global nodes total={len(graph.nodes)}")

    # -------------------------
    # Save output
    # -------------------------
    out = {
        "params": {
            "depth": args.depth,
            "branch": args.branch,
            "same_threshold": args.same_threshold,
            "related_threshold": args.related_threshold,
            "model_name": inferred_model_name,
            "dataset_authoritative": True,
            "use_lora": args.use_lora,
            "lora_path": args.lora_path if args.use_lora else None,
        },
        "graph": graph.to_dict(),
        "generated_at": time.time(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {args.output} nodes={len(graph.nodes)} edges={len(graph.edges)} events={len(graph.events)}")

if __name__ == "__main__":
    main()
