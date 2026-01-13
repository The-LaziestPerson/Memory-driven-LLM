#!/usr/bin/env python3
"""
Trainer3_fixed_with_eos_and_augmentation.py

Extended trainer that includes:
 - robust EOS handling (token-level preservation)
 - augmentation pipeline to make outputs unconditionally structured
   * converts unstructured (Style-B) outputs into structured E/H/V/C duplicates
   * creates helper-dropped variants from structured (Style-A) samples so the
     model learns to produce structured outputs even when helpers are absent
   * tag-noise and helper-shuffle utilities
 - augmentation is non-destructive: original samples are kept and augmented
   variants are appended (so Q4 = "no" policy is respected)
 - DPO uses structured outputs as positives; original unstructured outputs can
   be left as negatives or augmented too (configurable)
 - EOS-preservation and EOS-penalty during DPO maintained

CLI flags added for augmentation control:
  --augment                      enable augmentation (default True)
  --p_drop_helpers               prob to add a helper-dropped variant (default 0.5)
  --p_synthesize_unstructured    prob to synthesize structured dup for unstructured (default 1.0)
  --p_tag_noise                  prob to apply tag noise to ancestors (default 0.3)
  --seed                         RNG seed for reproducibility
  --keep_original_neg_as_negative  (default True) keep original neg as negative for DPO

Author: adapted by ChatGPT
"""
from __future__ import annotations
import argparse, json, os, random, copy, re, time
from typing import List, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# transformers / peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# AMP
from torch.cuda.amp import autocast, GradScaler

# -------------------------
# Utilities
# -------------------------

def load_jsonl(path: str):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def pad_to_length(input_ids: torch.Tensor, attention_mask: torch.Tensor, target_len: int, pad_token_id: int):
    cur_len = input_ids.size(1)
    if cur_len == target_len:
        return input_ids, attention_mask
    pad_len = target_len - cur_len
    input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
    return input_ids, attention_mask


# -------------------------
# Augmentation helpers
# -------------------------

def is_structured_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return ('hypothesis:' in t) or ('verification:' in t) or ('conclusion:' in t)


def split_sentences(text: str) -> List[str]:
    # simple, conservative sentence splitter that keeps semantics
    if not text:
        return []
    # replace newlines with spaces then split
    txt = re.sub(r"\s+", " ", text.strip())
    # split on sentence enders while keeping abbreviations naive handling
    parts = re.split(r'(?<=[\.!?])\s+', txt)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def choose_hypothesis_sentence(sentences: List[str]) -> int:
    # heuristics: prefer sentences with keywords likely indicating claims
    keywords = ['therefore', 'thus', 'hence', 'hypothesis', 'we propose', 'we hypothesize', 'suggests', 'implies', 'requires', 'ensures', 'forces', 'may', 'must', 'will', 'would', 'leads to', 'causes']
    for i, s in enumerate(sentences):
        ls = s.lower()
        for k in keywords:
            if k in ls:
                return i
    # prefer last sentence if it starts with a conclusion-like token
    last = sentences[-1] if sentences else ''
    if any(w in last.lower() for w in ['therefore', 'conclusion', 'we conclude', 'in conclusion']):
        return len(sentences)-1
    # default: first sentence
    return 0


def convert_flat_to_structured(flat_text: str) -> str:
    """
    Convert a flat paragraph into the canonical structured form:
    E (exposition, keep full paragraph)
    HYPOTHESIS: <pick or synthesize>
    VERIFICATION: <use remaining sentences as verification>
    CONCLUSION: <restatement>

    This converter is intentionally conservative: it does not invent novel domain facts,
    it reuses sentences from the input to build sections.
    """
    if not flat_text:
        return "\n\nHYPOTHESIS: \n\nVERIFICATION: \n\nCONCLUSION: \n"
    sentences = split_sentences(flat_text)
    if not sentences:
        return flat_text + '\n\nHYPOTHESIS: \n\nVERIFICATION: \n\nCONCLUSION: \n'

    # Exposition (E): keep original paragraph(s)
    exposition = flat_text.strip()

    # Hypothesis: choose a sentence conservatively
    idx_h = choose_hypothesis_sentence(sentences)
    hypothesis = sentences[idx_h]

    # Verification: use other sentences to justify; prefer sentences near the hypothesis
    verification_parts = [s for i,s in enumerate(sentences) if i != idx_h]
    if verification_parts:
        verification = ' '.join(verification_parts)
    else:
        # fallback: paraphrase exposition minimally
        verification = hypothesis

    # Conclusion: restate hypothesis succinctly with a safe prefix
    conclusion = hypothesis
    # if hypothesis already contains words like 'therefore' avoid duplication
    if not re.search(r'\b(therefore|thus|hence|in conclusion|we conclude)\b', conclusion, flags=re.I):
        conclusion = 'Therefore, ' + conclusion[0].lower() + conclusion[1:] if len(conclusion)>1 else 'Therefore.'

    structured = f"{exposition}\n\nHYPOTHESIS: {hypothesis}\n\nVERIFICATION: {verification}\n\nCONCLUSION: {conclusion}"
    return structured


def apply_tag_noise(ancestors: List[str], p_tag_noise: float = 0.3) -> List[str]:
    out = []
    for a in ancestors:
        if random.random() < p_tag_noise:
            # small noise: shorten tags or remove brackets or lowercase
            a2 = a
            a2 = re.sub(r"\[(Definition|Law|Constraint|Limit|Variable|Relation)\]", lambda m: '[' + m.group(1)[:3] + ']', a2)
            if random.random() < 0.3:
                a2 = a2.replace('[', '').replace(']', '')
            if random.random() < 0.2:
                a2 = a2.lower()
            out.append(a2)
        else:
            out.append(a)
    return out


def make_prefix_from_context(topic: str, ancestors: List[str]) -> str:
    if ancestors:
        formatted_ctx = (f"Topic: {topic}\n\nPrior hypotheses:\n" + "\n\n".join(ancestors)).strip()
    else:
        formatted_ctx = f"Topic: {topic}" if topic else ""
    prefix_full = f"Given the context below, evaluate the hypothesis.\n\n{formatted_ctx}\n\n"
    return prefix_full


def augment_sample(original: 'PairSample', tokenizer, pad_id: int, augment_cfg: Dict) -> List['PairSample']:
    """
    Return additional augmented PairSample objects (not replacing the original):
      - If original pos is unstructured -> add structured duplicate (converted)
      - If original is structured and has ancestors -> with probability p_drop_helpers add variant with helpers removed
      - With probability p_tag_noise apply tag noise to ancestors and add variant
    """
    out = []
    # keep original untouched (augmentation function returns only additional variants)
    topic = original.context.get('topic', '') if isinstance(original.context, dict) else ''
    ancestors = original.context.get('ancestors', []) if isinstance(original.context, dict) else []

    p_drop = augment_cfg.get('p_drop_helpers', 0.5)
    p_synth = augment_cfg.get('p_synthesize_unstructured', 1.0)
    p_tag_noise = augment_cfg.get('p_tag_noise', 0.3)

    # 1) If pos is unstructured, synthesize structured duplicate
    if not is_structured_text(original.pos):
        if random.random() < p_synth:
            s2 = copy.deepcopy(original)
            s2.pos = convert_flat_to_structured(original.pos)
            if s2.neg and not is_structured_text(s2.neg):
                s2.neg = convert_flat_to_structured(s2.neg)
            # rebuild textual fields
            prefix = make_prefix_from_context(topic, ancestors)
            s2.prefix_full = prefix + 'Hypothesis:\n'
            s2.pos_full = safe_ensure_eos_text(s2.prefix_full + (s2.pos or ''), tokenizer)
            s2.neg_full = safe_ensure_eos_text(s2.prefix_full + (s2.neg or ''), tokenizer)
            out.append(s2)

    # 2) If sample has ancestors (structured input), create helper-dropped variant
    if ancestors and random.random() < p_drop:
        s3 = copy.deepcopy(original)
        # drop ancestors from prefix but keep structured pos (so we teach model to produce structure without helpers)
        prefix = make_prefix_from_context(topic, [])
        s3.prefix_full = prefix + 'Hypothesis:\n'
        # keep pos full as structured text (if not structured, synthesize)
        if not is_structured_text(s3.pos):
            s3.pos = convert_flat_to_structured(s3.pos)
        if s3.neg and not is_structured_text(s3.neg):
            s3.neg = convert_flat_to_structured(s3.neg)
        s3.pos_full = safe_ensure_eos_text(s3.prefix_full + (s3.pos or ''), tokenizer)
        s3.neg_full = safe_ensure_eos_text(s3.prefix_full + (s3.neg or ''), tokenizer)
        out.append(s3)

    # 3) Tag noise variant
    if ancestors and random.random() < p_tag_noise:
        s4 = copy.deepcopy(original)
        noisy_anc = apply_tag_noise(ancestors, p_tag_noise=p_tag_noise)
        prefix = make_prefix_from_context(topic, noisy_anc)
        s4.prefix_full = prefix + 'Hypothesis:\n'
        if not is_structured_text(s4.pos):
            s4.pos = convert_flat_to_structured(s4.pos)
        if s4.neg and not is_structured_text(s4.neg):
            s4.neg = convert_flat_to_structured(s4.neg)
        s4.pos_full = safe_ensure_eos_text(s4.prefix_full + (s4.pos or ''), tokenizer)
        s4.neg_full = safe_ensure_eos_text(s4.prefix_full + (s4.neg or ''), tokenizer)
        out.append(s4)

    return out


# -------------------------
# Existing helpers (EOS tokenizer, dataset classes, etc.) remain unchanged
# -------------------------

def tokenize_preserve_eos(tokenizer, texts: List[str], max_length: int, pad_token_id: int, truncation_side: str = 'right'):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    encodings = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    token_lists = []
    for ids in encodings['input_ids']:
        ids = list(ids)
        if len(ids) > max_length:
            if truncation_side == 'right':
                ids = ids[:max_length]
            else:
                ids = ids[-max_length:]
        if eos_id is not None:
            if len(ids) == 0 or ids[-1] != eos_id:
                if len(ids) >= max_length:
                    ids[-1] = eos_id
                else:
                    ids.append(eos_id)
        token_lists.append(ids)
    max_len_batch = max(len(x) for x in token_lists)
    padded_ids = []
    attn_masks = []
    for ids in token_lists:
        attn = [1] * len(ids) + [0] * (max_len_batch - len(ids))
        padded = ids + [pad_token_id] * (max_len_batch - len(ids))
        padded_ids.append(padded)
        attn_masks.append(attn)
    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    attention_mask = torch.tensor(attn_masks, dtype=torch.long)
    return input_ids, attention_mask, token_lists


def safe_ensure_eos_text(text: str, tokenizer):
    if text is None:
        return ""
    text = text.rstrip()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_tok = getattr(tokenizer, "eos_token", None)
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > 0 and eos_id is not None and ids[-1] == eos_id:
            return text
    except Exception:
        pass
    if eos_tok:
        if not text.endswith(eos_tok):
            return text + eos_tok
    return text


# -------------------------
# Dataset / Sample (unchanged class definitions)
# -------------------------
class PairSample:
    def __init__(self, raw: Dict):
        self.raw = raw
        self.context = raw.get('context', {})
        self.pos = raw.get('accepted', '') or ''
        self.neg = raw.get('rejected', '') or ''
        self.E_acc = raw.get('E_acc', None)
        self.C_acc = raw.get('C_acc', None)
        self.G_acc = raw.get('G_acc', None)
        self.D_acc = raw.get('D_acc', None)
        self.H_acc = raw.get('H_acc', None)
        self.Q_acc = raw.get('Q_acc', None)
        self.E_rej = raw.get('E_rej', None)
        self.C_rej = raw.get('C_rej', None)
        self.G_rej = raw.get('G_rej', None)
        self.D_rej = raw.get('D_rej', None)
        self.H_rej = raw.get('H_rej', None)
        self.Q_rej = raw.get('Q_rej', None)
        self.Delta_Q = raw.get('Delta_Q', None)
        self.flag_reject = bool(raw.get('flag_reject', False))
        self.flag_low_reasoning = bool(raw.get('flag_low_reasoning', False))
        self.flag_hallucination = bool(raw.get('flag_hallucination', False))
        self.prefix_full = None
        self.pos_full = None
        self.neg_full = None
        self.example_len = 0


class PairDataset(Dataset):
    def __init__(self, samples: List[PairSample]):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# -------------------------
# Constraint head wrapper and other existing utilities (unchanged)
# -------------------------
class ConstraintLM(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int, constraint_k: int = 6, head_hidden: int = 512):
        super().__init__()
        self.model = base_model
        self.hidden_size = hidden_size
        self.constraint_k = constraint_k
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, constraint_k)
        )
    def predict_constraints_no_grad(self, input_ids, attention_mask=None, run_with_hook_fn=None):
        with torch.no_grad():
            if run_with_hook_fn is not None:
                out, last = run_with_hook_fn(self.model, input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                last = out.hidden_states[-1]
            if attention_mask is None:
                pooled = last.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(dtype=last.dtype)
                pooled = (last * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        logits = self.head(pooled)
        return logits
    def head_forward(self, pooled_features):
        return self.head(pooled_features)


def run_with_last_hidden_hook(model: AutoModelForCausalLM, **forward_kwargs):
    captured = {}
    handle = None
    block = None
    candidates = [
        ("transformer","h"),
        ("model","decoder","layers"),
        ("model","layers"),
        ("base_model","model","layers"),
        ("base_model","transformer","h"),
        ("model","transformer","h"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if not ok:
            continue
        try:
            if isinstance(obj, (list,tuple)) or hasattr(obj, "__len__"):
                block = obj[-1]
                break
        except Exception:
            block = None

    def hook_fn(module, input_, output_):
        _out = output_[0] if isinstance(output_, tuple) else output_
        try:
            captured["last"] = _out.detach() if isinstance(_out, torch.Tensor) else None
        except Exception:
            captured["last"] = None

    if block is not None:
        try:
            handle = block.register_forward_hook(hook_fn)
            out = model(**forward_kwargs, return_dict=True)
            if handle is not None:
                handle.remove()
            last = captured.get("last", None)
            if last is not None:
                return out, last
        except Exception:
            if handle is not None:
                try: handle.remove()
                except Exception: pass

    out = model(**forward_kwargs, output_hidden_states=True, return_dict=True)
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        last = out.hidden_states[-1]
        return out, last
    raise RuntimeError("Could not extract last hidden state. Inspect model structure.")


def compute_sequence_logprobs_from_logits(logits, input_ids, attn_mask, prompt_lens):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn_mask[:, 1:]
    B, Tm1 = shift_labels.shape
    include_mask = torch.zeros((B, Tm1), device=logits.device, dtype=shift_attn.dtype)
    for i, pl in enumerate(prompt_lens):
        start = max(int(pl.item()) - 1, 0)
        if start < Tm1:
            include_mask[i, start:] = 1
    final_mask = include_mask * shift_attn
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logp * final_mask).sum(dim=-1)
    return seq_logp


def lora_l1_penalty(model, l1_lambda: float):
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname or lname.endswith(".lora_a") or lname.endswith(".lora_b"):
            l1 = l1 + p.abs().sum()
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return l1 * l1_lambda


# -------------------------
# Trainer (design + implementation)
# -------------------------
def train(
    data_path: str,
    out_dir: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs_sft: int = 1,
    epochs_dpo: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    constraint_k: int = 6,
    alpha_sft: float = 0.2,
    alpha_dpo: float = 0.5,
    beta_start: float = 0.1,
    beta_target: float = 0.15,
    beta_warmup_steps: int = 2000,
    lambda_v: float = 0.7,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_ard_lora: bool = False,
    l1_lambda: float = 1e-6,
    max_len: int = 1024,
    quant_bits: int = 4,
    gradient_checkpointing: bool = True,
    ref_on_cpu: bool = True,
    grad_accum_steps: int = 4,
    gamma_margin: float = 1.0,
    constraint_penalty_weight: float = 1.0,
    flag_weights: Dict[str, float] = None,
    ema_decay: float = 0.999,
    ema_update_every: int = 100,
    eos_penalty_weight: float = 0.2,
    debug_eos: bool = False,
    augment: bool = True,
    p_drop_helpers: float = 0.5,
    p_synthesize_unstructured: float = 1.0,
    p_tag_noise: float = 0.3,
    seed: int = 1337,
    keep_original_neg_as_negative: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available() and device != 'cpu'
    device_t = torch.device(device if use_cuda else 'cpu')
    print("[train] device:", device_t)

    random.seed(seed)
    torch.manual_seed(seed)

    print('[train] loading tokenizer:', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, 'pad_token_id', None) is not None else (tokenizer.eos_token_id or 0)

    def make_bnb_config(bits: int):
        if bits == 16 or bits is None:
            return None
        if bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
        if bits == 4:
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        raise ValueError("quant bits must be 4, 8, or 16")

    qconf = make_bnb_config(quant_bits)

    def load_model(name, quant_conf=None):
        multi_gpu = torch.cuda.device_count() > 1
        device_map_arg = 'auto' if multi_gpu else None
        if quant_conf is not None:
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant_conf, device_map=device_map_arg, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map=device_map_arg, trust_remote_code=True)
        try:
            model.config.use_cache = False
        except Exception:
            pass
        if device_map_arg is None and device_t.type == 'cuda':
            try:
                model.to(device_t)
            except Exception:
                pass
        return model

    print('[train] loading base model')
    base_model = load_model(model_name, qconf)
    if quant_bits in (4,8):
        base_model = prepare_model_for_kbit_training(base_model)
    if gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False
        except Exception:
            pass

    # LoRA
    if target_modules is None:
        targets = ["q_proj","k_proj","v_proj","o_proj","wq","wk","wv","wo"]
    else:
        targets = target_modules
    print(f"[train] applying LoRA to {targets}")
    peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=targets, lora_dropout=lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM)
    try:
        peft_model = get_peft_model(base_model, peft_config)
    except Exception as e:
        print("LoRA apply failed, retrying with smaller target set. Error:", e)
        targets = ["q_proj","k_proj","v_proj","o_proj"]
        peft_config.target_modules = targets
        peft_model = get_peft_model(base_model, peft_config)

    hidden_size = peft_model.config.hidden_size
    constr_model = ConstraintLM(peft_model, hidden_size=hidden_size, constraint_k=constraint_k)
    constr_model.head = constr_model.head.to(device_t)

    # reference model: deepcopy base model weights; keep on CPU if requested
    print('[train] creating reference model (deepcopy)')
    ref_model = copy.deepcopy(constr_model.model)
    try:
        ref_device = 'cpu' if ref_on_cpu else device_t
        ref_model.to(ref_device)
    except Exception:
        pass
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # load data
    print('[train] loading dataset:', data_path)
    raw = load_jsonl(data_path)
    samples = []
    for r in raw:
        s = PairSample(r)
        # canonicalize context prefix from context dict
        formatted_ctx = ''
        if isinstance(s.context, dict):
            topic = s.context.get('topic', '')
            ancestors = s.context.get('ancestors', []) or []
            prefix_full = make_prefix_from_context(topic, ancestors) + 'Hypothesis:\n'
        else:
            topic = ''
            ancestors = []
            prefix_full = make_prefix_from_context(topic, []) + 'Hypothesis:\n'
        s.prefix_full = prefix_full
        # keep text-level SFT strings for compatibility but token-level EOS will be enforced at tokenization time
        s.pos_full = safe_ensure_eos_text(prefix_full + (s.pos or ''), tokenizer)
        s.neg_full = safe_ensure_eos_text(prefix_full + (s.neg or ''), tokenizer)
        try:
            s.example_len = len(tokenizer(s.pos_full, add_special_tokens=False)["input_ids"]) + len(tokenizer(s.neg_full, add_special_tokens=False)["input_ids"]) if s.pos_full and s.neg_full else 0
        except Exception:
            s.example_len = 0
        samples.append(s)

        # augmentation (non-destructive): create additional training variants
        if augment:
            augment_cfg = {'p_drop_helpers': p_drop_helpers, 'p_synthesize_unstructured': p_synthesize_unstructured, 'p_tag_noise': p_tag_noise}
            extra = augment_sample(s, tokenizer, pad_id, augment_cfg)
            for sx in extra:
                try:
                    # recompute example_len for augmented
                    sx.example_len = len(tokenizer(sx.pos_full, add_special_tokens=False)["input_ids"]) + (len(tokenizer(sx.neg_full, add_special_tokens=False)["input_ids"]) if sx.neg_full else 0)
                except Exception:
                    sx.example_len = 0
                samples.append(sx)

    random.shuffle(samples)
    n = len(samples)
    print(f"[train] loaded {n} pairs (including augmented)")

    # collate for SFT: supervise generation tokens only. Use token-level EOS-preserving tokenizer.
    def collate_pairs_sft(batch: List[PairSample]):
        pos_texts, neg_texts, prefix_cache, pos_cons, neg_cons = [], [], [], [] , []
        for s in batch:
            pos_texts.append(s.pos_full)
            neg_texts.append(s.neg_full)
            prefix_cache.append(s.prefix_full)
            pos_cons.append([s.E_acc, s.C_acc, s.G_acc, s.D_acc, s.H_acc, s.Q_acc])
            neg_cons.append([s.E_rej, s.C_rej, s.G_rej, s.D_rej, s.H_rej, s.Q_rej])

        enc_pos_ids, enc_pos_attn, pos_lists = tokenize_preserve_eos(tokenizer, pos_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right')
        enc_neg_ids, enc_neg_attn, neg_lists = tokenize_preserve_eos(tokenizer, neg_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right')

        prefix_token_lengths = []
        for p, token_list in zip(prefix_cache, pos_lists):
            try:
                pl = len(tokenizer(p, add_special_tokens=False)['input_ids'])
            except Exception:
                pl = 0
            pl = min(pl, len(token_list))
            prefix_token_lengths.append(pl)

        prompt_lens = torch.tensor(prefix_token_lengths, dtype=torch.long)

        enc_pos = {'input_ids': enc_pos_ids, 'attention_mask': enc_pos_attn}
        enc_neg = {'input_ids': enc_neg_ids, 'attention_mask': enc_neg_attn}
        return enc_pos, enc_neg, prompt_lens, pos_cons, neg_cons

    sft_loader = DataLoader(PairDataset(samples), batch_size=batch_size, shuffle=True, collate_fn=collate_pairs_sft)

    trainable_params = [p for n,p in constr_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=(device_t.type == 'cuda'))
    bce_loss = nn.BCEWithLogitsLoss()

    if flag_weights is None:
        flag_weights = {'reject':1.0, 'low_reasoning':0.4, 'hallucination':0.3}

    # -------------------------
    # SFT phase
    # -------------------------
    print('[train] START SFT')
    total_sft_steps = epochs_sft * len(sft_loader)
    sft_step = 0
    for epoch in range(1, epochs_sft+1):
        constr_model.model.train(); constr_model.train()
        epoch_loss = 0.0; micro = 0
        for enc_pos, enc_neg, prompt_lens, pos_cons_list, neg_cons_list in tqdm(sft_loader, desc=f'SFT epoch {epoch}'):
            input_ids_pos = enc_pos['input_ids'].to(device_t); attn_pos = enc_pos['attention_mask'].to(device_t)
            input_ids_neg = enc_neg['input_ids'].to(device_t); attn_neg = enc_neg['attention_mask'].to(device_t)
            max_len_batch = max(input_ids_pos.size(1), input_ids_neg.size(1))
            input_ids_pos, attn_pos = pad_to_length(input_ids_pos, attn_pos, max_len_batch, pad_id)
            input_ids_neg, attn_neg = pad_to_length(input_ids_neg, attn_neg, max_len_batch, pad_id)
            input_ids_both = torch.cat([input_ids_pos, input_ids_neg], dim=0)
            attn_both = torch.cat([attn_pos, attn_neg], dim=0)
            labels_both = input_ids_both.clone()
            Bpos = input_ids_pos.size(0)
            prompt_lens_both = torch.cat([prompt_lens, prompt_lens], dim=0).to(device_t)
            for idx in range(labels_both.size(0)):
                plen = int(prompt_lens_both[idx].item())
                if plen > 0:
                    labels_both[idx, :plen] = -100
            with autocast(enabled=(device_t.type=='cuda')):
                out_both, last_hidden = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both, attention_mask=attn_both, labels=labels_both)
                lm_loss_pos = out_both.loss
                mask = attn_both.unsqueeze(-1).to(dtype=last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
                pooled_pos = pooled[:Bpos].detach(); pooled_neg = pooled[Bpos:].detach()
                pred_pos_logits = constr_model.head_forward(pooled_pos)
                pred_neg_logits = constr_model.head_forward(pooled_neg)
                def make_targets(cons_list):
                    out = []
                    mask = []
                    for item in cons_list:
                        if item is None:
                            out.append([0.0]*constraint_k); mask.append(0.0)
                        else:
                            v = list(item[:constraint_k]) if isinstance(item, (list,tuple)) else [0.0]*constraint_k
                            out.append([float(x or 0.0) for x in v]); mask.append(1.0)
                    return torch.tensor(out, dtype=pred_pos_logits.dtype, device=pred_pos_logits.device), torch.tensor(mask, dtype=torch.float32, device=pred_pos_logits.device)
                tgt_pos, mask_pos = make_targets(pos_cons_list)
                tgt_neg, mask_neg = make_targets(neg_cons_list)
                cons_loss = torch.tensor(0., device=device_t, dtype=pred_pos_logits.dtype)
                cnt = 0
                if mask_pos.sum() > 0:
                    valid_idxs = (mask_pos == 1.0).nonzero(as_tuple=False).squeeze(-1)
                    cons_loss = cons_loss + bce_loss(pred_pos_logits[valid_idxs], tgt_pos[valid_idxs]); cnt += 1
                if mask_neg.sum() > 0:
                    valid_idxs = (mask_neg == 1.0).nonzero(as_tuple=False).squeeze(-1)
                    cons_loss = cons_loss + bce_loss(pred_neg_logits[valid_idxs], tgt_neg[valid_idxs]); cnt += 1
                if cnt>0:
                    cons_loss = cons_loss / cnt
                else:
                    cons_loss = torch.tensor(0., device=device_t)
                curr_alpha_sft = alpha_sft
                raw_loss = lm_loss_pos + curr_alpha_sft * cons_loss
                if use_ard_lora and l1_lambda > 0:
                    raw_loss = raw_loss + lora_l1_penalty(constr_model, l1_lambda)
            if not torch.isfinite(raw_loss):
                optimizer.zero_grad(set_to_none=True); continue
            loss_to_backward = raw_loss / max(1, grad_accum_steps)
            if device_t.type == 'cuda':
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            micro += 1; epoch_loss += float(raw_loss.detach().cpu()); sft_step += 1
            if (micro % grad_accum_steps) == 0:
                if device_t.type == 'cuda':
                    try: scaler.unscale_(optimizer)
                    except Exception: pass
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    try: scaler.step(optimizer); scaler.update()
                    except Exception:
                        try: optimizer.step()
                        except Exception: pass
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        print(f"[SFT] epoch {epoch} avg_loss={(epoch_loss/max(1,micro)):.6f}")
        epoch_dir = os.path.join(out_dir, f"sft_epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        try:
            constr_model.model.save_pretrained(epoch_dir); tokenizer.save_pretrained(epoch_dir)
            torch.save({'head_state': constr_model.head.state_dict()}, os.path.join(epoch_dir, 'constraint_head.pt'))
        except Exception:
            torch.save({k:v.cpu() for k,v in constr_model.state_dict().items()}, os.path.join(epoch_dir, 'full_state.pt'))

    # -------------------------
    # DPO phase: reward-matching with ΔQ margin
    # -------------------------
    print('[train] START DPO')
    constr_model.model.eval(); constr_model.eval()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    pair_items = samples
    random.shuffle(pair_items)
    pair_items.sort(key=lambda x: x.example_len)
    global_step = 0
    micro = 0
    dpo_opt_steps = 0

    for epoch in range(1, epochs_dpo+1):
        random.shuffle(pair_items); pair_items.sort(key=lambda x: x.example_len)
        pbar = tqdm(range(0, len(pair_items), batch_size), desc=f'DPO epoch {epoch}')
        for i in pbar:
            batch = pair_items[i:i+batch_size]
            pos_full_texts=[]; neg_full_texts=[]; prefix_lens_full=[]
            delta_q_list = []
            flags_batch = []
            for s in batch:
                pos_full_texts.append(s.pos_full); neg_full_texts.append(s.neg_full)
                try:
                    prefix_lens_full.append(len(tokenizer(s.prefix_full, add_special_tokens=False)['input_ids']))
                except Exception:
                    prefix_lens_full.append(0)
                delta_q_list.append(0.0 if s.Delta_Q is None else float(s.Delta_Q))
                flags_batch.append({'reject': s.flag_reject, 'low_reasoning': s.flag_low_reasoning, 'hallucination': s.flag_hallucination})

            input_ids_pos_full, attn_pos_full, pos_lists = tokenize_preserve_eos(
                tokenizer, pos_full_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right'
            )
            input_ids_neg_full, attn_neg_full, neg_lists = tokenize_preserve_eos(
                tokenizer, neg_full_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right'
            )
            clipped_prefix_lens = [min(pl, len(ids)) for pl, ids in zip(prefix_lens_full, pos_lists)]

            input_ids_pos_full = input_ids_pos_full.to(device_t)
            attn_pos_full = attn_pos_full.to(device_t)
            input_ids_neg_full = input_ids_neg_full.to(device_t)
            attn_neg_full = attn_neg_full.to(device_t)
            prompt_lens_full = torch.tensor(clipped_prefix_lens, dtype=torch.long, device=device_t)

            max_len_full = max(input_ids_pos_full.size(1), input_ids_neg_full.size(1))
            input_ids_pos_full, attn_pos_full = pad_to_length(input_ids_pos_full, attn_pos_full, max_len_full, pad_id)
            input_ids_neg_full, attn_neg_full = pad_to_length(input_ids_neg_full, attn_neg_full, max_len_full, pad_id)
            input_ids_both_full = torch.cat([input_ids_pos_full, input_ids_neg_full], dim=0)
            attn_both_full = torch.cat([attn_pos_full, attn_neg_full], dim=0)
            prompt_lens_both_full = torch.cat([prompt_lens_full, prompt_lens_full], dim=0)

            constr_model.model.train()
            with autocast(enabled=(device_t.type=='cuda')):
                out_both_theta_full, last_hidden_theta_full = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both_full, attention_mask=attn_both_full)
                logits_both_theta_full = out_both_theta_full.logits
                logp_both_theta_full = compute_sequence_logprobs_from_logits(logits_both_theta_full, input_ids_both_full, attn_both_full, prompt_lens_both_full)
                B_full = input_ids_pos_full.size(0)
                logp_pos_theta_full = logp_both_theta_full[:B_full]; logp_neg_theta_full = logp_both_theta_full[B_full:]

            ref_device = next(ref_model.parameters()).device
            with torch.no_grad():
                input_ids_pos_ref = input_ids_pos_full.cpu(); attn_pos_ref = attn_pos_full.cpu()
                input_ids_neg_ref = input_ids_neg_full.cpu(); attn_neg_ref = attn_neg_full.cpu()
                input_ids_both_ref = torch.cat([input_ids_pos_ref, input_ids_neg_ref], dim=0)
                attn_both_ref = torch.cat([attn_pos_ref, attn_neg_ref], dim=0)

                ref_batch_size = max(1, 4)
                logits_ref_list = []
                for start in range(0, input_ids_both_ref.size(0), ref_batch_size):
                    end = min(input_ids_both_ref.size(0), start + ref_batch_size)
                    ids_chunk = input_ids_both_ref[start:end].to(ref_device)
                    attn_chunk = attn_both_ref[start:end].to(ref_device)
                    out_ref = ref_model(input_ids=ids_chunk, attention_mask=attn_chunk, return_dict=True)
                    logits_chunk = out_ref.logits.detach().cpu()
                    logits_ref_list.append(logits_chunk)
                logits_both_ref_full = torch.cat(logits_ref_list, dim=0).to(device_t)
                logp_both_ref_full = compute_sequence_logprobs_from_logits(logits_both_ref_full, input_ids_both_full, attn_both_full, prompt_lens_both_full)
                logp_pos_ref_full = logp_both_ref_full[:B_full]; logp_neg_ref_full = logp_both_ref_full[B_full:]

            delta_logp_full = logp_pos_theta_full - logp_neg_theta_full
            d_theta_adj_full = delta_logp_full - lambda_v * torch.tensor(0.0, device=device_t)
            d_ref_full = logp_pos_ref_full - logp_neg_ref_full
            diff_full = d_theta_adj_full - d_ref_full

            delta_q_tensor = torch.tensor([min(max(d, -1.0), 1.0) for d in delta_q_list], dtype=torch.float32, device=device_t)
            scaled_full = (beta_target if global_step >= beta_warmup_steps else (beta_start + (beta_target - beta_start) * (global_step / max(1, beta_warmup_steps)))) * (diff_full - gamma_margin * delta_q_tensor)
            scaled_full = torch.clamp(scaled_full, min=-50.0, max=50.0)
            per_example_loss_full = -F.logsigmoid(scaled_full)
            loss_full = per_example_loss_full.mean()

            cons_penalty = torch.tensor(0.0, device=device_t)
            cnt_pen = 0
            for idx, s in enumerate(batch):
                e = 0.0 if s.E_acc is None else float(s.E_acc)
                c = 0.0 if s.C_acc is None else float(s.C_acc)
                g = 0.5 if s.G_acc is None else float(s.G_acc)
                d = 0.5 if s.D_acc is None else float(s.D_acc)
                h = 0.0 if s.H_acc is None else float(s.H_acc)
                pen = 0.25 * c + 0.35 * h + 0.2 * (1.0 - e) + 0.2 * (1.0 - d)
                cons_penalty = cons_penalty + pen; cnt_pen += 1
            if cnt_pen > 0:
                cons_penalty = cons_penalty / cnt_pen
            else:
                cons_penalty = torch.tensor(0.0, device=device_t)

            eos_id = getattr(tokenizer, "eos_token_id", None)
            eos_pen = 0.0
            if eos_id is not None:
                missing = 0.0
                for ids in pos_lists:
                    if len(ids) == 0 or ids[-1] != eos_id:
                        missing += 1.0
                for ids in neg_lists:
                    if len(ids) == 0 or ids[-1] != eos_id:
                        missing += 1.0
                total_checked = max(1.0, float(len(pos_lists) + len(neg_lists)))
                eos_pen = (missing / total_checked)
                cons_penalty = cons_penalty + eos_pen * eos_penalty_weight

            avg_flag_weight = 0.0
            for flags in flags_batch:
                w = 1.0
                if flags.get('low_reasoning', False):
                    w *= (1.0 - flag_weights.get('low_reasoning', 0.4))
                if flags.get('hallucination', False):
                    w *= (1.0 - flag_weights.get('hallucination', 0.3))
                if flags.get('reject', False):
                    w *= flag_weights.get('reject', 1.0)
                avg_flag_weight += w
            if len(flags_batch) > 0:
                avg_flag_weight = avg_flag_weight / len(flags_batch)
            else:
                avg_flag_weight = 1.0

            raw_loss = (alpha_dpo * loss_full) * avg_flag_weight + constraint_penalty_weight * cons_penalty
            if use_ard_lora and l1_lambda > 0:
                raw_loss = raw_loss + lora_l1_penalty(constr_model, l1_lambda)

            if not torch.isfinite(raw_loss):
                pbar.set_postfix({'dpo_raw_loss': float('nan')}); continue
            loss_to_backward = raw_loss / max(1, grad_accum_steps)
            if device_t.type == 'cuda':
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            micro += 1
            if (micro % grad_accum_steps) == 0:
                if device_t.type == 'cuda':
                    try: scaler.unscale_(optimizer)
                    except Exception: pass
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    try: scaler.step(optimizer); scaler.update()
                    except Exception:
                        try: optimizer.step()
                        except Exception: pass
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                dpo_opt_steps += 1
                if (dpo_opt_steps % ema_update_every) == 0:
                    try:
                        with torch.no_grad():
                            src_sd = constr_model.model.state_dict()
                            tgt_sd = ref_model.state_dict()
                            for k in tgt_sd.keys():
                                if k in src_sd:
                                    tgt_sd[k].mul_(ema_decay).add_(src_sd[k].to(tgt_sd[k].device), alpha=(1.0 - ema_decay))
                            ref_model.load_state_dict(tgt_sd)
                    except Exception:
                        pass

            global_step += 1
            if debug_eos:
                try:
                    with torch.no_grad():
                        logits = logits_both_theta_full[0]
                        last_pos_idx = (input_ids_both_full[0] != pad_id).nonzero(as_tuple=False).squeeze(-1).max().item()
                        eos_tok_id = eos_id
                        logp = F.log_softmax(logits, dim=-1)[last_pos_idx-1, eos_tok_id].item() if last_pos_idx>0 else None
                        pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_target), 'eos_logp_first': logp})
                except Exception:
                    pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_target)})
            else:
                pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_target)})
            if device_t.type == 'cuda':
                torch.cuda.empty_cache()

        epoch_dir = os.path.join(out_dir, f'dpo_epoch{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        try:
            constr_model.model.save_pretrained(epoch_dir); tokenizer.save_pretrained(epoch_dir)
            torch.save({'head_state': constr_model.head.state_dict()}, os.path.join(epoch_dir, 'constraint_head.pt'))
        except Exception:
            torch.save({k:v.cpu() for k,v in constr_model.state_dict().items()}, os.path.join(epoch_dir, 'full_state.pt'))
        print(f"[DPO] saved epoch {epoch} to {epoch_dir}")

    print('[train] training complete. adapters + head saved to', out_dir)


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='clone.jsonl')
    p.add_argument('--out_dir', type=str, default='out_trainer3_with_eos')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--epochs_sft', type=int, default=1)
    p.add_argument('--epochs_dpo', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--constraint_k', type=int, default=6)
    p.add_argument('--alpha_sft', type=float, default=0.2)
    p.add_argument('--alpha_dpo', type=float, default=0.5)
    p.add_argument('--beta_start', type=float, default=0.1)
    p.add_argument('--beta_target', type=float, default=0.15)
    p.add_argument('--beta_warmup_steps', type=int, default=2000)
    p.add_argument('--lambda_v', type=float, default=0.7)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--quant_bits', type=int, default=4, choices=[4,8,16])
    p.add_argument('--grad_ckpt', action='store_true')
    p.add_argument('--ref_on_cpu', action='store_true')
    p.add_argument('--gamma_margin', type=float, default=1.0, help='scale of ΔQ margin inside DPO')
    p.add_argument('--constraint_penalty_weight', type=float, default=1.0)
    p.add_argument('--use_ard_lora', default=True)
    p.add_argument('--l1_lambda', type=float, default=1e-6)
    p.add_argument('--eos_penalty_weight', type=float, default=0.2)
    p.add_argument('--debug_eos', action='store_true')
    p.add_argument('--augment', action='store_true', help='enable augmentation (default False in CLI unless specified)')
    p.add_argument('--p_drop_helpers', type=float, default=0.5)
    p.add_argument('--p_synthesize_unstructured', type=float, default=1.0)
    p.add_argument('--p_tag_noise', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--keep_original_neg_as_negative', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args =
