#!/usr/bin/env python3
"""
Optimized DPO trainer with step-local DPO and efficiency improvements.

Major optimizations:
- Pre-build step-pair strings at dataset load time (no repeated splits).
- Per-batch deduplication of prompts and single forward per unique prompt.
- EMA reference model updated periodically (instead of a fixed separate pretrained ref).
- Subsampling of step-pairs (keep first and last, random sample rest).
- Sorting/bucketing DPO items by token length to reduce padding waste.
- Vectorized loss computation; no per-step Python loops.

Usage: same CLI as before. New/changed defaults chosen for efficiency but keep learning strength.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")

import random
import time
import re
import copy
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Transformers + PEFT + bitsandbytes
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception as e:
    raise RuntimeError("Please install transformers and bitsandbytes: pip install transformers bitsandbytes") from e

try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
except Exception as e:
    raise RuntimeError("Please install peft: pip install peft") from e

from torch.cuda.amp import autocast, GradScaler

# -------------------------
# Utils
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
# Dataset classes (unchanged interface)
# -------------------------
class PairSample:
    def __init__(self, context: Any, pos: str, neg: str, pos_constraints=None, neg_constraints=None):
        self.context = context
        self.pos = pos
        self.neg = neg
        self.pos_constraints = pos_constraints
        self.neg_constraints = neg_constraints
        # OPTIM: placeholders for prebuilt step pairs (filled at load)
        self.step_pairs: List[Tuple[str, str, str]] = []  # list of (local_prefix, pos_next, neg_next)
        self.prefix_full: Optional[str] = None
        self.pos_full: Optional[str] = None
        self.neg_full: Optional[str] = None
        self.example_len = 0  # rough token count for bucketing

class PairDataset(Dataset):
    def __init__(self, samples: List[PairSample]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------
# Prompt/context helpers (unchanged except small helpers)
# -------------------------
def ensure_eos(text: str, tokenizer):
    if text is None:
        return ""
    text = text.rstrip()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_str = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "sep_token", None)
    try:
        if eos_id is not None:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > 0 and ids[-1] == eos_id:
                return text
            if eos_str:
                return text + eos_str
            return text
    except Exception:
        pass
    if eos_str is not None:
        if not text.endswith(eos_str):
            return text + eos_str
        return text
    return text

def format_context_with_topic(context_obj: Any) -> str:
    if context_obj is None:
        return "Prior hypotheses:\n"
    if isinstance(context_obj, str):
        return f"Topic : \n\nPrior hypotheses:\n{context_obj}"
    if not isinstance(context_obj, dict):
        return f"Topic : \n\nPrior hypotheses:\n{str(context_obj)}"
    topic = str(context_obj.get("topic", "") or "").strip()
    ancestors = context_obj.get("ancestors", []) or []
    ancestors_block = "\n\n".join(a for a in ancestors if a)
    if topic:
        if ancestors_block:
            return f"Topic : {topic}\n\nPrior hypotheses:\n{ancestors_block}"
        else:
            return f"Topic : {topic}\n\nPrior hypotheses:\n"
    else:
        return f"Prior hypotheses:\n{ancestors_block}"

def _enforce_eos_on_encoding(enc: Dict[str, torch.Tensor], tokenizer, pad_id: int):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return enc
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    B, L = ids.shape
    for b in range(B):
        row_attn = attn[b]
        nonpad_idxs = (row_attn == 1).nonzero(as_tuple=True)[0]
        if len(nonpad_idxs) == 0:
            if L > 0:
                ids[b, 0] = eos_id
                attn[b, 0] = 1
            continue
        last_idx = int(nonpad_idxs[-1].item())
        last_token = int(ids[b, last_idx].item())
        if last_token == eos_id:
            continue
        if last_idx + 1 < L and attn[b, last_idx + 1].item() == 0:
            ids[b, last_idx + 1] = eos_id
            attn[b, last_idx + 1] = 1
        else:
            if len(nonpad_idxs) > 1:
                ids[b, last_idx] = eos_id
            else:
                pass
    enc["input_ids"] = ids
    enc["attention_mask"] = attn
    return enc

# -------------------------
# Model wrapper (unchanged)
# -------------------------
class ConstraintLM(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int, constraint_k: int = 6, head_hidden: int = 512):
        super().__init__()
        self.model = base_model
        self.hidden_size = hidden_size
        self.constraint_k = constraint_k
        head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, constraint_k)
        )
        try:
            model_dtype = next(self.model.parameters()).dtype
            head = head.to(dtype=model_dtype)
        except Exception:
            pass
        self.head = head

    def lm_forward_with_labels(self, **forward_kwargs):
        return self.model(**forward_kwargs, return_dict=True)

    def head_forward(self, pooled_features):
        try:
            target_dtype = next(self.head[0].parameters()).dtype
            if pooled_features.dtype != target_dtype:
                pooled_features = pooled_features.to(dtype=target_dtype)
        except Exception:
            pass
        logits = self.head_forward_orig(pooled_features) if hasattr(self, "head_forward_orig") else self.head(pooled_features)
        return logits

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

# -------------------------
# Helpers: capture last hidden & compute logprobs
# -------------------------
def run_with_last_hidden_hook(model: AutoModelForCausalLM, **forward_kwargs):
    captured = {}
    handle = None
    block = None

    candidates = [
        ("transformer", "h"),
        ("model", "decoder", "layers"),
        ("model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "transformer", "h"),
        ("model", "transformer", "h"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if not ok:
            continue
        try:
            if isinstance(obj, (list, tuple)) or hasattr(obj, "__len__"):
                block = obj[-1]
                break
        except Exception:
            block = None

    if block is None:
        for name, module in model.named_modules():
            if name.endswith(".layers") or name.endswith(".h") or name.endswith(".blocks") or name.endswith("encoder"):
                try:
                    children = list(module.children())
                    if len(children) > 0:
                        block = children[-1]
                        break
                except Exception:
                    block = None

    def hook_fn(module, input_, output_):
        if isinstance(output_, tuple):
            _out = output_[0]
        else:
            _out = output_
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
    else:
        raise RuntimeError("Could not capture last hidden state via hook or hidden_states. Inspect model structure.")
    return out, last

def compute_sequence_logprobs_from_logits(logits, input_ids, attn_mask, prompt_lens):
    """
    logits:      [B, T, V]
    input_ids:   [B, T]
    attn_mask:   [B, T]
    prompt_lens: [B]
    """

    # Shift
    shift_logits = logits[:, :-1, :]          # [B, T-1, V]
    shift_labels = input_ids[:, 1:]           # [B, T-1]
    shift_attn   = attn_mask[:, 1:]            # [B, T-1]

    B, Tm1 = shift_labels.shape

    # Build include_mask directly at T-1
    include_mask = torch.zeros(
        (B, Tm1),
        device=logits.device,
        dtype=shift_attn.dtype,
    )

    for i, pl in enumerate(prompt_lens):
        # We want to include tokens AFTER the prompt
        # prompt token index pl corresponds to label index pl-1
        start = max(pl - 1, 0)
        if start < Tm1:
            include_mask[i, start:] = 1

    final_mask = include_mask * shift_attn  # ✅ shapes now match

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    seq_logp = (token_logp * final_mask).sum(dim=-1)
    return seq_logp


# -------------------------
# Utilities (unchanged)
# -------------------------
def token_set(text: str):
    return set(re.findall(r"\w{3,}", (text or "").lower()))

def symmetric_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    at = token_set(a); bt = token_set(b)
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    ra = inter / max(1, len(at))
    rb = inter / max(1, len(bt))
    return (ra + rb) / 2.0

def hardness_score_for_pair(item):
    _, pos, neg, posc, negc = item
    sim = symmetric_similarity(pos, neg)
    cons_diff = 0.0
    try:
        pos_sum = sum(posc) if posc else 0.0
        neg_sum = sum(negc) if negc else 0.0
        cons_diff = abs(pos_sum - neg_sum)
    except Exception:
        cons_diff = 0.0
    return 0.7 * sim + 0.3 * (1.0 - min(1.0, cons_diff))

# -------------------------
# Step-splitting & mutation (kept from prior design)
# -------------------------
def split_text_into_steps(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in re.split(r'\n(?=\d+\.)', text) if p.strip()]
    if len(parts) > 1:
        return parts
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) <= 1:
        return [text.strip()]
    out = []
    i = 0
    while i < len(sents):
        group_len = 1
        if len(sents[i]) < 40 and i+1 < len(sents):
            group_len = 2
        step = " ".join(sents[i:i+group_len])
        out.append(step)
        i += group_len
    return out

def mutate_textual_negative(step_text: str, context: Optional[str] = None) -> str:
    text = step_text.strip()
    if len(text) > 10 and random.random() < 0.3:
        return f"{text} Therefore, it necessarily follows that the effect is always present."
    t = re.sub(r'\b(may|might|can)\b', 'necessarily', text, flags=re.IGNORECASE)
    if t != text and random.random() < 0.6:
        return t
    if random.random() < 0.25:
        return text + " In fact this implies the quantity diverges to infinity."
    t2 = text.replace("increase", "decrease").replace("decreases", "increases")
    if t2 != text and random.random() < 0.4:
        return t2
    return text + " However, this implies the opposite effect, contradicting the prior step."

# Build step pairs but now store in sample object at load-time (OPTIM)
def build_step_pairs_from_sample_string(ctx_block: str, pos_text: str, neg_text: Optional[str],
                                 max_step_pairs: int = 6, mutate_neg_prob: float = 0.3) -> List[Tuple[str,str,str]]:
    pos_steps = split_text_into_steps(pos_text)
    neg_steps = split_text_into_steps(neg_text) if neg_text is not None else []
    step_pairs = []
    # Candidate indices: always include 0 and last, plus random sample of others
    if len(pos_steps) == 0 and pos_text.strip():
        pos_steps = [pos_text.strip()]
    indices = list(range(len(pos_steps)))
    if not indices:
        return []
    picks = []
    if 0 in indices:
        picks.append(0)
    if len(indices) > 1 and (len(indices)-1) not in picks:
        picks.append(len(indices)-1)
    remaining = [i for i in indices if i not in picks]
    random.shuffle(remaining)
    for idx in remaining[:max(0, max_step_pairs - len(picks))]:
        picks.append(idx)
    picks = sorted(list(dict.fromkeys(picks)))  # unique & sorted
    for k in picks:
        prefix_steps = " \n".join(pos_steps[:k]) if k > 0 else ""
        pos_next = pos_steps[k]
        if k < len(neg_steps):
            neg_next = neg_steps[k]
        else:
            if random.random() < mutate_neg_prob:
                neg_next = mutate_textual_negative(pos_next)
            else:
                neg_next = neg_text if neg_text else mutate_textual_negative(pos_next)
        # Build local prompt (we'll add context prefix later)
        step_pairs.append((prefix_steps, pos_next, neg_next))
        if len(step_pairs) >= max_step_pairs:
            break
    if len(step_pairs) == 0 and pos_text.strip():
        pos_next = pos_text.strip()
        neg_next = neg_text.strip() if neg_text and neg_text.strip() else mutate_textual_negative(pos_next)
        step_pairs.append(("", pos_next, neg_next))
    return step_pairs

# -------------------------
# Small helper: dedupe and forward unique texts, returns logits mapping in original order (OPTIM)
# -------------------------
def forward_unique_texts(model, tokenizer, texts: List[str], device, max_len=1024):
    """
    Tokenize texts, deduplicate identical strings, run forward once per unique string, and
    return logits list aligned with input `texts` order.
    """
    if len(texts) == 0:
        return []
    # dedupe
    uniq_map = {}
    uniq_list = []
    order_keys = []
    for t in texts:
        k = t  # string key; if inputs are large, consider hashing
        if k not in uniq_map:
            uniq_map[k] = len(uniq_list)
            uniq_list.append(t)
        order_keys.append(uniq_map[k])
    # tokenize uniq_list as a batch
    enc = tokenizer(uniq_list, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    enc = _enforce_eos_on_encoding(enc, tokenizer, tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = out.logits.detach().cpu()
    # expand to original order
    res = [logits[idx].to(device) for idx in order_keys]
    return res

# -------------------------
# EMA update helper (OPTIM)
# -------------------------
def ema_update(target_model, source_model, decay):
    with torch.no_grad():
        sd_src = source_model.state_dict()
        sd_tgt = target_model.state_dict()
        for k in sd_tgt.keys():
            if k in sd_src:
                sd_tgt[k].mul_(decay).add_(sd_src[k], alpha=1-decay)
        target_model.load_state_dict(sd_tgt)

# -------------------------
# TRAINER (main)
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
    max_len: int = 1024,
    quant_bits: int = 4,
    gradient_checkpointing: bool = True,
    ref_on_cpu: bool = False,
    sft_frac: float = 0.75,
    near_miss_frac: float = 0.30,
    dpo_strength: float = 0.5,
    grad_accum_steps: int = 4,
    # NEW args
    lambda_step: float = 2.0,
    lambda_seq: float = 0.5,
    max_step_pairs: int = 6,
    mutate_neg_prob: float = 0.3,
    ema_decay: float = 0.999,
    ema_update_every: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_cuda = torch.cuda.is_available() and device != 'cpu'
    device_t = torch.device(device if use_cuda else 'cpu')
    print("[train] device:", device_t)

    print("[train] loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    print("tokenizer.eos_token:", repr(getattr(tokenizer, "eos_token", None)), "eos_token_id:", getattr(tokenizer, "eos_token_id", None))

    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else (tokenizer.eos_token_id or 0)

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
        device_map_arg = "auto" if multi_gpu else None
        if quant_conf is not None:
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant_conf, device_map=device_map_arg, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map=device_map_arg, trust_remote_code=True)
        try:
            model.config.use_cache = False
        except Exception:
            pass
        if device_map_arg is None and device_t.type == "cuda":
            try:
                model.to(device_t)
            except Exception:
                pass
        return model

    print("[train] loading base model (may be memory heavy)")
    base_model = load_model(model_name, qconf)
    if quant_bits in (4, 8):
        base_model = prepare_model_for_kbit_training(base_model)
    if gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False
        except Exception:
            pass

    print("[train] applying LoRA")
    peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo"],
                             lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    try:
        peft_model = get_peft_model(base_model, peft_config)
    except Exception as e:
        print("[train] get_peft_model failed, retrying with smaller target modules. Error:", e)
        peft_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_model = get_peft_model(base_model, peft_config)

    hidden_size = peft_model.config.hidden_size
    constr_model = ConstraintLM(peft_model, hidden_size=hidden_size, constraint_k=constraint_k)
    try:
        constr_model.head = constr_model.head.to(device_t)
    except Exception:
        pass

    # Build ref_model as a deep copy of base_model and use EMA updating (OPTIM)
    print("[train] creating EMA reference model (deepcopy of base)")
    ref_model = copy.deepcopy(constr_model.model)
    # put ref_model initially on same device or CPU depending on ref_on_cpu
    if ref_on_cpu:
        try:
            ref_model.to("cpu")
        except Exception:
            pass
    else:
        try:
            ref_model.to(device_t)
        except Exception:
            pass
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load data and prebuild step pairs (OPTIM: precompute step strings once)
    print("[train] loading dataset:", data_path)
    raw = load_jsonl(data_path)
    samples = []
    for r in raw:
        pos_c = r.get("constraints_pos", None)
        neg_c = r.get("constraints_neg", None)
        ctx_val = r.get("context", {})
        ctx = ctx_val if isinstance(ctx_val, (dict, str)) else {}
        s = PairSample(context=ctx, pos=r.get("accepted", ""), neg=r.get("rejected", ""),
                       pos_constraints=pos_c, neg_constraints=neg_c)
        formatted_ctx = format_context_with_topic(s.context)
        # Full sequence prompts
        prefix_full = f"Given the context below, evaluate the hypothesis.\n\n{formatted_ctx}\n\nHypothesis:\n"
        s.prefix_full = prefix_full
        s.pos_full = prefix_full + ensure_eos(s.pos, tokenizer)
        s.neg_full = prefix_full + ensure_eos(s.neg, tokenizer)
        # Build step pairs strings (only text-level; will be tokenized later)
        ctx_block = formatted_ctx + "\n\n"
        step_pairs = build_step_pairs_from_sample_string(ctx_block, s.pos, s.neg, max_step_pairs=max_step_pairs, mutate_neg_prob=mutate_neg_prob)
        # Convert to local prompt strings now
        s.step_pairs = []
        for prefix_steps, pos_next, neg_next in step_pairs:
            local_prefix = f"Given the context below, continue the reasoning. Context:\n{formatted_ctx}\nReasoning so far:\n{prefix_steps}\nNext step (single declarative sentence):\n"
            pos_entry = ensure_eos(pos_next, tokenizer)
            neg_entry = ensure_eos(neg_next, tokenizer)
            s.step_pairs.append((local_prefix, pos_entry, neg_entry))
        # calculate rough length for bucketing
        s.example_len = len(tokenizer(s.pos_full, add_special_tokens=False)["input_ids"]) + len(tokenizer(s.neg_full, add_special_tokens=False)["input_ids"])
        samples.append(s)
    random.shuffle(samples)
    n = len(samples)
    print(f"[train] loaded {n} pairs (prebuilt step pairs).")

    # We'll use same samples for SFT and DPO phases; create loaders using collate for sft only
    sft_samples = samples
    dpo_samples = samples

    # Simple collate for SFT (unchanged behaviour)
    def collate_pairs_sft(batch: List[PairSample]):
        pos_texts, neg_texts, prefix_cache, pos_cons, neg_cons = [], [], [], [], []
        for s in batch:
            pos_texts.append(s.pos_full)
            neg_texts.append(s.neg_full)
            prefix_cache.append(s.prefix_full)
            pos_cons.append(s.pos_constraints)
            neg_cons.append(s.neg_constraints)
        enc_pos = tokenizer(pos_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        enc_neg = tokenizer(neg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        enc_pos = _enforce_eos_on_encoding(enc_pos, tokenizer, pad_id)
        enc_neg = _enforce_eos_on_encoding(enc_neg, tokenizer, pad_id)
        prefix_token_lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefix_cache]
        prompt_lens = torch.tensor(prefix_token_lengths, dtype=torch.long)
        return enc_pos, enc_neg, prompt_lens, pos_cons, neg_cons

    sft_loader = DataLoader(PairDataset(sft_samples), batch_size=batch_size, shuffle=True, collate_fn=collate_pairs_sft)

    trainable_params = [p for n,p in constr_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    bce_loss = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device_t.type == "cuda"))

    # -------------------------
    # SFT phase (unchanged)
    # -------------------------
    print("[train] START SFT phase")
    for epoch in range(1, epochs_sft + 1):
        constr_model.model.train()
        constr_model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(sft_loader, desc=f"SFT epoch {epoch}")
        sft_micro_contrib = 0
        running_loss_sum = 0.0
        for enc_pos, enc_neg, prompt_lens, pos_cons_list, neg_cons_list in pbar:
            input_ids_pos = enc_pos["input_ids"].to(device_t)
            attn_pos = enc_pos["attention_mask"].to(device_t)
            input_ids_neg = enc_neg["input_ids"].to(device_t)
            attn_neg = enc_neg["attention_mask"].to(device_t)
            prompt_lens = prompt_lens.to(device_t)
            max_len_batch = max(input_ids_pos.size(1), input_ids_neg.size(1))
            input_ids_pos, attn_pos = pad_to_length(input_ids_pos, attn_pos, max_len_batch, pad_id)
            input_ids_neg, attn_neg = pad_to_length(input_ids_neg, attn_neg, max_len_batch, pad_id)
            input_ids_both = torch.cat([input_ids_pos, input_ids_neg], dim=0)
            attn_both = torch.cat([attn_pos, attn_neg], dim=0)
            labels_both = input_ids_both.clone()
            B_pos = input_ids_pos.size(0)
            B_total = labels_both.size(0)
            prompt_lens_both = torch.cat([prompt_lens, prompt_lens], dim=0)
            for idx in range(B_total):
                plen = int(prompt_lens_both[idx].item())
                if plen > 0:
                    labels_both[idx, :plen] = -100
            if B_total > B_pos:
                labels_both[B_pos:] = -100
            supervised_counts = (labels_both != -100).sum(dim=1)
            if supervised_counts.sum().item() == 0:
                continue
            with autocast(enabled=(device_t.type == "cuda")):
                out_both, last_hidden = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both, attention_mask=attn_both, labels=labels_both)
                lm_loss_pos = out_both.loss
                mask = attn_both.unsqueeze(-1).to(dtype=last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
                pooled_pos = pooled[:input_ids_pos.size(0)].detach()
                pooled_neg = pooled[input_ids_pos.size(0):].detach()
                pred_pos_logits = constr_model.head_forward(pooled_pos)
                pred_neg_logits = constr_model.head_forward(pooled_neg)
                # prepare targets
                bpos = []; bpos_mask = []
                for item in pos_cons_list:
                    if item is None:
                        bpos.append([0.0]*constraint_k); bpos_mask.append(0.0)
                    else:
                        v = item + [0]*max(0, constraint_k - len(item)); bpos.append(v[:constraint_k]); bpos_mask.append(1.0)
                bneg = []; bneg_mask = []
                for item in neg_cons_list:
                    if item is None:
                        bneg.append([0.0]*constraint_k); bneg_mask.append(0.0)
                    else:
                        v = item + [0]*max(0, constraint_k - len(item)); bneg.append(v[:constraint_k]); bneg_mask.append(1.0)
                logits_device = pred_pos_logits.device
                logits_dtype = pred_pos_logits.dtype
                cons_loss = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)
                cnt = 0.0
                if len(bpos) > 0:
                    tgt_pos = torch.tensor(bpos, dtype=logits_dtype, device=logits_device)
                    mask_pos = torch.tensor(bpos_mask, dtype=torch.float32, device=logits_device)
                    if mask_pos.sum() > 0:
                        valid_idxs = (mask_pos == 1.0).nonzero(as_tuple=False).squeeze(-1)
                        cons_loss = cons_loss + bce_loss(pred_pos_logits[valid_idxs], tgt_pos[valid_idxs]); cnt += 1.0
                if len(bneg) > 0:
                    tgt_neg = torch.tensor(bneg, dtype=logits_dtype, device=logits_device)
                    mask_neg = torch.tensor(bneg_mask, dtype=torch.float32, device=logits_device)
                    if mask_neg.sum() > 0:
                        valid_idxs = (mask_neg == 1.0).nonzero(as_tuple=False).squeeze(-1)
                        cons_loss = cons_loss + bce_loss(pred_neg_logits[valid_idxs], tgt_neg[valid_idxs]); cnt += 1.0
                if cnt > 0:
                    cons_loss = cons_loss / cnt
                else:
                    cons_loss = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)
                raw_loss = lm_loss_pos + alpha_sft * cons_loss
            if not torch.isfinite(raw_loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_to_backward = raw_loss / grad_accum_steps
            if device_t.type == "cuda":
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            sft_micro_contrib += 1
            running_loss_sum += float(raw_loss.detach().cpu())
            if (sft_micro_contrib % grad_accum_steps) == 0:
                if device_t.type == "cuda":
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except Exception:
                        try:
                            optimizer.step()
                        except Exception:
                            raise
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            avg_loss = running_loss_sum / max(1, sft_micro_contrib)
            pbar.set_postfix({'sft_loss': avg_loss})
            try:
                del out_both, last_hidden, pooled, pooled_pos, pooled_neg, pred_pos_logits, pred_neg_logits
            except Exception:
                pass
            if device_t.type == "cuda":
                torch.cuda.empty_cache()
        # save checkpoint at end of SFT epoch
        epoch_dir = os.path.join(out_dir, f"sft_epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        try:
            constr_model.model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            torch.save({'head_state': constr_model.head.state_dict()}, os.path.join(epoch_dir, "constraint_head.pt"))
        except Exception:
            torch.save({k:v.cpu() for k,v in constr_model.state_dict().items()}, os.path.join(epoch_dir, "full_state.pt"))
        print(f"[SFT] saved epoch {epoch} to {epoch_dir}")

    if device_t.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # -------------------------
    # DPO phase (OPTIM: sorted pairs, dedupe forwards, EMA ref updates)
    # -------------------------
    print("[train] START DPO phase (optimized)")
    constr_model.model.eval()
    constr_model.eval()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    # Build list of items for DPO; we'll sort for bucketing to reduce padding
    pair_items_base = []
    for s in dpo_samples:
        # we'll keep reference to PairSample object to access prebuilt text
        pair_items_base.append(s)
    # compute hardness score for ordering but then we'll sort by example_len to bucket
    random.shuffle(pair_items_base)

    global_step_dpo = 0
    dpo_micro_contrib = 0
    dpo_opt_steps = 0

    for epoch in range(1, epochs_dpo + 1):
        print(f"[DPO] epoch {epoch} - shuffling and sorting for batching")
        random.shuffle(pair_items_base)
        # sort by example_len (smaller to larger) to reduce padding within batches (OPTIM)
        pair_items_base.sort(key=lambda x: x.example_len)
        # iterate in batch_size chunks
        pbar_outer = tqdm(range(0, len(pair_items_base), batch_size), desc=f"DPO epoch {epoch}")
        for i in pbar_outer:
            batch_samples = pair_items_base[i:i+batch_size]  # batch of PairSample
            # Build lists of texts for full and step items
            pos_full_texts = []; neg_full_texts = []; prefix_lens_full = []; pos_cons = []; neg_cons = []
            step_texts_all = []  # will be [pos0, neg0, pos1, neg1, ...] for all selected step pairs
            step_prompt_lens = []
            step_pair_counts_per_sample = []
            for s in batch_samples:
                pos_full_texts.append(s.pos_full)
                neg_full_texts.append(s.neg_full)
                prefix_lens_full.append(len(tokenizer(s.prefix_full, add_special_tokens=False)["input_ids"]))
                pos_cons.append(s.pos_constraints); neg_cons.append(s.neg_constraints)
                # Subsampling of step pairs already done at load; keep all prebuilt step pairs but ensure limit
                # (they were built with max_step_pairs). We'll use them as-is.
                step_pair_counts_per_sample.append(len(s.step_pairs))
                for (local_prefix, pos_next, neg_next) in s.step_pairs:
                    step_texts_all.append(local_prefix + pos_next)
                    step_texts_all.append(local_prefix + neg_next)
                    step_prompt_lens.append(len(tokenizer(local_prefix, add_special_tokens=False)["input_ids"]))

            # Tokenize full prompts (pos/neg) as a batch
            enc_pos_full = tokenizer(pos_full_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc_neg_full = tokenizer(neg_full_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc_pos_full = _enforce_eos_on_encoding(enc_pos_full, tokenizer, pad_id)
            enc_neg_full = _enforce_eos_on_encoding(enc_neg_full, tokenizer, pad_id)
            input_ids_pos_full = enc_pos_full["input_ids"].to(device_t); attn_pos_full = enc_pos_full["attention_mask"].to(device_t)
            input_ids_neg_full = enc_neg_full["input_ids"].to(device_t); attn_neg_full = enc_neg_full["attention_mask"].to(device_t)
            prompt_lens_full = torch.tensor(prefix_lens_full, dtype=torch.long, device=device_t)
            max_len_full = max(input_ids_pos_full.size(1), input_ids_neg_full.size(1))
            input_ids_pos_full, attn_pos_full = pad_to_length(input_ids_pos_full, attn_pos_full, max_len_full, pad_id)
            input_ids_neg_full, attn_neg_full = pad_to_length(input_ids_neg_full, attn_neg_full, max_len_full, pad_id)
            input_ids_both_full = torch.cat([input_ids_pos_full, input_ids_neg_full], dim=0)
            attn_both_full = torch.cat([attn_pos_full, attn_neg_full], dim=0)
            prompt_lens_both_full = torch.cat([prompt_lens_full, prompt_lens_full], dim=0)

            # STEP texts: dedupe and forward unique only (OPTIM). We'll get logits for each step text in order.
            if len(step_texts_all) > 0:
                # dedupe and forward unique via forward_unique_texts (which runs model in no-grad)
                # For theta (we need gradients) we still need to run model under train mode. But we can call model once on the *batch* of unique step texts with grad (since they may be numerous).
                # Strategy: dedupe step_texts_all -> unique list U
                uniq_keys = {}
                uniq_list = []
                order_map = []
                for t in step_texts_all:
                    if t not in uniq_keys:
                        uniq_keys[t] = len(uniq_list)
                        uniq_list.append(t)
                    order_map.append(uniq_keys[t])
                # Tokenize uniq_list
                enc_uniq = tokenizer(uniq_list, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
                enc_uniq = _enforce_eos_on_encoding(enc_uniq, tokenizer, pad_id)
                input_ids_uniq = enc_uniq["input_ids"].to(device_t); attn_uniq = enc_uniq["attention_mask"].to(device_t)
                # Forward once (theta) with grad for uniq_list
                constr_model.model.train()
                with autocast(enabled=(device_t.type == "cuda")):
                    out_uniq_theta, _ = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_uniq, attention_mask=attn_uniq)
                    logits_uniq_theta = out_uniq_theta.logits
                # Map back to step_texts_all order
                logits_step_theta_list = [logits_uniq_theta[order_map[idx]].unsqueeze(0) for idx in range(len(order_map))]
                # Now compute logprobs for each via compute_sequence_logprobs_from_logits in a batched way
                # Build batch tensors for step texts by stacking logits corresponding to each text
                logits_step_theta = torch.cat(logits_step_theta_list, dim=0)  # shape (N_step_texts, L, V)
                # Tokenize step_texts_all in the same order to get input_ids and attention for computing logprobs
                enc_step_all = tokenizer(step_texts_all, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
                enc_step_all = _enforce_eos_on_encoding(enc_step_all, tokenizer, pad_id)
                input_ids_step_all = enc_step_all["input_ids"].to(device_t); attn_step_all = enc_step_all["attention_mask"].to(device_t)
                prompt_lens_step_all = torch.tensor(step_prompt_lens, dtype=torch.long, device=device_t)
                # compute seq logprobs for theta for each step text
                logp_step_theta_all = compute_sequence_logprobs_from_logits(logits_step_theta, input_ids_step_all, attn_step_all, prompt_lens_step_all)
                # split into pos/neg halves: recall we appended pos,neg pairs alternately
                B_step_total = logp_step_theta_all.size(0)
                # pairwise: for each pair i -> pos idx 2i, neg idx 2i+1
                logp_pos_theta_step = logp_step_theta_all[0:B_step_total:2]
                logp_neg_theta_step = logp_step_theta_all[1:B_step_total:2]
            else:
                logp_pos_theta_step = None
                logp_neg_theta_step = None

            # Sequence-level forward for theta (pos & neg together) - this we need grad for
            constr_model.model.train()
            with autocast(enabled=(device_t.type == "cuda")):
                out_both_theta_full, last_hidden_theta_full = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both_full, attention_mask=attn_both_full)
                logits_both_theta_full = out_both_theta_full.logits
                logp_both_theta_full = compute_sequence_logprobs_from_logits(logits_both_theta_full, input_ids_both_full, attn_both_full, prompt_lens_both_full)
                B_full = input_ids_pos_full.size(0)
                logp_pos_theta_full = logp_both_theta_full[:B_full]
                logp_neg_theta_full = logp_both_theta_full[B_full:]

            # Constraint head proxies (no grad)
            with torch.no_grad():
                pos_pred_logits = constr_model.predict_constraints_no_grad(input_ids_pos_full, attention_mask=attn_pos_full, run_with_hook_fn=run_with_last_hidden_hook)
                neg_pred_logits = constr_model.predict_constraints_no_grad(input_ids_neg_full, attention_mask=attn_neg_full, run_with_hook_fn=run_with_last_hidden_hook)
                vpos_pred = torch.sigmoid(pos_pred_logits).sum(dim=1)
                vneg_pred = torch.sigmoid(neg_pred_logits).sum(dim=1)
            def build_violation_tensor_for_list(cons_list, pred_tensor):
                out_list = []
                for idx, entry in enumerate(cons_list):
                    if entry is None:
                        out_list.append(float(pred_tensor[idx].item()))
                    else:
                        s = float(sum(entry[:constraint_k]))
                        out_list.append(s)
                return torch.tensor(out_list, dtype=torch.float32, device=device_t)
            vpos = build_violation_tensor_for_list(pos_cons, vpos_pred)
            vneg = build_violation_tensor_for_list(neg_cons, vneg_pred)

            # reference logits: dedupe across union of uniq_list + full sequences
            # Build reference text list to dedupe: all uniq_list strings + pos_full_texts + neg_full_texts
            # We will call forward_unique_texts on the union and re-map to required sequences.
            # Build union_texts (theta uniq_list strings might already be present)
            union_texts = []
            # include full pos/neg first (we need mapping)
            union_texts.extend(pos_full_texts)
            union_texts.extend(neg_full_texts)
            # include step unique strings
            if len(step_texts_all) > 0:
                # uniq_list is available via earlier variable if created
                if 'uniq_list' in locals():
                    union_texts.extend(uniq_list)
                else:
                    # fallback: add step_texts_all (dedupe inside forward_unique_texts)
                    union_texts.extend(step_texts_all)
            # dedupe and forward using forward_unique_texts (no-grad)
            logits_union = forward_unique_texts(ref_model, tokenizer, union_texts, device_t, max_len=max_len)
            # map back: first |pos_full_texts| entries are pos full ref logits etc
            n_pos = len(pos_full_texts)
            n_neg = len(neg_full_texts)
            logits_pos_ref = logits_union[:n_pos]
            logits_neg_ref = logits_union[n_pos:n_pos+n_neg]
            # step ref logits: if uniq_list exists, we need to map uniq_list entries positions in union_texts
            if len(step_texts_all) > 0:
                # find index of first step-uniq in union_texts
                start_idx = n_pos + n_neg
                # build idx map for step_texts_all order -> union index
                # but we used uniq_list earlier; union_texts included uniq_list in same order as uniq_list
                # therefore mapping is straightforward: step_texts_all order -> position = start_idx + order_map[idx]
                logp_step_ref_all = []
                # get logits for union entries that correspond to step_texts_all via order_map
                enc_step_all_for_ref = tokenizer(step_texts_all, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
                enc_step_all_for_ref = _enforce_eos_on_encoding(enc_step_all_for_ref, tokenizer, pad_id)
                # compute logprobs for step ref logits similarly to theta
                # Build tensor of logits for step_texts_all by mapping order_map -> union logits
                logits_union_tensor_list = []
                # Build dict from uniq_list string to its ref logits tensor
                uniq_to_ref_logit = {}
                if 'uniq_list' in locals():
                    for idx_u, ustr in enumerate(uniq_list):
                        uniq_to_ref_logit[ustr] = logits_union[start_idx + idx_u]
                    # now build list in order of step_texts_all
                    for t in step_texts_all:
                        logits_union_tensor_list.append(uniq_to_ref_logit[t].unsqueeze(0))
                    logits_step_ref = torch.cat(logits_union_tensor_list, dim=0)
                    # compute logprobs for these
                    input_ids_step_all_ref = enc_step_all_for_ref["input_ids"].to(device_t); attn_step_all_ref = enc_step_all_for_ref["attention_mask"].to(device_t)
                    prompt_lens_step_all_ref = torch.tensor(step_prompt_lens, dtype=torch.long, device=device_t)
                    logp_step_ref_all = compute_sequence_logprobs_from_logits(logits_step_ref, input_ids_step_all_ref, attn_step_all_ref, prompt_lens_step_all_ref)
                    logp_pos_ref_step = logp_step_ref_all[0:logp_step_ref_all.size(0):2]
                    logp_neg_ref_step = logp_step_ref_all[1:logp_step_ref_all.size(0):2]
                else:
                    # fallback: forward ref_model on step_texts_all directly (should be rare)
                    enc_step_ref = enc_step_all_for_ref
                    input_ids_step_ref = enc_step_ref["input_ids"].to(device_t); attn_step_ref = enc_step_ref["attention_mask"].to(device_t)
                    out_step_ref, _ = run_with_last_hidden_hook(ref_model, input_ids=input_ids_step_ref, attention_mask=attn_step_ref)
                    logits_step_ref = out_step_ref.logits
                    logp_step_ref_all = compute_sequence_logprobs_from_logits(logits_step_ref, input_ids_step_ref, attn_step_ref, torch.tensor(step_prompt_lens, dtype=torch.long, device=device_t))
                    logp_pos_ref_step = logp_step_ref_all[0:logp_step_ref_all.size(0):2]
                    logp_neg_ref_step = logp_step_ref_all[1:logp_step_ref_all.size(0):2]
            else:
                logp_pos_ref_step = None
                logp_neg_ref_step = None

            # compute sequence-level ref logprobs from logits_pos_ref/logits_neg_ref tensors
            # need to compute seq logprobs for full pos/neg: convert logits_pos_ref[ idx ] to tensor then compute
            # Note: forward_unique_texts returned logits (on CPU and moved to device already)
            # Tokenize pos_full_texts & neg_full_texts again to get their input_ids/attn for ref logprobs (we already had enc_pos_full/enc_neg_full)
            # compute logprobs by using logits_pos_ref tensors aligned with enc_pos_full tokenization
            # First build stacked logits tensor for pos+neg in same order
            logits_pos_ref_stack = torch.stack([logits_pos_ref[j] for j in range(len(logits_pos_ref))], dim=0).to(device_t)
            logits_neg_ref_stack = torch.stack([logits_neg_ref[j] for j in range(len(logits_neg_ref))], dim=0).to(device_t)
            logits_both_ref_full = torch.cat([logits_pos_ref_stack, logits_neg_ref_stack], dim=0)
            logp_both_ref_full = compute_sequence_logprobs_from_logits(logits_both_ref_full, input_ids_both_full, attn_both_full, prompt_lens_both_full)
            logp_pos_ref_full = logp_both_ref_full[:B_full]
            logp_neg_ref_full = logp_both_ref_full[B_full:]

            # Deltas and diffs
            delta_logp_full = logp_pos_theta_full - logp_neg_theta_full
            d_theta_adj_full = delta_logp_full - lambda_v * (vpos - vneg)
            d_ref_full = logp_pos_ref_full - logp_neg_ref_full
            diff_full = d_theta_adj_full - d_ref_full

            if logp_pos_theta_step is not None:
                delta_logp_step = logp_pos_theta_step - logp_neg_theta_step
                d_theta_adj_step = delta_logp_step
                d_ref_step = logp_pos_ref_step - logp_neg_ref_step
                diff_step = d_theta_adj_step - d_ref_step
            else:
                diff_step = None

            # beta schedule
            global_step_dpo += 1
            if global_step_dpo < beta_warmup_steps:
                beta = beta_start + (beta_target - beta_start) * (global_step_dpo / max(1, beta_warmup_steps))
            else:
                beta = beta_target

            scaled_full = beta * diff_full
            scaled_full = torch.clamp(scaled_full, min=-50.0, max=50.0)
            per_example_loss_full = -F.logsigmoid(scaled_full)
            loss_full = per_example_loss_full.mean()
            if diff_step is not None:
                scaled_step = beta * diff_step
                scaled_step = torch.clamp(scaled_step, min=-50.0, max=50.0)
                per_example_loss_step = -F.logsigmoid(scaled_step)
                loss_step = per_example_loss_step.mean()
            else:
                loss_step = torch.tensor(0.0, device=device_t)

            # auxiliary constraint BCE
            with torch.no_grad():
                _, last_hidden_pooled = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both_full, attention_mask=attn_both_full)
                mask = attn_both_full.unsqueeze(-1).to(dtype=last_hidden_pooled.dtype)
                pooled_all = (last_hidden_pooled * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
                pooled_pos = pooled_all[:B_full].detach()
                pooled_neg = pooled_all[B_full:].detach()
            out_pos_logits = constr_model.head_forward(pooled_pos)
            out_neg_logits = constr_model.head_forward(pooled_neg)
            pos_targets = []; pos_mask = []
            neg_targets = []; neg_mask = []
            for item in pos_cons:
                if item is None:
                    pos_targets.append([0.0]*constraint_k); pos_mask.append(0.0)
                else:
                    v = item + [0]*max(0, constraint_k - len(item)); pos_targets.append(v[:constraint_k]); pos_mask.append(1.0)
            for item in neg_cons:
                if item is None:
                    neg_targets.append([0.0]*constraint_k); neg_mask.append(0.0)
                else:
                    v = item + [0]*max(0, constraint_k - len(item)); neg_targets.append(v[:constraint_k]); neg_mask.append(1.0)
            logits_dtype = out_pos_logits.dtype
            logits_device = out_pos_logits.device
            cons_loss_dpo = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)
            cnt = 0.0
            if sum(pos_mask) > 0:
                tgt_pos = torch.tensor(pos_targets, dtype=logits_dtype, device=logits_device)
                valid_idxs = (torch.tensor(pos_mask, device=logits_device) == 1.0).nonzero(as_tuple=False).squeeze(-1)
                cons_loss_dpo = cons_loss_dpo + bce_loss(out_pos_logits[valid_idxs], tgt_pos[valid_idxs]); cnt += 1.0
            if sum(neg_mask) > 0:
                tgt_neg = torch.tensor(neg_targets, dtype=logits_dtype, device=logits_device)
                valid_idxs = (torch.tensor(neg_mask, device=logits_device) == 1.0).nonzero(as_tuple=False).squeeze(-1)
                cons_loss_dpo = cons_loss_dpo + bce_loss(out_neg_logits[valid_idxs], tgt_neg[valid_idxs]); cnt += 1.0
            if cnt > 0:
                cons_loss_dpo = cons_loss_dpo / cnt
            else:
                cons_loss_dpo = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)

            # Combine
            raw_loss = dpo_strength * (lambda_step * loss_step + lambda_seq * loss_full) + 0.5 * alpha_dpo * cons_loss_dpo

            # backward & optimizer step (grad accum)
            if not torch.isfinite(raw_loss):
                pbar_outer.set_postfix({'dpo_raw_loss': float('nan')})
                continue
            loss_to_backward = raw_loss / grad_accum_steps
            if device_t.type == "cuda":
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            dpo_micro_contrib += 1
            if (dpo_micro_contrib % grad_accum_steps) == 0:
                if device_t.type == "cuda":
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except Exception:
                        try:
                            optimizer.step()
                        except Exception:
                            pass
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    try:
                        optimizer.step()
                    except Exception:
                        pass
                optimizer.zero_grad(set_to_none=True)
                dpo_opt_steps += 1
                # EMA update of ref_model parameters occasionally (OPTIM)
                if (dpo_opt_steps % ema_update_every) == 0:
                    try:
                        ema_update(ref_model, constr_model.model, ema_decay)
                    except Exception:
                        pass

            pbar_outer.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta), 'step_loss': float(loss_step.detach().cpu()) if isinstance(loss_step, torch.Tensor) else 0.0, 'seq_loss': float(loss_full.detach().cpu())})

            # cleanup
            try:
                del enc_uniq, input_ids_uniq, attn_uniq, out_uniq_theta, logits_uniq_theta
            except Exception:
                pass
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        # epoch save
        epoch_dir = os.path.join(out_dir, f"dpo_epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        try:
            constr_model.model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            torch.save({'head_state': constr_model.head.state_dict()}, os.path.join(epoch_dir, "constraint_head.pt"))
        except Exception:
            torch.save({k:v.cpu() for k,v in constr_model.state_dict().items()}, os.path.join(epoch_dir, "full_state.pt"))
        print(f"[DPO] saved epoch {epoch} to {epoch_dir}")

    print("[train] training complete. adapters + head saved to", out_dir)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default="dataset.jsonl", help='path to dataset.jsonl')
    p.add_argument('--out_dir', type=str, default='out_checkpoints')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--epochs_sft', type=int, default=1)
    p.add_argument('--epochs_dpo', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--constraint_k', type=int, default=12)
    p.add_argument('--alpha_sft', type=float, default=0.2)
    p.add_argument('--alpha_dpo', type=float, default=0.5)
    p.add_argument('--beta_start', type=float, default=0.1)
    p.add_argument('--beta_target', type=float, default=0.15)
    p.add_argument('--beta_warmup_steps', type=int, default=2000)
    p.add_argument('--lambda_v', type=float, default=0.7)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--quant_bits', type=int, default=4, choices=[4,8,16])
    p.add_argument('--grad_ckpt', action='store_true', default=True)
    p.add_argument('--ref_on_cpu', action='store_true', default=False)
    p.add_argument('--sft_frac', type=float, default=0.5)
    p.add_argument('--near_miss_frac', type=float, default=0.30)
    p.add_argument('--dpo_strength', type=float, default=0.5)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--lambda_step', type=float, default=2.0)
    p.add_argument('--lambda_seq', type=float, default=0.5)
    p.add_argument('--max_step_pairs', type=int, default=6)
    p.add_argument('--mutate_neg_prob', type=float, default=0.3)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--ema_update_every', type=int, default=1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        out_dir=args.out_dir,
        model_name=args.model,
        epochs_sft=args.epochs_sft,
        epochs_dpo=args.epochs_dpo,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        constraint_k=args.constraint_k,
        alpha_sft=args.alpha_sft,
        alpha_dpo=args.alpha_dpo,
        beta_start=args.beta_start,
        beta_target=args.beta_target,
        beta_warmup_steps=args.beta_warmup_steps,
        lambda_v=args.lambda_v,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_len=1024,
        quant_bits=args.quant_bits,
        gradient_checkpointing=args.grad_ckpt,
        ref_on_cpu=args.ref_on_cpu,
        sft_frac=args.sft_frac,
        near_miss_frac=args.near_miss_frac,
        dpo_strength=args.dpo_strength,
        grad_accum_steps=args.grad_accum_steps,
        lambda_step=args.lambda_step,
        lambda_seq=args.lambda_seq,
        max_step_pairs=args.max_step_pairs,
        mutate_neg_prob=args.mutate_neg_prob,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
    )
