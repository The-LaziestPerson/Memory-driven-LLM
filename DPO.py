#!/usr/bin/env python3
"""
trainer_sft_constraint_dpo_qloRA_curriculum_topic_fixed.py

Trainer with:
 - SFT + curriculum DPO
 - qLoRA / LoRA
 - constraint head
 - robust gradient accumulation
 - EOS enforcement fixes
 - guards against NaN when using small batch sizes + accumulation

Usage:
    python trainer_sft_constraint_dpo_qloRA_curriculum_topic_fixed.py \
        --data dataset.jsonl --batch_size 1 --grad_accum_steps 4
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

# pad helper to align pos/neg before concatenation
def pad_to_length(input_ids: torch.Tensor, attention_mask: torch.Tensor, target_len: int, pad_token_id: int):
    cur_len = input_ids.size(1)
    if cur_len == target_len:
        return input_ids, attention_mask
    pad_len = target_len - cur_len
    input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
    return input_ids, attention_mask

# -------------------------
# Dataset classes
# -------------------------
class PairSample:
    def __init__(self, context: Any, pos: str, neg: str, pos_constraints=None, neg_constraints=None):
        # context expected to be dict {"topic": str, "ancestors": [str,...]} or string fallback
        self.context = context
        self.pos = pos
        self.neg = neg
        self.pos_constraints = pos_constraints
        self.neg_constraints = neg_constraints

class PairDataset(Dataset):
    def __init__(self, samples: List[PairSample]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------
# Prompt/context helpers
# -------------------------
def ensure_eos(text: str, tokenizer):
    """
    Ensure the example will *result* in an eos token id after tokenization.
    Strategy:
      1) If tokenizer has eos_token_id, tokenizes the string (add_special_tokens=False)
         and checks if last id == eos_token_id. If yes -> return text unchanged.
      2) Otherwise append tokenizer.eos_token (or tokenizer.sep_token) string if available.
      3) Fallback to original string unchanged.
    """
    if text is None:
        return ""
    text = text.rstrip()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_str = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "sep_token", None)

    # If tokenizer exposes eos_token_id, check tokenized ids directly (robust)
    try:
        if eos_id is not None:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > 0 and ids[-1] == eos_id:
                return text
            # if last token is not eos, append textual eos if available
            if eos_str:
                return text + eos_str
            return text
    except Exception:
        # fallback to previous behavior if tokenization check fails
        pass

    # Fallback: string-level check / append
    if eos_str is not None:
        if not text.endswith(eos_str):
            return text + eos_str
        return text

    return text

def format_context_with_topic(context_obj: Any) -> str:
    """
    Format context that may be dict {"topic": str, "ancestors": [str,...]} or string.
    """
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

# -------------------------
# Token-id enforcement helper (CRITICAL) — SAFER
# -------------------------
def _enforce_eos_on_encoding(enc: Dict[str, torch.Tensor], tokenizer, pad_id: int):
    """
    Mutates enc (the output of tokenizer(..., return_tensors='pt')) so that each example
    contains tokenizer.eos_token_id as the last non-pad token. Safer than overwriting the
    only non-pad token: if the row has exactly 1 non-pad token, prefer to write eos into
    the next pad slot if available; otherwise leave the token as-is (we won't destroy supervision).
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return enc
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    B, L = ids.shape
    for b in range(B):
        # index of last non-pad token
        row_attn = attn[b]
        nonpad_idxs = (row_attn == 1).nonzero(as_tuple=True)[0]
        if len(nonpad_idxs) == 0:
            # empty row: set first token to eos if possible
            if L > 0:
                ids[b, 0] = eos_id
                attn[b, 0] = 1
            continue
        last_idx = int(nonpad_idxs[-1].item())
        last_token = int(ids[b, last_idx].item())
        if last_token == eos_id:
            continue
        # prefer to write eos into the next pad slot if available
        if last_idx + 1 < L and attn[b, last_idx + 1].item() == 0:
            ids[b, last_idx + 1] = eos_id
            attn[b, last_idx + 1] = 1
        else:
            # safer overwrite: only overwrite if there is at least one other non-pad token
            if len(nonpad_idxs) > 1:
                ids[b, last_idx] = eos_id
            else:
                # single-token non-pad row and no pad slot available:
                # don't overwrite the only token (preserve supervision)
                # (we accept that eos_token_id may not be present in this row)
                pass
    enc["input_ids"] = ids
    enc["attention_mask"] = attn
    return enc

# -------------------------
# Collation and prompt builder
# -------------------------
def build_prompt(context: str, hypothesis: str, prompt_prefix: str = None):
    if prompt_prefix is None:
        prompt_prefix = "Given the context below, evaluate the hypothesis.\n\nContext:\n"
    return f"{prompt_prefix}{context}\n\nHypothesis:\n{hypothesis}"

def collate_pairs(batch: List[PairSample], tokenizer, max_len=1024, pad_id: int = None):
    pos_texts = []
    neg_texts = []
    prefix_cache = []
    pos_constraints = []
    neg_constraints = []
    for s in batch:
        # Format the context: topic + ancestors
        formatted_ctx = format_context_with_topic(s.context)

        ctx_block = f"""{formatted_ctx}

Task:
Produce exactly ONE new hypothesis that extends the prior set under these constraints:

- The hypothesis MUST explicitly reference at least ONE distinct entities, variables,
or mechanisms introduced in the prior hypotheses,  their roles must be causally
necessary to the claim , and it must be explained before the new hypothesis.
- The hypothesis MUST introduce new variable or constraint/relation, and its removal. (can be in form of mathematical expression).
- You must show how is this hypothesis come/derived from, using logical reasoning (must explain throughly).
- The hypothesis MUST be derrived and explained how is this new idea come from based of prior hypothesis. (must refers to the prior hypothesis clearly) 
- The hypothesis MUST be logically implied by, or a minimal extension of, the prior
hypotheses; it must not be compatible with their negation.
- The hypothesis MUST make a determinate claim (no modal verbs, no hedging, no qualifiers
like "often", "may", or "tends to").
- Do NOT introduce internal agents, modules, engines, or named subsystems unless they
correspond to variables already present.
- DO NOT include a assumption that is not related to the Prior Hypothesis
- The hypothesis MUST only include 1 step only. Either of
    1. Define a variable
    2. Define a contraint from the variable
    3. Define a relationship between variables
    4. Futher explain a variable/contraint/relation
    5. Fix a explaination of prior hypothesis weather to be variable/contraint/relation.
- The hypothesis MUST ME TRUE AND CORRECT AT ALL COST (Very crucial)
- The hypothesis MUST NOT CONTAIN FALSE INFORMATION THAT HASNT SCIENTIFICALLY PROVEN YET (if the topic is scientific).
- IF THE HYPOTHESIS IS COMPLEX, the hypothesis must have a verification check (for example, units violation? dimention?)
- The hypothesis MUST end with a conclusion (Very crucial)
- The hypothesis MUST stay on topic.
Summary : List a prior hypotheises that will going to be use first, explain how they might relate each other and form a new hypothesis (must be detailed as if you are showing this to someone). Finish off with a conclusion and capabilities of this new hypothesis/variable/contraints.
    (IF there is a mathematical prove, the hypothesis must explain how is this expression come from (show mathmematical work))
    (Right after the explaination of derrivation of new hypothesis, there must be a further explaination about the new idea's constraint and potential relation)
Output only the hypothesis as a single declarative sentence.
"""
        prefix_cache.append(ctx_block)
        # ensure EOS on pos/neg using tokenizer defaults (no token additions)
        pos_text = ensure_eos(s.pos, tokenizer)
        neg_text = ensure_eos(s.neg, tokenizer)
        pos_texts.append(ctx_block + pos_text)
        neg_texts.append(ctx_block + neg_text)
        pos_constraints.append(s.pos_constraints if s.pos_constraints is not None else None)
        neg_constraints.append(s.neg_constraints if s.neg_constraints is not None else None)

    enc_pos = tokenizer(pos_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    enc_neg = tokenizer(neg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    # enforce eos at token-id level (critical)
    enc_pos = _enforce_eos_on_encoding(enc_pos, tokenizer, pad_id)
    enc_neg = _enforce_eos_on_encoding(enc_neg, tokenizer, pad_id)

    prefix_token_lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefix_cache]
    prompt_lens = torch.tensor(prefix_token_lengths, dtype=torch.long)

    # If pad_id wasn't provided, extract from tokenizer safely (but we prefer caller provided)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else None

    return enc_pos, enc_neg, prompt_lens, pos_constraints, neg_constraints

# -------------------------
# qLoRA util
# -------------------------
def make_bnb_config(bits: int):
    if bits == 16 or bits is None:
        return None
    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    raise ValueError("quant bits must be 4, 8, or 16")

# -------------------------
# ConstraintLM wrapper
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
        logits = self.head(pooled_features)
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
        logits = self.head_forward(pooled)
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

def compute_sequence_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    B, L, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    device = logits.device
    idx = torch.arange(1, L, device=device).unsqueeze(0)
    prompt_lens = torch.as_tensor(prompt_lens, device=device).unsqueeze(1)
    include_mask = (idx >= prompt_lens).to(dtype=shift_logits.dtype)
    shift_attn = attention_mask[:, 1:].contiguous().to(dtype=shift_logits.dtype)
    final_mask = include_mask * shift_attn
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * final_mask
    seq_logprob = token_logprobs.sum(dim=1)
    return seq_logprob

# -------------------------
# Utilities for curriculum (hardness etc.)
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
    # item = (ctx, pos, neg, posc, negc)
    _, pos, neg, posc, negc = item
    sim = symmetric_similarity(pos, neg)
    cons_diff = 0.0
    try:
        pos_sum = sum(posc) if posc else 0.0
        neg_sum = sum(negc) if negc else 0.0
        cons_diff = abs(pos_sum - neg_sum)
    except Exception:
        cons_diff = 0.0
    # near-miss if high sim and small cons_diff
    return 0.7 * sim + 0.3 * (1.0 - min(1.0, cons_diff))

# -------------------------
# Trainer
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
):
    os.makedirs(out_dir, exist_ok=True)
    if grad_accum_steps is None or int(grad_accum_steps) < 1:
        grad_accum_steps = 1
    grad_accum_steps = int(grad_accum_steps)

    use_cuda = torch.cuda.is_available() and device != 'cpu'
    device_t = torch.device(device if use_cuda else 'cpu')
    print("[train] device:", device_t)

    print("[train] loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    # debug print for eos tokens
    print("tokenizer.eos_token:", repr(getattr(tokenizer, "eos_token", None)), "eos_token_id:", getattr(tokenizer, "eos_token_id", None))

    print(f"BEFORE TOKENIZER SIZE : {len(tokenizer)}")

    # Determine pad token id to use for padding operations (do NOT add tokens).
    pad_id = None
    if getattr(tokenizer, "pad_token_id", None) is not None:
        pad_id = tokenizer.pad_token_id
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        pad_id = tokenizer.eos_token_id
    elif getattr(tokenizer, "sep_token_id", None) is not None:
        pad_id = tokenizer.sep_token_id
    elif getattr(tokenizer, "unk_token_id", None) is not None:
        pad_id = tokenizer.unk_token_id
    else:
        pad_id = 0

    qconf = make_bnb_config(quant_bits)

    def load_model(name, quant_conf=None):
        multi_gpu = torch.cuda.device_count() > 1
        device_map_arg = "auto" if multi_gpu else None
        if quant_conf is not None:
            print(f"[train] Loading model {name} with quantization {quant_bits}-bit. device_map={device_map_arg}")
            model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=quant_conf,
                device_map=device_map_arg,
                trust_remote_code=True
            )
        else:
            print(f"[train] Loading model {name} in fp16. device_map={device_map_arg}")
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16,
                device_map=device_map_arg,
                trust_remote_code=True
            )
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

    print("[train] loading ref and base models (may be memory heavy)")
    ref_model = load_model(model_name, qconf)
    if ref_on_cpu:
        try: ref_model.to("cpu")
        except Exception: pass
    else:
        try: ref_model.to(device_t)
        except Exception: pass
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    base_model = load_model(model_name, qconf)
    if quant_bits in (4, 8):
        print("[train] prepare_model_for_kbit_training")
        base_model = prepare_model_for_kbit_training(base_model)

    if gradient_checkpointing:
        try:
            print("[train] enabling gradient checkpointing and disabling use_cache")
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False
        except Exception:
            pass

    print("[train] applying LoRA")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
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

    print("[train] loading dataset:", data_path)
    raw = load_jsonl(data_path)
    samples = []
    for r in raw:
        pos_c = r.get("constraints_pos", None)
        neg_c = r.get("constraints_neg", None)
        # Keep context as dict (if present); fallback to empty dict
        ctx_val = r.get("context", {})
        # allow both dict or string
        ctx = ctx_val if isinstance(ctx_val, (dict, str)) else {}
        samples.append(PairSample(context=ctx, pos=r.get("accepted", ""), neg=r.get("rejected", ""),
                                  pos_constraints=pos_c, neg_constraints=neg_c))
    random.shuffle(samples)
    n = len(samples)
    print(f"[train] loaded {n} pairs")

    # split SFT / DPO (we use same full set for both by default)
    sft_samples = samples
    dpo_samples = samples
    print(f"[train] SFT pairs: {len(sft_samples)} | DPO pairs: {len(dpo_samples)}")

    n_val = max(1, int(0.05 * n))
    val_samples = samples[:n_val]

    # pass pad_id into collate via lambda so pad behavior uses existing token ids only
    sft_loader = DataLoader(PairDataset(sft_samples), batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_pairs(b, tokenizer, max_len=max_len, pad_id=pad_id))
    val_loader = DataLoader(PairDataset(val_samples), batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_pairs(b, tokenizer, max_len=max_len, pad_id=pad_id))

    trainable_params = [p for n,p in constr_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    bce_loss = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device_t.type == "cuda"))

    # -------------------------
    # SFT phase (weakened constraints)
    # -------------------------
    print("[train] START SFT phase (format/fluency focus)")
    print(f"AFTER TOKENIZER SIZE : {len(tokenizer)}")

    # counters and bookkeeping for gradient accumulation
    sft_loop_iter = 0       # raw loop iterations (useful for logging)
    sft_micro_contrib = 0   # micro-batches that actually contributed grads
    sft_opt_steps = 0       # number of optimizer (actual) steps taken
    running_loss_sum = 0.0
    skipped_sft_batches = 0

    for epoch in range(1, epochs_sft + 1):
        constr_model.model.train()
        constr_model.train()
        # ensure grads are zeroed at epoch start
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(sft_loader, desc=f"SFT epoch {epoch}")
        for enc_pos, enc_neg, prompt_lens, pos_cons_list, neg_cons_list in pbar:
            sft_loop_iter += 1

            # enc_pos/enc_neg are already tokenized and eos enforced
            input_ids_pos = enc_pos["input_ids"].to(device_t)
            attn_pos = enc_pos["attention_mask"].to(device_t)
            input_ids_neg = enc_neg["input_ids"].to(device_t)
            attn_neg = enc_neg["attention_mask"].to(device_t)

            # move prompt_lens to device and ensure dtype long
            if isinstance(prompt_lens, torch.Tensor):
                prompt_lens = prompt_lens.to(device_t)
            else:
                prompt_lens = torch.tensor(prompt_lens, dtype=torch.long, device=device_t)

            # pad pos and neg to same length before concatenation
            max_len_batch = max(input_ids_pos.size(1), input_ids_neg.size(1))
            input_ids_pos, attn_pos = pad_to_length(input_ids_pos, attn_pos, max_len_batch, pad_id)
            input_ids_neg, attn_neg = pad_to_length(input_ids_neg, attn_neg, max_len_batch, pad_id)

            # concat B->2B
            input_ids_both = torch.cat([input_ids_pos, input_ids_neg], dim=0)
            attn_both = torch.cat([attn_pos, attn_neg], dim=0)

            # build labels and mask prompt tokens
            labels_both = input_ids_both.clone()
            B_pos = input_ids_pos.size(0)
            B_total = labels_both.size(0)

            # build prompt_lens_both (first half pos, second half neg)
            prompt_lens_both = torch.cat([prompt_lens, prompt_lens], dim=0)
            if prompt_lens_both.size(0) != B_total:
                prompt_lens_both = prompt_lens_both.repeat(int(B_total / prompt_lens_both.size(0)))

            for idx in range(B_total):
                plen = int(prompt_lens_both[idx].item())
                if plen > 0:
                    labels_both[idx, :plen] = -100

            # mask negatives entirely for SFT (as before)
            if B_total > B_pos:
                labels_both[B_pos:] = -100

            # Count supervised tokens per sample — if none, skip this micro-batch
            supervised_counts = (labels_both != -100).sum(dim=1)
            if supervised_counts.sum().item() == 0:
                skipped_sft_batches += 1
                # don't increment sft_micro_contrib, do not backward, just log and continue
                pbar.set_postfix({'sft_loss': (running_loss_sum / max(1, sft_micro_contrib)) if sft_micro_contrib>0 else float('nan'),
                                  'skipped': skipped_sft_batches, 'accum_steps': grad_accum_steps, 'opt_steps': sft_opt_steps})
                continue

            # forward pass (with AMP if enabled)
            with autocast(enabled=(device_t.type == "cuda")):
                out_both, last_hidden = run_with_last_hidden_hook(constr_model.model,
                                                                  input_ids=input_ids_both,
                                                                  attention_mask=attn_both,
                                                                  labels=labels_both)
                lm_loss_pos = out_both.loss

                mask = attn_both.unsqueeze(-1).to(dtype=last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
                pooled_pos = pooled[:input_ids_pos.size(0)].detach()
                pooled_neg = pooled[input_ids_pos.size(0):].detach()

                pred_pos_logits = constr_model.head_forward(pooled_pos)
                pred_neg_logits = constr_model.head_forward(pooled_neg)

                logits_device = pred_pos_logits.device
                logits_dtype = pred_pos_logits.dtype

                # build BCE targets (as before)
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

                if len(bpos) > 0:
                    tgt_pos = torch.tensor(bpos, dtype=logits_dtype, device=logits_device)
                    mask_pos = torch.tensor(bpos_mask, dtype=torch.float32, device=logits_device)
                else:
                    tgt_pos = None; mask_pos = None
                if len(bneg) > 0:
                    tgt_neg = torch.tensor(bneg, dtype=logits_dtype, device=logits_device)
                    mask_neg = torch.tensor(bneg_mask, dtype=torch.float32, device=logits_device)
                else:
                    tgt_neg = None; mask_neg = None

                cons_loss = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)
                cnt = 0.0
                if tgt_pos is not None and mask_pos.sum() > 0:
                    valid_idxs = (mask_pos == 1.0).nonzero(as_tuple=False).squeeze(-1)
                    l = bce_loss(pred_pos_logits[valid_idxs], tgt_pos[valid_idxs])
                    cons_loss = cons_loss + l
                    cnt += 1.0
                if tgt_neg is not None and mask_neg.sum() > 0:
                    valid_idxs = (mask_neg == 1.0).nonzero(as_tuple=False).squeeze(-1)
                    l = bce_loss(pred_neg_logits[valid_idxs], tgt_neg[valid_idxs])
                    cons_loss = cons_loss + l
                    cnt += 1.0
                if cnt > 0:
                    cons_loss = cons_loss / cnt
                else:
                    cons_loss = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)

                raw_loss = lm_loss_pos + alpha_sft * cons_loss

            # guard against NaN/Inf loss before backward
            if not torch.isfinite(raw_loss):
                skipped_sft_batches += 1
                # clear any accidental grads (just in case)
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({'sft_loss': (running_loss_sum / max(1, sft_micro_contrib)) if sft_micro_contrib>0 else float('nan'),
                                  'skipped': skipped_sft_batches, 'accum_steps': grad_accum_steps, 'opt_steps': sft_opt_steps})
                continue

            # gradient accumulation: scale loss down before backward
            loss_to_backward = raw_loss / grad_accum_steps

            # backward (AMP)
            if device_t.type == "cuda":
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            # this micro-batch contributed grads
            sft_micro_contrib += 1
            running_loss_sum += float(raw_loss.detach().cpu())

            # perform optimizer step every grad_accum_steps micro-batches
            if (sft_micro_contrib % grad_accum_steps) == 0:
                # unscale then clip grads and step
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
                        # fallback to normal step if scaler.step fails
                        try:
                            optimizer.step()
                        except Exception:
                            raise
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                sft_opt_steps += 1

            # update pbar
            avg_loss = running_loss_sum / max(1, sft_micro_contrib)
            pbar.set_postfix({'sft_loss': avg_loss, 'accum_steps': grad_accum_steps, 'opt_steps': sft_opt_steps, 'skipped': skipped_sft_batches})

            # cleanup
            try:
                del out_both, last_hidden, pooled, pooled_pos, pooled_neg, pred_pos_logits, pred_neg_logits
            except Exception:
                pass
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        # if there are leftover gradients (micro batches not divisible by grad_accum_steps), flush them
        if (sft_micro_contrib % grad_accum_steps) != 0:
            if sft_micro_contrib % grad_accum_steps != 0:
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
                sft_opt_steps += 1

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
    # Stage 2: DPO (delta-based)
    # -------------------------
    print("[train] START DPO phase (delta-based, curriculum) -- note: DPO strength scaled by dpo_strength=", dpo_strength)
    constr_model.model.eval()
    constr_model.eval()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    # counters for dpo
    dpo_loop_iter = 0
    dpo_micro_contrib = 0
    dpo_opt_steps = 0

    global_step_dpo = 0

    # Precompute hardness scores for dpo_samples
    pair_items_base = [(s.context, s.pos, s.neg, s.pos_constraints, s.neg_constraints) for s in dpo_samples]
    scores = [(hardness_score_for_pair(item), item) for item in pair_items_base]
    scores.sort(key=lambda x: x[0], reverse=True)
    near_pool = [p for s,p in scores[:max(1,int(len(scores)*near_miss_frac))]]
    other_pool = [p for s,p in scores[max(1,int(len(scores)*near_miss_frac)):]]

    # pbar over steps computed per epoch
    for epoch in range(1, epochs_dpo + 1):
        # construct epoch pair_items with oversampling of near-miss
        pair_items = []
        num_near = max(1, int(near_miss_frac * len(pair_items_base)))
        if len(near_pool) > 0:
            pair_items.extend(random.choices(near_pool, k=num_near))
        # fill remaining with random other_pool or random samples
        remaining = len(pair_items_base) - len(pair_items)
        pool_for_rest = other_pool if len(other_pool) >= remaining else (other_pool + near_pool)
        if remaining > 0:
            if len(pool_for_rest) < remaining:
                pair_items.extend(random.choices(pool_for_rest, k=remaining))
            else:
                pair_items.extend(random.sample(pool_for_rest, k=remaining))
        random.shuffle(pair_items)

        # zero grads at epoch start
        optimizer.zero_grad(set_to_none=True)

        pbar_outer = tqdm(range(0, len(pair_items), batch_size), desc=f"DPO epoch {epoch}")
        for i in pbar_outer:
            dpo_loop_iter += 1
            batch = pair_items[i:i+batch_size]
            # build pos/neg texts and prefix lens
            pos_texts = []; neg_texts = []; prefix_lens = []; pos_cons = []; neg_cons = []
            for (ctx, pos, neg, pcons, ncons) in batch:
                # ctx may be dict -> format it
                formatted_ctx = format_context_with_topic(ctx)
                prefix = f"Given the context below, evaluate the hypothesis.\n\n{formatted_ctx}\n\nHypothesis:\n"
                # ensure we append EOS (or sep) to answers without modifying tokenizer
                pos = ensure_eos(pos, tokenizer)
                neg = ensure_eos(neg, tokenizer)
                pos_texts.append(prefix + pos)
                neg_texts.append(prefix + neg)
                prefix_lens.append(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]))
                pos_cons.append(pcons); neg_cons.append(ncons)
            enc_pos = tokenizer(pos_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc_neg = tokenizer(neg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

            # critical: enforce eos at token-id level on these batch encodings
            enc_pos = _enforce_eos_on_encoding(enc_pos, tokenizer, pad_id)
            enc_neg = _enforce_eos_on_encoding(enc_neg, tokenizer, pad_id)

            input_ids_pos = enc_pos["input_ids"].to(device_t); attn_pos = enc_pos["attention_mask"].to(device_t)
            input_ids_neg = enc_neg["input_ids"].to(device_t); attn_neg = enc_neg["attention_mask"].to(device_t)
            prompt_lens = torch.tensor(prefix_lens, dtype=torch.long, device=device_t)

            # pad pos/neg to same length using pad_id computed earlier (never adds tokens)
            max_len_batch = max(input_ids_pos.size(1), input_ids_neg.size(1))
            input_ids_pos, attn_pos = pad_to_length(input_ids_pos, attn_pos, max_len_batch, pad_id)
            input_ids_neg, attn_neg = pad_to_length(input_ids_neg, attn_neg, max_len_batch, pad_id)

            # predict constraints (no grad)
            with torch.no_grad():
                pos_pred_logits = constr_model.predict_constraints_no_grad(input_ids_pos, attention_mask=attn_pos, run_with_hook_fn=run_with_last_hidden_hook)
                neg_pred_logits = constr_model.predict_constraints_no_grad(input_ids_neg, attention_mask=attn_neg, run_with_hook_fn=run_with_last_hidden_hook)
                vpos_pred = torch.sigmoid(pos_pred_logits).sum(dim=1)
                vneg_pred = torch.sigmoid(neg_pred_logits).sum(dim=1)

            def build_violation_tensor(cons_list, pred_tensor):
                out_list = []
                for idx, entry in enumerate(cons_list):
                    if entry is None:
                        out_list.append(float(pred_tensor[idx].item()))
                    else:
                        s = float(sum(entry[:constraint_k]))
                        out_list.append(s)
                return torch.tensor(out_list, dtype=torch.float32, device=device_t)
            vpos = build_violation_tensor(pos_cons, vpos_pred)
            vneg = build_violation_tensor(neg_cons, vneg_pred)

            # concat pos+neg for theta forward
            input_ids_both = torch.cat([input_ids_pos, input_ids_neg], dim=0)
            attn_both = torch.cat([attn_pos, attn_neg], dim=0)
            prompt_lens_both = torch.cat([prompt_lens, prompt_lens], dim=0)

            constr_model.model.train()
            with autocast(enabled=(device_t.type == "cuda")):
                out_both_theta, last_hidden_theta = run_with_last_hidden_hook(constr_model.model,
                                                                              input_ids=input_ids_both,
                                                                              attention_mask=attn_both)
                logits_both_theta = out_both_theta.logits

            logp_both_theta = compute_sequence_logprobs_from_logits(logits_both_theta, input_ids_both, attn_both, prompt_lens_both)
            B = input_ids_pos.size(0)
            logp_pos_theta = logp_both_theta[:B]
            logp_neg_theta = logp_both_theta[B:]

            # delta-based adj
            delta_logp = logp_pos_theta - logp_neg_theta
            delta_v = vpos - vneg
            d_theta_adj = delta_logp - lambda_v * delta_v

            # reference logprobs (no grad)
            with torch.no_grad():
                if ref_on_cpu and device_t.type == "cuda":
                    in_ids_ref = input_ids_both.cpu()
                    attn_ref = attn_both.cpu()
                    out_both_ref, _ = run_with_last_hidden_hook(ref_model, input_ids=in_ids_ref, attention_mask=attn_ref)
                    logits_ref = out_both_ref.logits.to(device_t)
                    logp_both_ref = compute_sequence_logprobs_from_logits(logits_ref, input_ids_both, attn_both, prompt_lens_both)
                else:
                    out_both_ref, _ = run_with_last_hidden_hook(ref_model, input_ids=input_ids_both, attention_mask=attn_both)
                    logits_ref = out_both_ref.logits
                    logp_both_ref = compute_sequence_logprobs_from_logits(logits_ref, input_ids_both, attn_both, prompt_lens_both)

            logp_pos_ref = logp_both_ref[:B]
            logp_neg_ref = logp_both_ref[B:]
            d_ref = logp_pos_ref - logp_neg_ref

            diff = d_theta_adj - d_ref

            # beta warmup schedule
            global_step_dpo += 1
            if global_step_dpo < beta_warmup_steps:
                beta = beta_start + (beta_target - beta_start) * (global_step_dpo / max(1, beta_warmup_steps))
            else:
                beta = beta_target

            scaled = beta * diff
            scaled = torch.clamp(scaled, min=-50.0, max=50.0)
            per_example_loss = -F.logsigmoid(scaled)
            loss_dpo = per_example_loss.mean()

            # auxiliary constraint BCE (compute pooled last hidden no_grad then head in train mode)
            with torch.no_grad():
                _, last_hidden_pooled = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both, attention_mask=attn_both)
                mask = attn_both.unsqueeze(-1).to(dtype=last_hidden_pooled.dtype)
                pooled_all = (last_hidden_pooled * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
                pooled_pos = pooled_all[:B].detach()
                pooled_neg = pooled_all[B:].detach()

            out_pos_logits = constr_model.head_forward(pooled_pos)
            out_neg_logits = constr_model.head_forward(pooled_neg)

            # build targets
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
                cons_loss_dpo = cons_loss_dpo + bce_loss(out_pos_logits[valid_idxs], tgt_pos[valid_idxs])
                cnt += 1.0
            if sum(neg_mask) > 0:
                tgt_neg = torch.tensor(neg_targets, dtype=logits_dtype, device=logits_device)
                valid_idxs = (torch.tensor(neg_mask, device=logits_device) == 1.0).nonzero(as_tuple=False).squeeze(-1)
                cons_loss_dpo = cons_loss_dpo + bce_loss(out_neg_logits[valid_idxs], tgt_neg[valid_idxs])
                cnt += 1.0
            if cnt > 0:
                cons_loss_dpo = cons_loss_dpo / cnt
            else:
                cons_loss_dpo = torch.tensor(0.0, device=logits_device, dtype=logits_dtype)

            # Make DPO weaker by scaling its primary loss term
            raw_loss = dpo_strength * loss_dpo + 0.5 * alpha_dpo * cons_loss_dpo

            # guard loss for finiteness
            if not torch.isfinite(raw_loss):
                # skip this micro-batch
                pbar_outer.set_postfix({'dpo_raw_loss': float('nan'), 'beta': float(beta), 'accum_steps': grad_accum_steps, 'opt_steps': dpo_opt_steps})
                continue

            # gradient accumulation
            loss_to_backward = raw_loss / grad_accum_steps

            if device_t.type == "cuda":
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            dpo_micro_contrib += 1

            # perform optimizer step every grad_accum_steps micro-batches
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

            # if leftover at epoch end will be flushed below
            global_step_dpo += 1

            pbar_outer.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta),
                                    'accum_steps': grad_accum_steps, 'opt_steps': dpo_opt_steps})

            # cleanup
            try:
                for v in ("out_both_theta","logits_both_theta","out_both_ref","logits_ref","last_hidden_theta","last_hidden_pooled",
                          "out_pos_logits","out_neg_logits","pooled_all","pooled_pos","pooled_neg"):
                    if v in locals():
                        del locals()[v]
            except Exception:
                pass
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        # flush leftover grads if any
        if (dpo_micro_contrib % grad_accum_steps) != 0:
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
    p.add_argument('--quant_bits', type=int, default=16, choices=[4,8,16], help='Quantization bits for base model (4/8/16). 16 = fp16 (no quant).')
    p.add_argument('--grad_ckpt', action='store_true', default=True, help='Enable gradient checkpointing to save memory')
    p.add_argument('--ref_on_cpu', action='store_true', default=False, help='Keep reference model on CPU to save GPU memory (slower)')
    p.add_argument('--sft_frac', type=float, default=0.5, help='Fraction of data used for SFT (rest for DPO)')
    p.add_argument('--near_miss_frac', type=float, default=0.30, help='Fraction of DPO set to oversample near-miss examples')
    p.add_argument('--dpo_strength', type=float, default=0.5, help='Scale factor for DPO primary loss (0 disables DPO objective entirely)')
    p.add_argument('--grad_accum_steps', type=int, default=4, help='Number of micro-batches to accumulate before performing an optimizer step')
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
    )
