#!/usr/bin/env python3
"""
Trainer3_fixed_full.py

Fixed Trainer (full) — SFT + DPO with robust EOS preservation.
Key fixes:
 - reference snapshot taken after SFT (prevents DPO from erasing SFT behavior)
 - token-level EOS preservation (tokenize_preserve_eos)
 - separate prefix clipping for pos and neg
 - optional tokenwise KL anchor to keep theta close to ref on generation tokens
 - improved EOS diagnostics and safe guards
Usage: same as previous trainer. See CLI flags (--help).
"""
from __future__ import annotations
import argparse, json, os, random, copy, time, warnings
from typing import List, Any, Optional, Dict

# HF cache defaults (adjust if you want)
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
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

def tokenize_preserve_eos(tokenizer, texts: List[str], max_length: int, pad_token_id: int, truncation_side: str = 'right'):
    """
    Tokenize a list of texts and ensure each sequence ends with tokenizer.eos_token_id.
    Returns: input_ids tensor, attention_mask tensor, and raw token lists (pre-padding).
    This avoids tokenizer truncation removing the eos id.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # Tokenize without truncation nor padding to inspect raw ids
    encodings = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    token_lists = []
    for ids in encodings['input_ids']:
        ids = list(ids)
        # clip/truncate to max_length (raw token list) but reserve final slot for eos if possible
        if len(ids) > max_length:
            if truncation_side == 'right':
                ids = ids[:max_length]
            else:
                ids = ids[-max_length:]
        # ensure eos present as final token-id
        if eos_id is not None:
            if len(ids) == 0:
                # create eos-only sequence
                ids = [eos_id]
            elif ids[-1] != eos_id:
                if len(ids) >= max_length:
                    # sacrifice last token to ensure eos fits
                    ids[-1] = eos_id
                else:
                    ids.append(eos_id)
        token_lists.append(ids)

    # pad to same length
    max_len_batch = max(len(x) for x in token_lists) if token_lists else 0
    padded_ids = []
    attn_masks = []
    for ids in token_lists:
        attn = [1] * len(ids) + [0] * (max_len_batch - len(ids))
        padded = ids + [pad_token_id] * (max_len_batch - len(ids))
        padded_ids.append(padded)
        attn_masks.append(attn)
    if padded_ids:
        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long)
    else:
        input_ids = torch.empty((0, 0), dtype=torch.long)
        attention_mask = torch.empty((0, 0), dtype=torch.long)
    return input_ids, attention_mask, token_lists

def safe_ensure_eos_text(text: str, tokenizer):
    """Text-level fallback: append eos token string if not present."""
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
# Data types
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
# Constraint head wrapper
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

# -------------------------
# Hook helper (robust)
# -------------------------
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

# -------------------------
# Sequence logprobs helper
# -------------------------
def compute_sequence_logprobs_from_logits(logits, input_ids, attn_mask, prompt_lens):
    # logits: [B, T, V], input_ids: [B,T], attn_mask: [B,T], prompt_lens: [B]
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

# -------------------------
# LoRA L1 penalty
# -------------------------
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
    target_modules: Optional[List[str]] = None,
    use_ard_lora: bool = False,
    l1_lambda: float = 1e-6,
    max_len: int = 1024,
    quant_bits: int = 4,
    gradient_checkpointing: bool = True,
    ref_on_cpu: bool = True,
    grad_accum_steps: int = 8,
    gamma_margin: float = 1.0,
    constraint_penalty_weight: float = 1.0,
    flag_weights: Dict[str, float] = None,
    ema_decay: float = 0.999,
    ema_update_every: int = 100,
    eos_penalty_weight: float = 0.2,
    debug_eos: bool = False,
    anchor_coeff: float = 0.01,   # token-KL anchor coefficient (0 to disable)
):
    os.makedirs(out_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available() and device != 'cpu'
    device_t = torch.device(device if use_cuda else 'cpu')
    print("[train] device:", device_t)

    print('[train] loading tokenizer:', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, 'pad_token_id', None) is not None else (tokenizer.eos_token_id or 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id == eos_id:
        warnings.warn("pad_token_id equals eos_token_id — padding will be indistinguishable from EOS. Consider setting a distinct pad token.")

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

    # load data
    print('[train] loading dataset:', data_path)
    raw = load_jsonl(data_path)
    samples = []
    for r in raw:
        s = PairSample(r)
        formatted_ctx = ''
        if isinstance(s.context, dict):
            topic = s.context.get('topic', '')
            ancestors = s.context.get('ancestors', []) or []
            formatted_ctx = (f"Topic: {topic}\n\nPrior hypotheses:\n" + "\n\n".join(ancestors)).strip()
        else:
            formatted_ctx = str(s.context)
        prefix_full = f"Given the context below, evaluate the hypothesis.\n\n{formatted_ctx}\n\nHypothesis:\n"
        s.prefix_full = prefix_full
        # text-level fallback (we rely on token-level tokenizer for true EOS enforcement)
        s.pos_full = safe_ensure_eos_text(prefix_full + (s.pos or ''), tokenizer)
        s.neg_full = safe_ensure_eos_text(prefix_full + (s.neg or ''), tokenizer)
        try:
            s.example_len = len(tokenizer(s.pos_full, add_special_tokens=False)["input_ids"]) + len(tokenizer(s.neg_full, add_special_tokens=False)["input_ids"]) if s.pos_full and s.neg_full else 0
        except Exception:
            s.example_len = 0
        samples.append(s)
    random.shuffle(samples)
    n = len(samples)
    print(f"[train] loaded {n} pairs")

    # collate for SFT: use token-level EOS-preserving tokenizer and compute prefix_lens for pos/neg separately
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

        # compute prefix lens per side and clip to each tokenized sequence length
        prefix_lens_pos = []
        prefix_lens_neg = []
        for p, pos_tok, neg_tok in zip(prefix_cache, pos_lists, neg_lists):
            try:
                pl = len(tokenizer(p, add_special_tokens=False)['input_ids'])
            except Exception:
                pl = 0
            prefix_lens_pos.append(min(pl, len(pos_tok)))
            prefix_lens_neg.append(min(pl, len(neg_tok)))

        # For SFT collate, we'll return prompt_lens as two tensors (pos then neg)
        prompt_lens_pos = torch.tensor(prefix_lens_pos, dtype=torch.long)
        prompt_lens_neg = torch.tensor(prefix_lens_neg, dtype=torch.long)

        enc_pos = {'input_ids': enc_pos_ids, 'attention_mask': enc_pos_attn}
        enc_neg = {'input_ids': enc_neg_ids, 'attention_mask': enc_neg_attn}
        return enc_pos, enc_neg, prompt_lens_pos, prompt_lens_neg, pos_cons, neg_cons

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
        for enc_pos, enc_neg, prompt_lens_pos, prompt_lens_neg, pos_cons_list, neg_cons_list in tqdm(sft_loader, desc=f'SFT epoch {epoch}'):
            input_ids_pos = enc_pos['input_ids'].to(device_t); attn_pos = enc_pos['attention_mask'].to(device_t)
            input_ids_neg = enc_neg['input_ids'].to(device_t); attn_neg = enc_neg['attention_mask'].to(device_t)

            # make both same width
            max_len_batch = max(input_ids_pos.size(1), input_ids_neg.size(1))
            input_ids_pos, attn_pos = pad_to_length(input_ids_pos, attn_pos, max_len_batch, pad_id)
            input_ids_neg, attn_neg = pad_to_length(input_ids_neg, attn_neg, max_len_batch, pad_id)

            input_ids_both = torch.cat([input_ids_pos, input_ids_neg], dim=0)
            attn_both = torch.cat([attn_pos, attn_neg], dim=0)
            labels_both = input_ids_both.clone()
            Bpos = input_ids_pos.size(0)

            # create prompt lens for pos and neg then combine
            prompt_lens_pos = prompt_lens_pos.to(device_t)
            prompt_lens_neg = prompt_lens_neg.to(device_t)
            prompt_lens_both = torch.cat([prompt_lens_pos, prompt_lens_neg], dim=0)

            # mask prompt tokens in labels
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

                # build BCE constraint targets
                def make_targets(cons_list):
                    out = []
                    mask_t = []
                    for item in cons_list:
                        if item is None:
                            out.append([0.0]*constraint_k); mask_t.append(0.0)
                        else:
                            v = list(item[:constraint_k]) if isinstance(item, (list,tuple)) else [0.0]*constraint_k
                            out.append([float(x or 0.0) for x in v]); mask_t.append(1.0)
                    return torch.tensor(out, dtype=pred_pos_logits.dtype, device=pred_pos_logits.device), torch.tensor(mask_t, dtype=torch.float32, device=pred_pos_logits.device)

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
            # save adapters + tokenizer + head
            constr_model.model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            torch.save({'head_state': constr_model.head.state_dict()}, os.path.join(epoch_dir, 'constraint_head.pt'))
        except Exception:
            torch.save({k:v.cpu() for k,v in constr_model.state_dict().items()}, os.path.join(epoch_dir, 'full_state.pt'))

    # --- CRITICAL: create reference snapshot from SFT snapshot (instead of earlier) ---
    print('[train] creating reference model (deepcopy) from SFT snapshot')
    ref_model = copy.deepcopy(constr_model.model)
    try:
        ref_device = 'cpu' if ref_on_cpu else device_t
        ref_model.to(ref_device)
    except Exception:
        pass
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # -------------------------
    # DPO phase
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
            # build full texts
            pos_full_texts=[]; neg_full_texts=[]; prefix_lens_full_pos=[]; prefix_lens_full_neg=[]
            delta_q_list = []
            flags_batch = []
            for s in batch:
                pos_full_texts.append(s.pos_full); neg_full_texts.append(s.neg_full)
                try:
                    pl = len(tokenizer(s.prefix_full, add_special_tokens=False)['input_ids'])
                except Exception:
                    pl = 0
                prefix_lens_full_pos.append(pl)
                prefix_lens_full_neg.append(pl)
                delta_q_list.append(0.0 if s.Delta_Q is None else float(s.Delta_Q))
                flags_batch.append({'reject': s.flag_reject, 'low_reasoning': s.flag_low_reasoning, 'hallucination': s.flag_hallucination})

            # tokenize fulls using token-level EOS-preserving helper
            input_ids_pos_full, attn_pos_full, pos_lists = tokenize_preserve_eos(
                tokenizer, pos_full_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right'
            )
            input_ids_neg_full, attn_neg_full, neg_lists = tokenize_preserve_eos(
                tokenizer, neg_full_texts, max_length=max_len, pad_token_id=pad_id, truncation_side='right'
            )

            # clip prefix lens to side-specific tokenized lengths
            clipped_prefix_lens_pos = [min(pl, len(ids)) for pl, ids in zip(prefix_lens_full_pos, pos_lists)]
            clipped_prefix_lens_neg = [min(pl, len(ids)) for pl, ids in zip(prefix_lens_full_neg, neg_lists)]

            input_ids_pos_full = input_ids_pos_full.to(device_t); attn_pos_full = attn_pos_full.to(device_t)
            input_ids_neg_full = input_ids_neg_full.to(device_t); attn_neg_full = attn_neg_full.to(device_t)
            prompt_lens_full_pos = torch.tensor(clipped_prefix_lens_pos, dtype=torch.long, device=device_t)
            prompt_lens_full_neg = torch.tensor(clipped_prefix_lens_neg, dtype=torch.long, device=device_t)

            max_len_full = max(input_ids_pos_full.size(1), input_ids_neg_full.size(1))
            input_ids_pos_full, attn_pos_full = pad_to_length(input_ids_pos_full, attn_pos_full, max_len_full, pad_id)
            input_ids_neg_full, attn_neg_full = pad_to_length(input_ids_neg_full, attn_neg_full, max_len_full, pad_id)

            input_ids_both_full = torch.cat([input_ids_pos_full, input_ids_neg_full], dim=0)
            attn_both_full = torch.cat([attn_pos_full, attn_neg_full], dim=0)
            prompt_lens_both_full = torch.cat([prompt_lens_full_pos, prompt_lens_full_neg], dim=0)

            # compute theta logits and logprobs (with grad)
            constr_model.model.train()
            with autocast(enabled=(device_t.type=='cuda')):
                out_both_theta_full, last_hidden_theta_full = run_with_last_hidden_hook(constr_model.model, input_ids=input_ids_both_full, attention_mask=attn_both_full)
                logits_both_theta_full = out_both_theta_full.logits
                logp_both_theta_full = compute_sequence_logprobs_from_logits(logits_both_theta_full, input_ids_both_full, attn_both_full, prompt_lens_both_full)
                B_full = input_ids_pos_full.size(0)
                logp_pos_theta_full = logp_both_theta_full[:B_full]; logp_neg_theta_full = logp_both_theta_full[B_full:]

            # compute ref logits/logprobs on CPU with no grad in small batches to save VRAM
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

            # compute DPO deltas
            delta_logp_full = logp_pos_theta_full - logp_neg_theta_full
            d_theta_adj_full = delta_logp_full - lambda_v * torch.tensor(0.0, device=device_t)
            d_ref_full = logp_pos_ref_full - logp_neg_ref_full
            diff_full = d_theta_adj_full - d_ref_full

            # incorporate dataset margin ΔQ
            delta_q_tensor = torch.tensor([min(max(d, -1.0), 1.0) for d in delta_q_list], dtype=torch.float32, device=device_t)
            beta_val = (beta_target if global_step >= beta_warmup_steps else (beta_start + (beta_target - beta_start) * (global_step / max(1, beta_warmup_steps))))
            scaled_full = beta_val * (diff_full - gamma_margin * delta_q_tensor)
            scaled_full = torch.clamp(scaled_full, min=-50.0, max=50.0)
            per_example_loss_full = -F.logsigmoid(scaled_full)
            loss_full = per_example_loss_full.mean()

            # constraint penalty using available metrics per-sample
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

            # EOS missing penalty: check pos_lists and neg_lists (raw token lists returned earlier)
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

            # Flag-based downweighting
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

            # optional tokenwise KL anchor between theta and ref on generation tokens
            if anchor_coeff and anchor_coeff > 0.0:
                # compute p_ref and logp_theta
                with torch.no_grad():
                    # logits_both_ref_full is already computed and on device_t
                    pass
                # careful: logits could be float16 -- cast to float32 for stable softmax
                logp_theta_all = F.log_softmax(logits_both_theta_full.float(), dim=-1)
                p_ref_all = F.softmax(logits_both_ref_full.float(), dim=-1)
                kl_per_token = (p_ref_all * (torch.log(torch.clamp(p_ref_all, 1e-9)) - logp_theta_all)).sum(dim=-1)  # [B, T]
                # build generation mask same as earlier (shifted)
                B_mask, T_mask = input_ids_both_full.size()
                # We need a mask shaped [B, T-1] for token positions corresponding to labels
                gen_mask = torch.zeros((B_mask, T_mask-1), device=device_t, dtype=torch.float32)
                for bi, pl in enumerate(prompt_lens_both_full):
                    start = max(int(pl.item()) - 1, 0)
                    if start < (T_mask - 1):
                        gen_mask[bi, start:] = 1.0
                # align kl_per_token to labels: remove last time-step to match (we computed kl per token)
                kl_for_labels = kl_per_token[:, :-1]  # shape [B, T-1]
                masked_kl = (kl_for_labels * gen_mask).sum() / max(1.0, gen_mask.sum())
                raw_loss = raw_loss + anchor_coeff * masked_kl

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
                # occasional EMA update (optional)
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

            # debug eos diagnostics: compute P_theta(eos | prompt) for first pos example (if requested)
            if debug_eos:
                try:
                    with torch.no_grad():
                        # first example logit matrix (theta)
                        logits_first = logits_both_theta_full[0].float()  # [T, V]
                        pl_first = int(prompt_lens_both_full[0].item())
                        eos_logp = None
                        if eos_id is not None and pl_first > 0:
                            idx = pl_first - 1
                            if 0 <= idx < logits_first.size(0):
                                eos_logp = F.log_softmax(logits_first, dim=-1)[idx, eos_id].item()
                        pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_val), 'eos_logp_first': eos_logp})
                except Exception:
                    pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_val)})
            else:
                pbar.set_postfix({'dpo_raw_loss': float(raw_loss.detach().cpu()), 'beta': float(beta_val)})

            if device_t.type == 'cuda':
                torch.cuda.empty_cache()

        epoch_dir = os.path.join(out_dir, f'dpo_epoch{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        try:
            constr_model.model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
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
    p.add_argument('--epochs_dpo', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--constraint_k', type=int, default=6)
    p.add_argument('--alpha_sft', type=float, default=0.2)
    p.add_argument('--alpha_dpo', type=float, default=0.5)
    p.add_argument('--beta_start', type=float, default=0.05)
    p.add_argument('--beta_target', type=float, default=0.10)
    p.add_argument('--beta_warmup_steps', type=int, default=500)
    p.add_argument('--lambda_v', type=float, default=0.7)
    p.add_argument('--lora_r', type=int, default=4)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--quant_bits', type=int, default=4, choices=[4,8,16])
    p.add_argument('--grad_ckpt', default=True)
    p.add_argument('--ref_on_cpu', default=False)
    p.add_argument('--gamma_margin', type=float, default=1.0, help='scale of ΔQ margin inside DPO')
    p.add_argument('--constraint_penalty_weight', type=float, default=0.5)
    p.add_argument('--use_ard_lora', default=True, help='Enable ARD L1 on LoRA params')
    p.add_argument('--l1_lambda', type=float, default=5e-7)
    p.add_argument('--eos_penalty_weight', type=float, default=0.1)
    p.add_argument('--debug_eos', default=True)
    p.add_argument('--anchor_coeff', type=float, default=0.01, help='Token-KL anchor coefficient (0 disables)')
    return p.parse_args()

if __name__ == '__main__':
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
        quant_bits=args.quant_bits,
        gradient_checkpointing=args.grad_ckpt,
        ref_on_cpu=args.ref_on_cpu,
        gamma_margin=args.gamma_margin,
        constraint_penalty_weight=args.constraint_penalty_weight,
        use_ard_lora=args.use_ard_lora,
        l1_lambda=args.l1_lambda,
        eos_penalty_weight=args.eos_penalty_weight,
        debug_eos=args.debug_eos,
        anchor_coeff=args.anchor_coeff,
    )
