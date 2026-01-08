#!/usr/bin/env python3
"""
continue_from_lora_fixed_v2.py

Resumes training from a LoRA adapter while training:
 - LM loss on accepted hypotheses (SFT)
 - Aux BCE constraint head on rejected hypotheses (when labeled)
Fixes:
 - Aux head trained on rejected text (not accepted)
 - Pooling only over hypothesis tokens (prefix excluded)
 - No unnecessary requires_grad manipulations
 - Proper aux warmup and stable gradient accumulation
"""

from __future__ import annotations
import os
os.environ.setdefault('HF_HOME', './hf_cache')
os.environ.setdefault("TRANSFORMERS_CACHE", "./hf_cache/transformers")

import json
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training

from torch.cuda.amp import autocast, GradScaler

# ---------------------------
# Utilities
# ---------------------------
def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for L in f:
            L = L.strip()
            if not L:
                continue
            out.append(json.loads(L))
    return out

def make_bnb_config(bits: int):
    if bits is None or bits == 16:
        return None
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    raise ValueError("quant_bits must be one of 4,8,16")

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
    if eos_str is not None and not text.endswith(eos_str):
        return text + eos_str
    return text

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

# ---------------------------
# Dataset & collate
# ---------------------------
class SingleHypDataset(Dataset):
    """
    Each record may contain:
      - context: structured context (topic + ancestors)
      - accepted: accepted hypothesis (string)
      - rejected: rejected hypothesis (string)  <-- used for aux supervision
      - label: optional dict with 'violations': [ { "type": "...", ... }, ... ]
    """
    def __init__(self, records: List[Dict], violation_types: List[str]):
        self.records = records
        self.violation_types = violation_types

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ctx = r.get("context", {})
        accepted = r.get("accepted", "")
        rejected = r.get("rejected", "")  # may be empty
        label = r.get("label", None)
        # Convert structured label -> multi-hot vector for allowed violation types if present
        multi = None
        mask = 0.0
        if label and isinstance(label, dict):
            vio_list = label.get("violations", []) or []
            onehot = [0.0]*len(self.violation_types)
            found_any = False
            for v in vio_list:
                if isinstance(v, dict):
                    t = v.get("type")
                else:
                    t = v
                if t in self.violation_types:
                    onehot[self.violation_types.index(t)] = 1.0
                    found_any = True
            if found_any:
                multi = onehot
                mask = 1.0
        return {"context": ctx, "accepted": accepted, "rejected": rejected, "violation_multi": multi, "violation_mask": mask}

def format_context_with_topic(context_obj: dict) -> str:
    if context_obj is None:
        return "Prior hypotheses:\n"
    if isinstance(context_obj, str):
        return f"Topic : \n\nPrior hypotheses:\n{context_obj}"
    if not isinstance(context_obj, dict):
        return f"Prior hypotheses:\n{str(context_obj)}"
    topic = str(context_obj.get("topic","") or "").strip()
    ancestors = context_obj.get("ancestors",[]) or []
    ancestors_block = "\n\n".join(a for a in ancestors if a)
    if topic:
        if ancestors_block:
            return f"Topic : {topic}\n\nPrior hypotheses:\n{ancestors_block}"
        else:
            return f"Topic : {topic}\n\nPrior hypotheses:\n"
    else:
        return f"Prior hypotheses:\n{ancestors_block}"

def collate_batch(batch: List[dict], tokenizer, max_len=1024, pad_id=None, violation_dim: int = 0):
    """
    Returns:
      - accepted: tokenization (input_ids, attention_mask, labels) -> LM training
      - rejected: tokenization (rejected_input_ids, rejected_attention_mask) -> aux training
      - prefix_lens: tensor([plen,...]) for hypothesis pooling
      - violation_multi, violation_mask
    """
    accepted_texts = []
    rejected_texts = []
    prefix_lens = []
    multis = []
    masks = []

    for item in batch:
        ctx_block = format_context_with_topic(item["context"])
        prefix = f"""
{ctx_block}

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
        acc_hyp = ensure_eos(item["accepted"], tokenizer)
        rej_hyp = ensure_eos(item.get("rejected", ""), tokenizer)
        accepted_texts.append(prefix + acc_hyp)
        rejected_texts.append(prefix + rej_hyp)
        # prefix token length (same prefix for both tokenizations)
        plen = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        prefix_lens.append(plen)
        multis.append(item.get("violation_multi", None))
        masks.append(float(item.get("violation_mask", 0.0)))

    enc_acc = tokenizer(accepted_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_acc = _enforce_eos_on_encoding(enc_acc, tokenizer, pad_id)
    enc_rej = tokenizer(rejected_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_rej = _enforce_eos_on_encoding(enc_rej, tokenizer, pad_id)

    input_ids = enc_acc["input_ids"]
    attn = enc_acc["attention_mask"]
    labels = input_ids.clone()
    for i, plen in enumerate(prefix_lens):
        if plen > 0:
            labels[i, :plen] = -100

    if pad_id is None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    multi_tensor = None
    mask_tensor = torch.tensor(masks, dtype=torch.float32)
    if violation_dim > 0:
        onehots = []
        for m in multis:
            if m is None:
                onehots.append([0.0]*violation_dim)
            else:
                v = list(m) + [0.0]*max(0, violation_dim - len(m))
                onehots.append(v[:violation_dim])
        multi_tensor = torch.tensor(onehots, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.long),
        "violation_multi": multi_tensor,
        "violation_mask": mask_tensor,
        "rejected_input_ids": enc_rej["input_ids"],
        "rejected_attention_mask": enc_rej["attention_mask"],
    }

# ---------------------------
# Constraint head
# ---------------------------
class ConstraintHead(nn.Module):
    def __init__(self, hidden_size: int, k: int = 8, head_hidden: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, k)
        )

    def forward(self, pooled):
        return self.head(pooled)

# ---------------------------
# Training (full fixed)
# ---------------------------
def resume_train(
    lora_dir: str,
    base_model_name: str,
    data_path: str,
    out_dir: str,
    device: str = "cuda",
    quant_bits: int = 4,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    epochs: int = 1,
    lr: float = 2e-4,
    lambda_v: float = 0.7,
    aux_warmup_epochs: int = 1,
    gradient_checkpointing: bool = True,
    enable_amp: bool = True,
    save_every_epoch: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    print("[resume] device:", device_t)

    qconf = make_bnb_config(quant_bits)

    print("[resume] load tokenizer:", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def load_base(name, quant_conf):
        device_map = "auto" if torch.cuda.device_count() > 1 else None
        if quant_conf is not None:
            print(f"[resume] loading base model {name} with quantization={quant_bits} bits")
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant_conf, device_map=device_map, trust_remote_code=True)
        else:
            print(f"[resume] loading base model {name} in fp16")
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True)
        try:
            model.config.use_cache = False
        except Exception:
            pass
        return model

    base = load_base(base_model_name, qconf)

    if quant_bits in (4,8):
        print("[resume] prepare_model_for_kbit_training")
        base = prepare_model_for_kbit_training(base)

    # load PEFT adapter
    peft_model = None
    try:
        print(f"[resume] attempting PeftModel.from_pretrained(base, '{lora_dir}')")
        peft_model = PeftModel.from_pretrained(base, lora_dir, device_map="auto")
    except Exception as e:
        print("[resume] PeftModel.from_pretrained failed:", e)
        print("[resume] fallback: attach get_peft_model and try load state_dict")
        peft_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        peft_model = get_peft_model(base, peft_cfg)
        try:
            peft_model.load_state_dict(torch.load(os.path.join(lora_dir, "pytorch_model.bin")), strict=False)
            print("[resume] loaded state_dict into PEFT model (fallback).")
        except Exception as e2:
            print("[resume] fallback load failed:", e2)
            raise RuntimeError("Unable to load LoRA adapter from the provided directory.") from e2

    model = peft_model
    model.train()

    try:
        if gradient_checkpointing:
            print("[resume] enabling gradient checkpointing on base model")
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
    except Exception:
        pass

    raw = load_jsonl(data_path)
    print(f"[resume] loaded {len(raw)} records from {data_path}")

    VIOLATION_TYPES = [
        "dimension_mismatch", "illegal_exponent", "sign_error", "definition_violation",
        "monotonicity_violation", "illegal_variable_introduction", "contradicts_prior_axiom",
        "non_removable_constraint", "unjustified_operation", "circular_reasoning", "category_error"
    ]
    violation_dim = len(VIOLATION_TYPES)

    # compute pos_weight
    N = len(raw)
    pos_counts = [0]*violation_dim
    for r in raw:
        lab = r.get("label", {}) or {}
        vio_list = lab.get("violations", []) if isinstance(lab, dict) else []
        for v in vio_list:
            t = v.get("type") if isinstance(v, dict) else v
            if t in VIOLATION_TYPES:
                pos_counts[VIOLATION_TYPES.index(t)] += 1
    pos_weight_list = []
    eps = 1e-6
    for c in pos_counts:
        neg = max(1, N - c)
        if c == 0:
            w = float(min(100.0, neg / 1.0))
        else:
            w = float(neg / (c + eps))
            w = min(w, 100.0)
        pos_weight_list.append(w)
    pos_weight_tensor = torch.tensor(pos_weight_list, dtype=torch.float32, device=device_t)

    dataset = SingleHypDataset(raw, VIOLATION_TYPES)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer, max_len=1024, pad_id=pad_id, violation_dim=violation_dim))

    # constraint head
    hidden_size = getattr(model, "config", None).hidden_size if hasattr(model, "config") and hasattr(model.config, "hidden_size") else getattr(model.config, "hidden_size", 1024)
    # try loading existing constraint head from lora_dir
    constraint_head = None
    head_path_candidates = [os.path.join(lora_dir, "constraint_head.pt"), os.path.join(lora_dir, "constraint_head.pth")]
    loaded_head = False
    for hp in head_path_candidates:
        if os.path.exists(hp):
            try:
                ck = torch.load(hp, map_location="cpu")
                saved_state = ck.get("head_state", ck)
                ch = ConstraintHead(hidden_size=hidden_size, k=violation_dim)
                ch.head.load_state_dict(saved_state)
                constraint_head = ch.to(device_t)
                loaded_head = True
                print("[resume] loaded constraint head from", hp)
                break
            except Exception as e:
                print("[resume] failed to load constraint head from", hp, "error:", e)
    if not loaded_head:
        constraint_head = ConstraintHead(hidden_size=hidden_size, k=violation_dim).to(device_t)
        print("[resume] created new constraint head with k =", violation_dim)

    peft_params = [p for n,p in model.named_parameters() if p.requires_grad]
    trainable = list(peft_params) + list(constraint_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)

    if (pos_weight_tensor is not None) and pos_weight_tensor.shape[0] == violation_dim:
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        bce_loss = nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=(device_t.type == "cuda" and enable_amp))

    # training loop
    global_step = 0
    for epoch in range(1, epochs+1):
        model.train()
        constraint_head.train()
        pbar = tqdm(loader, desc=f"resume-epoch-{epoch}")
        running_loss = 0.0
        micro = 0
        opt_steps = 0

        aux_scale = (epoch / max(1, aux_warmup_epochs)) if aux_warmup_epochs > 0 else 1.0
        aux_scale = min(1.0, aux_scale)
        lambda_v_current = float(lambda_v * aux_scale)

        for batch in pbar:
            acc_input_ids = batch["input_ids"].to(device_t)
            acc_attn = batch["attention_mask"].to(device_t)
            acc_labels = batch["labels"].to(device_t)
            prefix_lens = batch["prefix_lens"].to(device_t)
            vio_multi = batch.get("violation_multi", None)
            vio_mask = batch.get("violation_mask", None).to(device_t)

            rej_input_ids = batch["rejected_input_ids"].to(device_t)
            rej_attn = batch["rejected_attention_mask"].to(device_t)

            if vio_multi is not None:
                vio_multi = vio_multi.to(device_t)

            # Forward LM on accepted (LM-only forward)
            with autocast(enabled=(device_t.type == "cuda" and enable_amp)):
                out_acc = model(input_ids=acc_input_ids, attention_mask=acc_attn, labels=acc_labels, return_dict=True)
                lm_loss = out_acc.loss

            # Forward on rejected to compute aux logits (AUX-only forward)
            cons_loss = torch.tensor(0.0, device=device_t)
            if vio_multi is not None and vio_mask.sum() > 0:
                # request hidden_states for pooling
                with autocast(enabled=(device_t.type == "cuda" and enable_amp)):
                    out_rej = model(input_ids=rej_input_ids, attention_mask=rej_attn, output_hidden_states=True, return_dict=True)
                    last = out_rej.hidden_states[-1]  # [B, L, H]
                    B, L, H = last.shape
                    device_for_pool = last.device
                    pooled_list = []
                    # Use prefix_lens to pool only hypothesis region
                    for b in range(B):
                        plen = int(prefix_lens[b].item())
                        row_attn = rej_attn[b]
                        nonpad_idxs = (row_attn == 1).nonzero(as_tuple=True)[0]
                        if len(nonpad_idxs) == 0:
                            pooled_list.append(torch.zeros(H, device=device_for_pool, dtype=last.dtype))
                            continue
                        last_idx = int(nonpad_idxs[-1].item())
                        # pool over [plen, last_idx] inclusive
                        if plen > last_idx:
                            pooled_list.append(torch.zeros(H, device=device_for_pool, dtype=last.dtype))
                            continue
                        seg = last[b, plen:(last_idx+1), :]  # [S, H]
                        # mean pool (seg should be non-pad)
                        denom = seg.shape[0]
                        if denom <= 0:
                            pooled_list.append(torch.zeros(H, device=device_for_pool, dtype=last.dtype))
                        else:
                            pooled_vec = seg.mean(dim=0)
                            pooled_list.append(pooled_vec)
                    pooled = torch.stack(pooled_list, dim=0)  # [B, H]

                    pred_logits = constraint_head(pooled)  # [B, K]

                    # compute cons_loss only for labelled samples
                    sel = (vio_mask == 1.0).nonzero(as_tuple=True)[0]
                    if sel.numel() > 0:
                        tgt = vio_multi[sel]
                        out_sel = pred_logits[sel]
                        if out_sel.shape[-1] != tgt.shape[-1]:
                            raise RuntimeError(f"Shape mismatch aux head {out_sel.shape[-1]} vs target {tgt.shape[-1]}")
                        cons_loss = bce_loss(out_sel, tgt)
                    else:
                        cons_loss = torch.tensor(0.0, device=device_t)

                    # cleanup for memory
                    try:
                        del out_rej, last, pooled, pred_logits
                    except Exception:
                        pass

            # final combined loss
            loss = lm_loss + lambda_v_current * cons_loss

            loss_to_back = loss / max(1, grad_accum_steps)
            if device_t.type == "cuda" and enable_amp:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            micro += 1
            running_loss += float(loss.detach().cpu())

            if (micro % grad_accum_steps) == 0:
                if device_t.type == "cuda" and enable_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            global_step += 1
            pbar.set_postfix({'loss': running_loss / max(1, micro), 'opt_steps': opt_steps, 'aux_w': lambda_v_current})
            try:
                del out_acc, lm_loss, cons_loss
            except Exception:
                pass
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        # flush leftover gradients if any
        if (micro % grad_accum_steps) != 0:
            if device_t.type == "cuda" and enable_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

        print(f"[resume] epoch {epoch} done, avg loss: {running_loss / max(1, micro):.4f}")

        # save adapters + head
        epoch_dir = os.path.join(out_dir, f"resume_epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        print("[resume] saving PEFT adapters to", epoch_dir)
        try:
            model.save_pretrained(epoch_dir)
        except Exception:
            torch.save({k:v.cpu() for k,v in model.state_dict().items()}, os.path.join(epoch_dir, "full_state.pt"))

        try:
            torch.save({'head_state': constraint_head.state_dict()}, os.path.join(epoch_dir, "constraint_head.pt"))
        except Exception:
            torch.save(constraint_head.state_dict(), os.path.join(epoch_dir, "constraint_head_fallback.pt"))

    print("[resume] training finished. latest adapters saved to", out_dir)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lora_dir', type=str, default="out_checkpoints/dpo_epoch1", help='Path to existing LoRA/adapter checkpoint dir (PEFT format)')
    p.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help='Name or path of base model')
    p.add_argument('--data', type=str, default="dataset_groq_fixed.jsonl", help='jsonl dataset (records with accepted and optional label)')
    p.add_argument('--out_dir', type=str, default='resumed_out', help='Where to save updated adapters')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--quant_bits', type=int, choices=[4,8,16], default=4)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lambda_v', type=float, default=0.7, help='Target aux loss weight (will be warmed up)')
    p.add_argument('--aux_warmup_epochs', type=int, default=0, help='Number of epochs to warm up auxiliary weight from 0 to lambda_v (0 => immediate full weight)')
    p.add_argument('--grad_ckpt', action='store_true', default=True)
    p.add_argument('--enable_amp', action='store_true', default=False)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    resume_train(
        lora_dir=args.lora_dir,
        base_model_name=args.base_model,
        data_path=args.data,
        out_dir=args.out_dir,
        device=args.device,
        quant_bits=args.quant_bits,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        lambda_v=args.lambda_v,
        aux_warmup_epochs=args.aux_warmup_epochs,
        gradient_checkpointing=args.grad_ckpt,
        enable_amp=args.enable_amp
    )
