from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from trainer.jsonl import read_jsonl


@dataclass(frozen=True)
class SFTArgs:
    model_name_or_path: str
    train_path: str
    output_dir: str
    eval_path: Optional[str] = None
    max_length: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 200
    per_device_batch_size: int = 1
    grad_accum: int = 16
    seed: int = 0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 10
    save_merged: bool = False
    fp16: bool = False                  # mixed-precision (fp16) — saves ~40% VRAM on T4
    bf16: bool = False                  # mixed-precision (bf16) — better numerics on A100/H100
    gradient_checkpointing: bool = False  # trade compute for memory; ~30% slower
    # Path to write per-step training metrics as JSONL (None = don't write)
    train_log_path: Optional[str] = None


class PromptTargetJsonl(Dataset):
    def __init__(self, path: str):
        self.rows = read_jsonl(path)
        for k in ("prompt", "target"):
            if any(k not in r for r in self.rows):
                raise ValueError(f"Expected key '{k}' in every jsonl row.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        return {
            "prompt": r["prompt"],
            "target": r["target"],
        }


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_lora_targets(model: torch.nn.Module) -> list[str]:
    candidate_groups: list[list[str]] = [
        # Qwen/LLaMA-like
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Some GPT-style models
        ["c_attn", "c_proj", "c_fc"],
    ]

    supported_types = (torch.nn.Linear, Conv1D)

    supported_module_names = [
        name for name, module in model.named_modules() if isinstance(module, supported_types)
    ]
    for candidates in candidate_groups:
        found = [c for c in candidates if any(n.endswith(c) for n in supported_module_names)]
        if found:
            return found

    raise RuntimeError(
        "Could not infer LoRA target modules. "
        "Update _infer_lora_targets() for this model architecture."
    )


def _collate(batch: list[dict[str, Any]], tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]
    full_texts = [p + t for p, t in zip(prompts, targets)]

    prompt_tok = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tok = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    input_ids = full_tok["input_ids"]
    attention_mask = full_tok["attention_mask"]
    labels = input_ids.clone()

    # Mask prompt tokens and padding tokens.
    for i, p_ids in enumerate(prompt_tok["input_ids"]):
        p_len = min(len(p_ids), labels.shape[1])
        labels[i, :p_len] = -100
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_sft(args: SFTArgs) -> None:
    _set_seed(args.seed)
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Choose dtype for model weights
    if args.bf16 and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    elif args.fp16:
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=load_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = _infer_lora_targets(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # AMP scaler — only used for fp16 (bf16 doesn't need it)
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    train_ds = PromptTargetJsonl(args.train_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer, args.max_length),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    opt_step = 0
    micro_step = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps, desc="sft")
    running_loss = 0.0
    running_count = 0

    while opt_step < args.max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss / args.grad_accum).backward()
            micro_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1

            if micro_step % args.grad_accum == 0:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, running_count)
                    tqdm.write(f"step={opt_step} avg_loss={avg_loss:.4f}")
                    if args.train_log_path:
                        with open(args.train_log_path, "a") as _lf:
                            _lf.write(json.dumps({"step": opt_step, "loss": round(avg_loss, 6)}) + "\n")
                    running_loss = 0.0
                    running_count = 0

                pbar.update(1)
                if opt_step >= args.max_steps:
                    break

    pbar.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter by default; optionally merge and save full model.
    if args.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir)
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

