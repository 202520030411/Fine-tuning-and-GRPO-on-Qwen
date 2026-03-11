"""
DoRA (Weight-Decomposed Low-Rank Adaptation) training script.

Identical pipeline to train_sft.py but uses DoRA instead of LoRA.
DoRA decomposes weight updates into magnitude and direction components,
which can improve training stability compared to standard LoRA.

This is a standalone script — it does NOT modify any existing files.
It reuses helper utilities from trainer/ but has its own training loop.

Usage (same as train_sft.py, same hyperparameters for fair comparison):
  python scripts/train_dora.py \
    --model-name-or-path Qwen/Qwen3-0.6B-Base \
    --train-path dataset/processed/gsm8k_train.jsonl \
    --output-dir model/dora_gsm8k \
    --train-log-path model/dora_gsm8k/train_log.jsonl \
    --max-steps 300 \
    --per-device-batch-size 2 \
    --grad-accum 16 \
    --max-length 384 \
    --fp16 \
    --gradient-checkpointing \
    --log-every 10

Evaluate with the same eval.py:
  python scripts/eval.py \
    --base-model Qwen/Qwen3-0.6B-Base \
    --adapter-path model/dora_gsm8k \
    --test-path dataset/processed/gsm8k_test.jsonl \
    --output-path model/eval_dora.jsonl \
    --max-new-tokens 256 --batch-size 8 --max-examples 500
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

hf_home = PROJECT_ROOT / ".hf"
os.environ.setdefault("HF_HOME", str(hf_home))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Reuse teammate's data loading and collate utilities (read-only imports)
from trainer.sft import PromptTargetJsonl, _collate, _infer_lora_targets, _set_seed, _get_device  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name_or_path: str = typer.Option("Qwen/Qwen3-0.6B-Base", help="HF model id or local path."),
    train_path: str = typer.Option(..., help="Path to train JSONL (needs keys: prompt, target)."),
    output_dir: str = typer.Option("model/dora_gsm8k", help="Where to save the DoRA adapters."),
    max_length: int = typer.Option(512, help="Max tokens for prompt+target."),
    learning_rate: float = typer.Option(2e-4, help="AdamW LR."),
    max_steps: int = typer.Option(200, help="Number of optimizer steps."),
    per_device_batch_size: int = typer.Option(1, help="Microbatch size."),
    grad_accum: int = typer.Option(16, help="Gradient accumulation steps."),
    lora_r: int = typer.Option(8, help="LoRA/DoRA rank."),
    lora_alpha: int = typer.Option(16, help="LoRA/DoRA alpha."),
    lora_dropout: float = typer.Option(0.05, help="Dropout."),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping norm."),
    log_every: int = typer.Option(10, help="Log avg loss every N steps."),
    seed: int = typer.Option(0, help="Random seed."),
    save_merged: bool = typer.Option(False, help="Merge DoRA and save full model."),
    train_log_path: Optional[str] = typer.Option(None, help="JSONL file to write per-step loss."),
    fp16: bool = typer.Option(False, help="Enable fp16 mixed precision."),
    bf16: bool = typer.Option(False, help="Enable bf16 mixed precision."),
    gradient_checkpointing: bool = typer.Option(False, help="Gradient checkpointing."),
) -> None:
    _set_seed(seed)
    device = _get_device()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if train_log_path:
        Path(train_log_path).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if bf16 and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    elif fp16:
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=load_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = _infer_lora_targets(model)

    # ── The only difference from train_sft.py: use_dora=True ──────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_dora=True,  # <-- DoRA: weight-decomposed low-rank adaptation
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    typer.echo("DoRA config applied (use_dora=True)")
    model.print_trainable_parameters()

    use_amp = (fp16 or bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    train_ds = PromptTargetJsonl(train_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer, max_length),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    opt_step = 0
    micro_step = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc="dora")
    running_loss = 0.0
    running_count = 0

    while opt_step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if int((batch["labels"] != -100).sum().item()) == 0:
                continue
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1

            if micro_step % grad_accum == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % log_every == 0:
                    avg_loss = running_loss / max(1, running_count)
                    tqdm.write(f"step={opt_step} avg_loss={avg_loss:.4f}")
                    if train_log_path:
                        with open(train_log_path, "a") as lf:
                            lf.write(json.dumps({"step": opt_step, "loss": round(avg_loss, 6)}) + "\n")
                    running_loss = 0.0
                    running_count = 0

                pbar.update(1)
                if opt_step >= max_steps:
                    break

    pbar.close()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir)
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    typer.echo(f"DoRA adapter saved to {out_dir}")


if __name__ == "__main__":
    app()
