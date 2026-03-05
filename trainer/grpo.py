"""
Group Relative Policy Optimization (GRPO) trainer for GSM8K.

Algorithm per optimizer step
─────────────────────────────
For each prompt in the micro-batch:
  1. Generate G completions from the current policy  (no gradient).
  2. Score each completion with compute_gsm8k_reward.
  3. Normalize rewards within the group → advantages:
       A_g = (r_g − mean_r) / (std_r + ε)
  4. Forward pass through policy  (with gradient) to get per-token log-probs.
  5. Forward pass through frozen reference  (no gradient).
  6. Per-prompt loss:
       L_pg  = −(1/G) Σ_g  A_g · (1/|C_g|) Σ_t  log π_θ(t)
       L_kl  =  (1/G) Σ_g  (1/|C_g|) Σ_t  (log π_θ(t) − log π_ref(t))
       L     =  L_pg + kl_coef · L_kl
  7. Accumulate gradients over batch_prompts × grad_accum prompts, then update.

Reference: "DeepSeek-Math: Pushing the Limits of Mathematical Reasoning in
Open Language Models", Shao et al., 2024. https://arxiv.org/abs/2402.03300
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.jsonl import read_jsonl
from trainer.reward import compute_gsm8k_reward
from trainer.sft import _get_device, _infer_lora_targets, _set_seed


@dataclass(frozen=True)
class GRPOArgs:
    model_name_or_path: str
    train_path: str
    output_dir: str
    # Optional SFT checkpoint used as (a) the starting policy and (b) the reference model.
    # If None the raw base model is used for both.
    ref_model_path: Optional[str] = None
    # Generation
    max_prompt_length: int = 256
    max_new_tokens: int = 256
    group_size: int = 8           # G – completions sampled per prompt
    temperature: float = 0.9
    # Loss
    kl_coef: float = 0.04         # β – weight for KL penalty
    # Optimisation
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_steps: int = 200
    batch_prompts: int = 1        # prompts processed per micro-step
    grad_accum: int = 1           # micro-steps per optimizer update
    grad_clip: float = 1.0
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Misc
    seed: int = 0
    log_every: int = 10
    save_merged: bool = False
    # Path to write per-step training metrics as JSONL (None = don't write)
    train_log_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_base(model_name_or_path: str, ref_model_path: Optional[str]) -> torch.nn.Module:
    """
    Load base weights, optionally merge an SFT LoRA adapter.
    Returns a plain (non-PEFT) model.
    """
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    if ref_model_path:
        base = PeftModel.from_pretrained(base, ref_model_path)
        base = base.merge_and_unload()
    return base


def _per_token_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,   # [1, total_len]
    prompt_len: int,
) -> torch.Tensor:
    """
    Returns per-token log-probs for the completion part of input_ids.
    Shape: [comp_len]

    The completion starts at position prompt_len in input_ids.
    Using the standard shifted-logits trick:
      logits[i] predicts token[i+1], so
      the log-prob of completion token k  (absolute position prompt_len + k)
      lives at shifted index (prompt_len + k) − 1 = (prompt_len − 1) + k.
    """
    logits = model(input_ids=input_ids).logits[0]           # [total_len, vocab]
    shift_logits = logits[prompt_len - 1 : -1]              # [comp_len, vocab]
    shift_ids    = input_ids[0, prompt_len:]                 # [comp_len]
    log_probs    = F.log_softmax(shift_logits, dim=-1)       # [comp_len, vocab]
    token_lp     = log_probs.gather(1, shift_ids.unsqueeze(1)).squeeze(1)  # [comp_len]
    return token_lp



def _grpo_loss_for_prompt_with_gold(
    policy: torch.nn.Module,
    ref_model: torch.nn.Module,
    prompt_ids: torch.Tensor,   # [1, P]
    prompt_len: int,
    gold_answer: str,
    tokenizer,
    args: GRPOArgs,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], dict]:
    """
    Full GRPO forward for one prompt.
    Returns (loss_tensor or None if no gradient signal, info_dict).
    """
    # ── Generate completions ─────────────────────────────────────────────────
    policy.eval()
    comp_ids_list: list[torch.Tensor] = []
    comp_texts: list[str] = []
    rewards: list[float] = []

    with torch.no_grad():
        for _ in range(args.group_size):
            out = policy.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            comp = out[:, prompt_len:]          # [1, C_g]
            text = tokenizer.decode(comp[0], skip_special_tokens=True)
            r    = compute_gsm8k_reward(text, gold_answer)["reward"]
            comp_ids_list.append(comp)
            comp_texts.append(text)
            rewards.append(r)

    # ── Compute advantages ───────────────────────────────────────────────────
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r    = rewards_t.mean().item()
    std_r     = rewards_t.std().item()

    if std_r < 1e-8:
        # All rewards identical → no learning signal for this prompt
        return None, {
            "mean_reward": mean_r,
            "pg_loss": 0.0,
            "kl": 0.0,
            "n_correct": int(sum(r >= 1.0 for r in rewards)),
        }

    advantages = (rewards_t - mean_r) / (std_r + 1e-8)  # [G]

    # ── Compute policy + KL losses ───────────────────────────────────────────
    policy.train()

    pg_sum  = torch.tensor(0.0, device=device, requires_grad=False)
    kl_sum  = torch.tensor(0.0, device=device, requires_grad=False)
    pg_loss_tensor  = None
    kl_loss_tensor  = None
    valid = 0

    for g, (comp, adv) in enumerate(zip(comp_ids_list, advantages)):
        comp_len = comp.shape[1]
        if comp_len == 0:
            continue

        full_ids = torch.cat([prompt_ids, comp], dim=1)  # [1, P+C]

        # Policy log-probs (gradient flows here)
        policy_lp = _per_token_log_probs(policy, full_ids, prompt_len)   # [C]

        # Reference log-probs (no gradient)
        with torch.no_grad():
            ref_lp = _per_token_log_probs(ref_model, full_ids, prompt_len)  # [C]

        # Per-completion contributions (mean over tokens)
        adv_dev = adv.to(device)
        pg_g  = -adv_dev * policy_lp.mean()
        kl_g  = (policy_lp - ref_lp).mean()         # ≈ per-token KL

        pg_loss_tensor = pg_g  if pg_loss_tensor is None else pg_loss_tensor + pg_g
        kl_loss_tensor = kl_g  if kl_loss_tensor is None else kl_loss_tensor + kl_g
        valid += 1

    if valid == 0 or pg_loss_tensor is None:
        return None, {"mean_reward": mean_r, "pg_loss": 0.0, "kl": 0.0,
                      "n_correct": int(sum(r >= 1.0 for r in rewards))}

    pg_loss_tensor = pg_loss_tensor / valid
    kl_loss_tensor = kl_loss_tensor / valid

    loss = pg_loss_tensor + args.kl_coef * kl_loss_tensor

    return loss, {
        "mean_reward": mean_r,
        "pg_loss":     float(pg_loss_tensor.detach()),
        "kl":          float(kl_loss_tensor.detach()),
        "n_correct":   int(sum(r >= 1.0 for r in rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_grpo(args: GRPOArgs) -> None:
    _set_seed(args.seed)
    device = _get_device()

    # Create output dir early so log file can be written during training
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.train_log_path:
        Path(args.train_log_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = args.ref_model_path if args.ref_model_path else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Reference model (frozen) ──────────────────────────────────────────────
    ref_model = _load_base(args.model_name_or_path, args.ref_model_path)
    ref_model.eval()
    ref_model.to(device)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Policy model (trainable LoRA on top of same base) ─────────────────────
    policy_base = _load_base(args.model_name_or_path, args.ref_model_path)
    policy_base.config.pad_token_id = tokenizer.pad_token_id

    target_modules = _infer_lora_targets(policy_base)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    policy = get_peft_model(policy_base, lora_cfg)
    policy.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data = read_jsonl(args.train_path)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    opt.zero_grad(set_to_none=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    opt_step   = 0
    micro_step = 0
    pbar = tqdm(total=args.max_steps, desc="grpo")

    # Running statistics for logging
    log_reward = 0.0
    log_pg     = 0.0
    log_kl     = 0.0
    log_correct = 0
    log_prompts = 0

    while opt_step < args.max_steps:
        # Sample a micro-batch of prompts
        rows = random.sample(train_data, min(args.batch_prompts, len(train_data)))

        batch_has_grad = False

        for row in rows:
            prompt    = row["prompt"]
            gold      = row["final_answer"]

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            )
            prompt_ids = enc["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]

            loss, info = _grpo_loss_for_prompt_with_gold(
                policy, ref_model,
                prompt_ids, prompt_len,
                gold, tokenizer, args, device,
            )

            # Accumulate logging stats
            log_reward  += info["mean_reward"]
            log_pg      += info["pg_loss"]
            log_kl      += info["kl"]
            log_correct += info["n_correct"]
            log_prompts += 1

            if loss is not None:
                (loss / (args.batch_prompts * args.grad_accum)).backward()
                batch_has_grad = True

        if batch_has_grad:
            micro_step += 1

        if micro_step > 0 and micro_step % args.grad_accum == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            pbar.update(1)

            if opt_step % args.log_every == 0 and log_prompts > 0:
                mean_reward = log_reward / log_prompts
                mean_pg     = log_pg / log_prompts
                mean_kl     = log_kl / log_prompts
                accuracy    = log_correct / (log_prompts * args.group_size)
                tqdm.write(
                    f"step={opt_step}"
                    f"  reward={mean_reward:.3f}"
                    f"  pg={mean_pg:.4f}"
                    f"  kl={mean_kl:.4f}"
                    f"  correct={log_correct}/{log_prompts * args.group_size}"
                )
                if args.train_log_path:
                    _log_step = {
                        "step":     opt_step,
                        "reward":   round(mean_reward, 6),
                        "pg_loss":  round(mean_pg, 6),
                        "kl":       round(mean_kl, 6),
                        "accuracy": round(accuracy, 6),
                    }
                    with open(args.train_log_path, "a") as _lf:
                        _lf.write(json.dumps(_log_step) + "\n")
                log_reward = log_pg = log_kl = 0.0
                log_correct = log_prompts = 0

            if opt_step >= args.max_steps:
                break

        elif not batch_has_grad:
            # No prompt in this batch produced a gradient (all rewards tied);
            # increment micro_step anyway to avoid stalling.
            micro_step += 1
            opt.zero_grad(set_to_none=True)

    pbar.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_merged:
        merged = policy.merge_and_unload()
        merged.save_pretrained(out_dir)
    else:
        policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    tqdm.write(f"Saved GRPO adapter to {out_dir}")
