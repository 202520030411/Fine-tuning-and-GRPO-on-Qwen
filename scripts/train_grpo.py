from __future__ import annotations

import os
import sys
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

from trainer.grpo import GRPOArgs, run_grpo  # noqa: E402


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name_or_path: str = typer.Option(
        "Qwen/Qwen3-0.6B-Base",
        help="Base HF model id or local path.",
    ),
    train_path: str = typer.Option(
        ...,
        help="Path to train JSONL (needs keys: prompt, final_answer).",
    ),
    output_dir: str = typer.Option(
        "model/grpo_gsm8k",
        help="Directory to save the GRPO LoRA adapter.",
    ),
    ref_model_path: Optional[str] = typer.Option(
        None,
        help="SFT adapter path to use as reference + policy init. "
             "If omitted, the raw base model is used.",
    ),
    max_prompt_length: int = typer.Option(256, help="Truncate prompts to this many tokens."),
    max_new_tokens: int = typer.Option(256, help="Max tokens generated per completion."),
    group_size: int = typer.Option(8, help="Number of completions sampled per prompt (G)."),
    temperature: float = typer.Option(0.9, help="Sampling temperature for generation."),
    kl_coef: float = typer.Option(0.04, help="KL penalty coefficient (β)."),
    learning_rate: float = typer.Option(1e-5, help="AdamW learning rate."),
    max_steps: int = typer.Option(200, help="Total optimizer steps."),
    batch_prompts: int = typer.Option(1, help="Prompts processed per micro-step."),
    grad_accum: int = typer.Option(1, help="Micro-steps per optimizer update."),
    lora_r: int = typer.Option(8, help="LoRA rank."),
    lora_alpha: int = typer.Option(16, help="LoRA alpha."),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout."),
    grad_clip: float = typer.Option(1.0, help="Gradient norm clip (0 to disable)."),
    log_every: int = typer.Option(10, help="Log metrics every N optimizer steps."),
    seed: int = typer.Option(0, help="Random seed."),
    save_merged: bool = typer.Option(False, help="Merge LoRA into base and save full model."),
    train_log_path: Optional[str] = typer.Option(None, help="JSONL file to write per-step reward/loss metrics."),
) -> None:
    args = GRPOArgs(
        model_name_or_path=model_name_or_path,
        train_path=train_path,
        output_dir=output_dir,
        ref_model_path=ref_model_path,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        group_size=group_size,
        temperature=temperature,
        kl_coef=kl_coef,
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_prompts=batch_prompts,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        grad_clip=grad_clip,
        log_every=log_every,
        seed=seed,
        save_merged=save_merged,
        train_log_path=train_log_path,
    )
    run_grpo(args)


if __name__ == "__main__":
    app()
