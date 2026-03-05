from __future__ import annotations

from pathlib import Path
from typing import Optional

import os
import sys

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep HF caches inside the repo so this works in restricted environments.
hf_home = PROJECT_ROOT / ".hf"
os.environ.setdefault("HF_HOME", str(hf_home))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from trainer.sft import SFTArgs, run_sft  # noqa: E402


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name_or_path: str = typer.Option("Qwen/Qwen3-0.6B-Base", help="HF model id or local path."),
    train_path: str = typer.Option(..., help="Path to train jsonl (needs keys: prompt, target)."),
    output_dir: str = typer.Option("model/sft_gsm8k", help="Where to save the fine-tuned model/adapters."),
    eval_path: Optional[str] = typer.Option(None, help="(Optional) eval jsonl (currently unused)."),
    max_length: int = typer.Option(512, help="Max tokens for prompt+target."),
    learning_rate: float = typer.Option(2e-4, help="AdamW LR."),
    max_steps: int = typer.Option(200, help="Number of optimizer steps."),
    per_device_batch_size: int = typer.Option(1, help="Microbatch size."),
    grad_accum: int = typer.Option(16, help="Gradient accumulation steps."),
    lora_r: int = typer.Option(8, help="LoRA rank."),
    lora_alpha: int = typer.Option(16, help="LoRA alpha."),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout."),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping norm (0 disables)."),
    log_every: int = typer.Option(10, help="Log avg loss every N optimizer steps."),
    seed: int = typer.Option(0, help="Random seed."),
    save_merged: bool = typer.Option(False, help="If set, merge LoRA and save a full model."),
    train_log_path: Optional[str] = typer.Option(None, help="JSONL file to write per-step loss metrics."),
) -> None:
    args = SFTArgs(
        model_name_or_path=model_name_or_path,
        train_path=train_path,
        eval_path=eval_path,
        output_dir=output_dir,
        max_length=max_length,
        learning_rate=learning_rate,
        max_steps=max_steps,
        per_device_batch_size=per_device_batch_size,
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
    run_sft(args)


if __name__ == "__main__":
    app()

