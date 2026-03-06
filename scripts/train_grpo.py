"""GRPO training using TRL's GRPOTrainer."""
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

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from trainer.jsonl import read_jsonl  # noqa: E402
from trainer.reward import compute_gsm8k_reward  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name_or_path: str = typer.Option("Qwen/Qwen3-0.6B-Base", help="Base model."),
    ref_model_path: Optional[str] = typer.Option(None, help="SFT adapter to start from."),
    train_path: str = typer.Option(..., help="Train JSONL (prompt, final_answer)."),
    output_dir: str = typer.Option("model/grpo_gsm8k", help="Save directory."),
    train_log_path: Optional[str] = typer.Option(None, help="Unused (TRL logs internally)."),
    max_steps: int = typer.Option(200, help="Optimizer steps."),
    group_size: int = typer.Option(8, help="Completions per prompt (G)."),
    max_completion_length: int = typer.Option(256, help="Max tokens generated (GRPOConfig.max_completion_length)."),
    learning_rate: float = typer.Option(1e-5, help="Learning rate."),
    kl_coef: float = typer.Option(0.01, help="KL penalty (beta)."),
    lora_r: int = typer.Option(8, help="LoRA rank."),
    lora_alpha: int = typer.Option(16, help="LoRA alpha."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base model in fp16 ───────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # ── Merge SFT adapter if provided ─────────────────────────────────────
    if ref_model_path:
        model = PeftModel.from_pretrained(model, ref_model_path)
        model = model.merge_and_unload()
        typer.echo(f"Merged SFT adapter from {ref_model_path}")

    # ── Add fresh LoRA for GRPO ───────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_path = ref_model_path if ref_model_path else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Dataset ───────────────────────────────────────────────────────────
    raw = read_jsonl(train_path)
    dataset = Dataset.from_list([
        {"prompt": r["prompt"], "final_answer": r["final_answer"]}
        for r in raw
    ])
    typer.echo(f"Loaded {len(dataset)} training examples")

    # ── Reward function ───────────────────────────────────────────────────
    def reward_fn(completions, final_answer, **kwargs):
        return [
            compute_gsm8k_reward(c, a)["reward"]
            for c, a in zip(completions, final_answer)
        ]

    # ── TRL GRPO config ───────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        fp16=True,
        num_generations=group_size,
        max_completion_length=max_completion_length,
        beta=kl_coef,
        seed=seed,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Saved GRPO model to {output_dir}")


if __name__ == "__main__":
    app()
