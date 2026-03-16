"""
Prompt Design Comparison Experiment.

Tests three prompt strategies on the SAME model and dataset, measuring
how prompt engineering affects math reasoning accuracy.

Strategies:
  1. direct — bare question, no guidance
  2. cot    — "Let's think step by step" (teammate's default prompt)
  3. rules  — explicit math operation rules + step-by-step instruction

This script reuses the teammate's model loading pattern from eval.py
and reward function from trainer/reward.py. No existing files are modified.

Usage:
  # Test on base model
  python scripts/eval_prompts.py \
    --test-path dataset/processed/gsm8k_test.jsonl \
    --base-model Qwen/Qwen3-0.6B-Base \
    --output-dir model/prompt_comparison_base \
    --max-examples 200

  # Test on SFT model
  python scripts/eval_prompts.py \
    --test-path dataset/processed/gsm8k_test.jsonl \
    --base-model Qwen/Qwen3-0.6B-Base \
    --adapter-path model/sft_gsm8k \
    --output-dir model/prompt_comparison_sft \
    --max-examples 200
"""
from __future__ import annotations

import json
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
from peft import PeftModel  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from trainer.jsonl import read_jsonl, write_jsonl  # noqa: E402
from trainer.reward import compute_gsm8k_reward  # noqa: E402

app = typer.Typer(add_completion=False)


# ── Three Prompt Strategies ───────────────────────────────────────────────────
#
# "direct" — just the question
# "cot"    — matches teammate's default in dataset/gsm8k.py
# "rules"  — adds explicit math operation definitions
#
PROMPTS = {
    "direct": (
        "Question: {question}\n"
        "Answer:"
    ),
    "cot": (
        "You are a helpful assistant that solves grade-school math word problems.\n\n"
        "Question: {question}\n"
        "Answer: Let's think step by step."
    ),
    "rules": (
        "You are a math tutor. Follow these rules:\n"
        "- Addition (+): combining quantities\n"
        "- Subtraction (-): finding the difference\n"
        "- Multiplication (*): repeated addition or scaling\n"
        "- Division (/): splitting into equal parts\n"
        "Show every calculation step clearly. "
        "End with #### followed by the final number.\n\n"
        "Question: {question}\n"
        "Answer: Let's solve this step by step."
    ),
}


# ── Model loading (same pattern as teammate's eval.py) ────────────────────────

def _load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str],
    device: torch.device,
):
    if adapter_path and not (Path(adapter_path) / "adapter_config.json").exists():
        base_model = adapter_path
        adapter_path = None

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    multi_gpu = torch.cuda.device_count() > 1 and adapter_path is None
    load_kwargs: dict = {"trust_remote_code": True, "dtype": torch.float16}
    if multi_gpu:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if adapter_path:
        model = model.to(device)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    if not multi_gpu:
        model.to(device)
    return model, tokenizer


# ── Evaluation per prompt strategy ────────────────────────────────────────────

def _evaluate_strategy(
    model, tokenizer, examples: list[dict],
    prompt_template: str, strategy_name: str,
    max_new_tokens: int,
) -> tuple[dict, list[dict]]:
    results = []
    n_correct = 0
    n_format = 0
    input_device = next(model.parameters()).device

    for ex in tqdm(examples, desc=f"  {strategy_name:<8}"):
        prompt = prompt_template.format(question=ex["question"])
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(input_device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        completion = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
        gold = ex["final_answer"]
        r = compute_gsm8k_reward(completion, gold)
        n_correct += int(r["correct"])
        n_format += int(r["format"] > 0)
        results.append({
            "question": ex["question"],
            "gold": gold,
            "prompt_strategy": strategy_name,
            "completion": completion,
            "correct": r["correct"],
            "format": r["format"],
            "reward": r["reward"],
        })

    total = len(results)
    return {
        "strategy": strategy_name,
        "accuracy": n_correct / total if total else 0.0,
        "format_rate": n_format / total if total else 0.0,
        "correct": n_correct,
        "total": total,
    }, results


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_comparison(metrics: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [m["strategy"] for m in metrics]
    accs = [m["accuracy"] * 100 for m in metrics]
    colors = ["#2196F3", "#FF9800", "#4CAF50"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold",
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Prompt Strategy Comparison on GSM8K", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    labels = {
        "direct": "Direct\n(no guidance)",
        "cot": "CoT\n(step by step)",
        "rules": "Math Rules\n(+, -, *, /)",
    }
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"Chart saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    test_path: str = typer.Option(
        "dataset/processed/gsm8k_test.jsonl",
        help="Path to test JSONL (needs keys: question, final_answer).",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3-0.6B-Base", help="Base model.",
    ),
    adapter_path: Optional[str] = typer.Option(
        None, help="LoRA adapter path (optional).",
    ),
    output_dir: str = typer.Option(
        "model/prompt_comparison", help="Output directory for results.",
    ),
    max_new_tokens: int = typer.Option(256, help="Max tokens to generate."),
    max_examples: Optional[int] = typer.Option(200, help="Limit number of examples."),
    strategies: str = typer.Option(
        "direct,cot,rules",
        help="Comma-separated prompt strategies to test.",
    ),
) -> None:
    """Compare prompt strategies on the same model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading model: {base_model}" + (f" + {adapter_path}" if adapter_path else ""))
    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_path, device)

    examples = read_jsonl(test_path)
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
    typer.echo(f"Evaluating {len(examples)} examples\n")

    strategy_list = [s.strip() for s in strategies.split(",")]
    all_metrics = []

    for name in strategy_list:
        if name not in PROMPTS:
            typer.echo(f"Unknown strategy '{name}', skipping")
            continue
        metrics, results = _evaluate_strategy(
            model, tokenizer, examples, PROMPTS[name], name, max_new_tokens,
        )
        all_metrics.append(metrics)
        write_jsonl(out_path / f"results_{name}.jsonl", results)

    typer.echo(f"\n{'='*55}")
    typer.echo(f"{'Strategy':<10} {'Accuracy':>10} {'Format%':>10} {'N':>6}")
    typer.echo(f"{'-'*55}")
    for m in all_metrics:
        typer.echo(
            f"{m['strategy']:<10} {m['accuracy']:>9.1%} "
            f"{m['format_rate']:>9.1%} {m['total']:>6}"
        )
    typer.echo(f"{'='*55}")

    summary_path = out_path / "prompt_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    typer.echo(f"\nSummary: {summary_path}")

    _plot_comparison(all_metrics, out_path / "prompt_comparison.png")


if __name__ == "__main__":
    app()
