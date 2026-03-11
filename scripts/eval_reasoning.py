"""
Step-by-step Reasoning Validity Evaluation.

Goes beyond final-answer accuracy: extracts every arithmetic expression
from model completions (e.g. "16 - 3 = 13") and verifies whether each
calculation is correct.

This script can run in two modes:
  (A) Analyze existing eval JSONL files (from teammate's eval.py)
  (B) Run live evaluation and then analyze

Metrics:
  - step_accuracy         : fraction of arithmetic steps that are correct
  - avg_steps             : average number of reasoning steps per completion
  - reasoning_valid_rate  : fraction where ALL extracted steps are correct
  - final_answer_accuracy : standard accuracy (for comparison)

Produces:
  - reasoning_summary.json   : aggregate metrics
  - reasoning_detailed.jsonl : per-example breakdown
  - reasoning_validity.png   : visualization

Usage (mode A — recommended, reuses existing eval results):
  python scripts/eval_reasoning.py \
    --results-path model/eval_sft.jsonl \
    --output-dir model/reasoning_sft

Usage (mode B — live evaluation):
  python scripts/eval_reasoning.py \
    --test-path dataset/processed/gsm8k_test.jsonl \
    --base-model Qwen/Qwen3-0.6B-Base \
    --adapter-path model/sft_gsm8k \
    --output-dir model/reasoning_sft \
    --max-examples 200
"""
from __future__ import annotations

import json
import os
import re
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

from trainer.jsonl import read_jsonl, write_jsonl  # noqa: E402
from trainer.reward import compute_gsm8k_reward  # noqa: E402

app = typer.Typer(add_completion=False)


# ── Arithmetic Step Extraction & Verification ─────────────────────────────────

# "16 - 3 = 13" or "5 * 4 = 20"
_ARITH_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*"
    r"([+\-\*/×÷])\s*"
    r"([\d,]+(?:\.\d+)?)\s*"
    r"=\s*"
    r"([\d,]+(?:\.\d+)?)"
)

# GSM8K annotation "<<16-3=13>>"
_ANNO_RE = re.compile(
    r"<<\s*([\d,]+(?:\.\d+)?)\s*"
    r"([+\-\*/])\s*"
    r"([\d,]+(?:\.\d+)?)\s*"
    r"=\s*"
    r"([\d,]+(?:\.\d+)?)\s*>>"
)


def _parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def _normalize_op(op: str) -> str:
    if op in ("×", "x", "X"):
        return "*"
    if op == "÷":
        return "/"
    return op


def _compute(left: float, op: str, right: float) -> Optional[float]:
    op = _normalize_op(op)
    if op == "+": return left + right
    if op == "-": return left - right
    if op == "*": return left * right
    if op == "/" and right != 0: return left / right
    return None


def extract_and_verify_steps(text: str) -> list[dict]:
    """Extract every arithmetic expression and verify correctness."""
    steps = []
    seen = set()
    for pattern in [_ANNO_RE, _ARITH_RE]:
        for m in pattern.finditer(text):
            raw = m.group(0)
            if raw in seen:
                continue
            seen.add(raw)
            try:
                left = _parse_num(m.group(1))
                op = m.group(2)
                right = _parse_num(m.group(3))
                claimed = _parse_num(m.group(4))
            except (ValueError, IndexError):
                continue
            actual = _compute(left, op, right)
            if actual is None:
                continue
            steps.append({
                "expression": raw,
                "left": left,
                "op": _normalize_op(op),
                "right": right,
                "claimed": claimed,
                "actual": round(actual, 6),
                "correct": abs(actual - claimed) < 0.01,
            })
    return steps


# ── Aggregate analysis ────────────────────────────────────────────────────────

def analyze_results(results: list[dict]) -> tuple[dict, list[dict]]:
    """
    Analyze reasoning validity across a set of eval results.
    Returns (summary_dict, per_example_details).
    """
    total_completions = 0
    completions_with_steps = 0
    total_steps = 0
    correct_steps = 0
    all_valid_count = 0
    final_correct_count = 0
    # Quadrant counts
    valid_and_correct = 0
    invalid_but_correct = 0
    valid_but_wrong = 0

    detailed = []

    for r in results:
        completion = r.get("completion", "")
        gold = r.get("gold", r.get("final_answer", ""))
        is_final_ok = r.get("correct", 0) > 0
        if not is_final_ok and gold:
            is_final_ok = compute_gsm8k_reward(completion, gold)["correct"] > 0

        steps = extract_and_verify_steps(completion)
        n_ok = sum(s["correct"] for s in steps)
        all_valid = all(s["correct"] for s in steps) if steps else False

        total_completions += 1
        total_steps += len(steps)
        correct_steps += n_ok
        if is_final_ok:
            final_correct_count += 1
        if steps:
            completions_with_steps += 1
            if all_valid:
                all_valid_count += 1

        if steps and all_valid and is_final_ok:
            valid_and_correct += 1
        elif steps and not all_valid and is_final_ok:
            invalid_but_correct += 1
        elif steps and all_valid and not is_final_ok:
            valid_but_wrong += 1

        wrong_steps = [s for s in steps if not s["correct"]]
        detailed.append({
            "question": r.get("question", "")[:120],
            "gold": gold,
            "final_correct": is_final_ok,
            "num_steps": len(steps),
            "correct_steps": n_ok,
            "all_valid": all_valid,
            "wrong_steps": [
                {"expr": s["expression"], "claimed": s["claimed"], "actual": s["actual"]}
                for s in wrong_steps
            ],
        })

    summary = {
        "total_completions": total_completions,
        "completions_with_arith_steps": completions_with_steps,
        "total_arithmetic_steps": total_steps,
        "correct_arithmetic_steps": correct_steps,
        "step_accuracy": round(correct_steps / total_steps, 4) if total_steps else 0.0,
        "avg_steps_per_completion": round(total_steps / completions_with_steps, 2) if completions_with_steps else 0.0,
        "reasoning_validity_rate": round(all_valid_count / completions_with_steps, 4) if completions_with_steps else 0.0,
        "final_answer_accuracy": round(final_correct_count / total_completions, 4) if total_completions else 0.0,
        "quadrant": {
            "valid_reasoning_correct_answer": valid_and_correct,
            "invalid_reasoning_correct_answer": invalid_but_correct,
            "valid_reasoning_wrong_answer": valid_but_wrong,
            "other": total_completions - valid_and_correct - invalid_but_correct - valid_but_wrong,
        },
    }
    return summary, detailed


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Step-by-Step Reasoning Validity", fontsize=14, fontweight="bold")

    ax = axes[0]
    labels = ["Step\nAccuracy", "Reasoning\nValidity", "Final Answer\nAccuracy"]
    vals = [
        summary["step_accuracy"] * 100,
        summary["reasoning_validity_rate"] * 100,
        summary["final_answer_accuracy"] * 100,
    ]
    colors = ["#FF9800", "#4CAF50", "#2196F3"]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Accuracy Breakdown")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    q = summary["quadrant"]
    cats = ["Valid + Correct", "Invalid + Correct", "Valid + Wrong", "Other"]
    qvals = [q["valid_reasoning_correct_answer"], q["invalid_reasoning_correct_answer"],
             q["valid_reasoning_wrong_answer"], q["other"]]
    colors2 = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]
    bars2 = ax.bar(cats, qvals, color=colors2, width=0.6, edgecolor="white")
    for bar, v in zip(bars2, qvals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Reasoning × Correctness")
    ax.set_xticklabels(cats, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Chart saved to {out_path}")


# ── Live eval helper ──────────────────────────────────────────────────────────

def _run_live_eval(
    test_path: str, base_model: str,
    adapter_path: Optional[str],
    max_new_tokens: int, max_examples: Optional[int],
) -> list[dict]:
    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if adapter_path and not (Path(adapter_path) / "adapter_config.json").exists():
        base_model = adapter_path
        adapter_path = None

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else base_model,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, dtype=torch.float16,
    )
    if adapter_path:
        model = model.to(device)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval().to(device)

    examples = read_jsonl(test_path)
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]

    results = []
    input_device = next(model.parameters()).device
    for ex in tqdm(examples, desc="generating"):
        enc = tokenizer(
            ex["prompt"], return_tensors="pt", truncation=True, max_length=512,
        ).to(input_device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        gold = ex["final_answer"]
        r = compute_gsm8k_reward(completion, gold)
        results.append({
            "question": ex["question"],
            "gold": gold,
            "completion": completion,
            "correct": r["correct"],
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    results_path: Optional[str] = typer.Option(
        None, help="Existing eval JSONL from eval.py (mode A — no model needed).",
    ),
    test_path: Optional[str] = typer.Option(
        None, help="Test JSONL for live evaluation (mode B).",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3-0.6B-Base", help="Base model (mode B only).",
    ),
    adapter_path: Optional[str] = typer.Option(
        None, help="LoRA adapter path (mode B only).",
    ),
    output_dir: str = typer.Option(
        "model/reasoning_analysis", help="Output directory.",
    ),
    max_new_tokens: int = typer.Option(256, help="Max generation tokens (mode B)."),
    max_examples: Optional[int] = typer.Option(None, help="Limit examples."),
) -> None:
    """Evaluate step-by-step reasoning validity in model completions."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if results_path:
        typer.echo(f"Loading existing results: {results_path}")
        results = read_jsonl(results_path)
        if max_examples:
            results = results[:max_examples]
    elif test_path:
        typer.echo("Running live evaluation...")
        results = _run_live_eval(test_path, base_model, adapter_path, max_new_tokens, max_examples)
    else:
        typer.echo("ERROR: provide either --results-path (mode A) or --test-path (mode B)")
        raise typer.Exit(1)

    typer.echo(f"Analyzing {len(results)} completions...\n")
    summary, detailed = analyze_results(results)

    typer.echo("=" * 55)
    typer.echo("Step-by-Step Reasoning Validity Analysis")
    typer.echo("=" * 55)
    typer.echo(f"  Completions analyzed       : {summary['total_completions']}")
    typer.echo(f"  Completions with arith     : {summary['completions_with_arith_steps']}")
    typer.echo(f"  Total arithmetic steps     : {summary['total_arithmetic_steps']}")
    typer.echo(f"  Correct arithmetic steps   : {summary['correct_arithmetic_steps']}")
    typer.echo(f"  Step accuracy              : {summary['step_accuracy']:.1%}")
    typer.echo(f"  Avg steps per completion   : {summary['avg_steps_per_completion']:.1f}")
    typer.echo(f"  Reasoning validity rate    : {summary['reasoning_validity_rate']:.1%}")
    typer.echo(f"  Final answer accuracy      : {summary['final_answer_accuracy']:.1%}")
    typer.echo("-" * 55)
    q = summary["quadrant"]
    typer.echo(f"  Valid reasoning + correct  : {q['valid_reasoning_correct_answer']}")
    typer.echo(f"  Invalid reasoning + correct: {q['invalid_reasoning_correct_answer']}")
    typer.echo(f"  Valid reasoning + wrong    : {q['valid_reasoning_wrong_answer']}")
    typer.echo(f"  Other (no steps / both bad): {q['other']}")
    typer.echo("=" * 55)

    with open(out_path / "reasoning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_jsonl(out_path / "reasoning_detailed.jsonl", detailed)

    wrong_examples = [x for x in detailed if x["wrong_steps"]]
    if wrong_examples:
        typer.echo(f"\n─── Arithmetic errors (first 5) ───")
        for ex in wrong_examples[:5]:
            typer.echo(f"\nQ: {ex['question']}...")
            for ws in ex["wrong_steps"][:2]:
                typer.echo(f"  Error: {ws['expr']}  → actual={ws['actual']}")

    _plot(summary, out_path / "reasoning_validity.png")
    typer.echo(f"\nAll outputs saved to {out_path}/")


if __name__ == "__main__":
    app()
