"""Evaluate a model on a subset of MMLU (multiple-choice)."""
from __future__ import annotations

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
# HF_HUB_OFFLINE intentionally NOT set — MMLU subjects are downloaded from HF at eval time

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from dataset.mmlu import SUBJECTS, load_mmlu_subject, preprocess_mmlu_example  # noqa: E402
from trainer.jsonl import write_jsonl  # noqa: E402

app = typer.Typer(add_completion=False)

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_letter(text: str) -> str:
    """Extract the first A/B/C/D letter from model output."""
    text = text.strip()
    # "Answer: A" or starts with a letter
    m = re.match(r"^([ABCD])[^a-zA-Z]", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|Answer|ANSWER)\s*[:\-]?\s*([ABCD])\b", text)
    if m:
        return m.group(1)
    m = _LETTER_RE.search(text)
    if m:
        return m.group(1)
    return ""


def _load_model_and_tokenizer(base_model: str, adapter_path: Optional[str], device: torch.device):
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


@app.command()
def main(
    base_model: str = typer.Option("Qwen/Qwen3-0.6B-Base", help="Base model path or HF id."),
    adapter_path: Optional[str] = typer.Option(None, help="LoRA adapter directory (optional)."),
    output_path: str = typer.Option(..., help="Path to write per-example JSONL results."),
    subjects: str = typer.Option(
        ",".join(SUBJECTS),
        help="Comma-separated MMLU subjects to evaluate.",
    ),
    max_new_tokens: int = typer.Option(16, help="Max tokens to generate (short for MCQ)."),
    batch_size: int = typer.Option(8, help="Generation batch size."),
    max_examples_per_subject: Optional[int] = typer.Option(
        150, help="Cap examples per subject (None = all)."
    ),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    typer.echo(f"Device: {device}  ({n_gpus} GPU(s))")

    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    typer.echo(f"Subjects: {subject_list}")

    typer.echo(f"Loading model: {base_model}" + (f" + {adapter_path}" if adapter_path else ""))
    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_path, device)

    all_results = []
    subject_stats: dict[str, dict] = {}

    for subject in subject_list:
        typer.echo(f"\n── {subject} ──")
        try:
            ds = load_mmlu_subject(subject, split="test")
        except Exception as e:
            typer.echo(f"  WARNING: could not load {subject}: {e}")
            continue

        examples = [
            preprocess_mmlu_example(subject, ex["question"], ex["choices"], ex["answer"])
            for ex in ds
        ]
        if max_examples_per_subject and len(examples) > max_examples_per_subject:
            examples = examples[:max_examples_per_subject]

        n_correct = 0
        input_device = next(model.parameters()).device

        for i in tqdm(range(0, len(examples), batch_size), desc=subject[:30]):
            batch = examples[i : i + batch_size]
            prompts = [ex.prompt for ex in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
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
            completions = tokenizer.batch_decode(
                out_ids[:, prompt_len:], skip_special_tokens=True
            )

            for ex, completion in zip(batch, completions):
                predicted = extract_letter(completion)
                correct = int(predicted == ex.correct_letter)
                n_correct += correct
                all_results.append({
                    "subject":        ex.subject,
                    "question":       ex.question,
                    "choices":        ex.choices,
                    "correct_letter": ex.correct_letter,
                    "predicted":      predicted,
                    "completion":     completion,
                    "correct":        correct,
                })

        acc = n_correct / len(examples) if examples else 0.0
        subject_stats[subject] = {"n": len(examples), "correct": n_correct, "accuracy": acc}
        typer.echo(f"  {subject}: {n_correct}/{len(examples)} = {acc:.1%}")

    # Summary
    total = sum(s["n"] for s in subject_stats.values())
    total_correct = sum(s["correct"] for s in subject_stats.values())
    overall = total_correct / total if total else 0.0
    typer.echo(f"\n{'='*50}")
    typer.echo(f"  Overall MMLU accuracy: {total_correct}/{total} = {overall:.1%}")
    typer.echo(f"{'='*50}")

    write_jsonl(output_path, all_results)
    typer.echo(f"Results written to {output_path}")


if __name__ == "__main__":
    app()
