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

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from trainer.jsonl import read_jsonl, write_jsonl  # noqa: E402
from trainer.reward import compute_gsm8k_reward  # noqa: E402


app = typer.Typer(add_completion=False)


def _load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str],
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # device_map="auto" spreads layers across GPUs but breaks merge_and_unload()
    # so only use it for the base model (no adapter). With an adapter, load on
    # a single GPU, merge, then the merged model stays on that GPU.
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
    test_path: str = typer.Option(
        "dataset/processed/gsm8k_test.jsonl",
        help="Path to test JSONL (needs keys: prompt, final_answer).",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3-0.6B-Base",
        help="HF model id or local path for the base weights.",
    ),
    adapter_path: Optional[str] = typer.Option(
        None,
        help="Path to a saved LoRA adapter directory. If omitted, evaluates the base model.",
    ),
    output_path: Optional[str] = typer.Option(
        None,
        help="If set, write per-example results to this JSONL file.",
    ),
    max_new_tokens: int = typer.Option(256, help="Max tokens to generate per example."),
    temperature: float = typer.Option(0.0, help="Sampling temperature (0 = greedy)."),
    max_examples: Optional[int] = typer.Option(
        None,
        help="Evaluate only the first N examples (useful for quick checks).",
    ),
    batch_size: int = typer.Option(1, help="Generation batch size."),
) -> None:
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Device: {device}  ({n_gpus} GPU(s) detected)")

    typer.echo(f"Loading model: {base_model}" + (f" + adapter: {adapter_path}" if adapter_path else ""))
    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_path, device)

    examples = read_jsonl(test_path)
    if max_examples:
        examples = examples[:max_examples]
    typer.echo(f"Evaluating {len(examples)} examples from {test_path}")

    results = []
    n_correct = 0
    n_format = 0

    for i in tqdm(range(0, len(examples), batch_size), desc="eval"):
        batch = examples[i : i + batch_size]
        prompts = [ex["prompt"] for ex in batch]

        # With device_map="auto" inputs go to the model's first device
        input_device = next(model.parameters()).device
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(input_device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)

        # Decode only the newly generated tokens (skip the prompt)
        prompt_len = enc["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            out_ids[:, prompt_len:],
            skip_special_tokens=True,
        )

        for ex, completion in zip(batch, completions):
            gold = ex["final_answer"]
            r = compute_gsm8k_reward(completion, gold)
            n_correct += int(r["correct"])
            n_format += int(r["format"] > 0)
            results.append(
                {
                    "question": ex["question"],
                    "gold": gold,
                    "completion": completion,
                    "correct": r["correct"],
                    "format": r["format"],
                    "reward": r["reward"],
                }
            )

    total = len(results)
    accuracy = n_correct / total if total else 0.0
    format_rate = n_format / total if total else 0.0

    typer.echo(f"\n{'='*50}")
    typer.echo(f"  Examples evaluated : {total}")
    typer.echo(f"  Accuracy (correct) : {n_correct}/{total} = {accuracy:.1%}")
    typer.echo(f"  Format rate (####) : {n_format}/{total} = {format_rate:.1%}")
    typer.echo(f"{'='*50}")

    if output_path:
        write_jsonl(output_path, results)
        typer.echo(f"Results written to {output_path}")


if __name__ == "__main__":
    app()
