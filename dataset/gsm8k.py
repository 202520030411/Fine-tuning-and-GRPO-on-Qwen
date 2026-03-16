from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from datasets import load_dataset


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str
    prompt: str
    target: str
    final_answer: str


_FINAL_ANSWER_RE = re.compile(r"####\s*(.+)\s*$")


def _extract_final_answer(gsm8k_answer_field: str) -> str:
    """
    GSM8K 'answer' field usually ends with: '#### <final>'.
    We return the raw final string (not normalized).
    """
    m = _FINAL_ANSWER_RE.search(gsm8k_answer_field.strip())
    if not m:
        return ""
    return m.group(1).strip()


def _format_prompt(question: str) -> str:
    return (
        "You are a helpful assistant that solves grade-school math word problems. "
        "Use basic arithmetic operations (+, −, ×, ÷) and show your reasoning step by step.\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def _format_target(answer_field: str) -> str:
    # Keep the original GSM8K rationale but make the end explicit.
    final = _extract_final_answer(answer_field)
    rationale = answer_field.strip()
    if final:
        return f" Let's think step by step.\n{rationale}\nFinal answer: {final}"
    return f" Let's think step by step.\n{rationale}"


def preprocess_gsm8k_example(question: str, answer: str) -> GSM8KExample:
    prompt = _format_prompt(question)
    target = _format_target(answer)
    final = _extract_final_answer(answer)
    return GSM8KExample(
        question=question,
        answer=answer,
        prompt=prompt,
        target=target,
        final_answer=final,
    )


def load_gsm8k_hf(split: Literal["train", "test"]):
    # Official dataset name on HF: 'gsm8k' config 'main'
    return load_dataset("openai/gsm8k", "main", split=split)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

