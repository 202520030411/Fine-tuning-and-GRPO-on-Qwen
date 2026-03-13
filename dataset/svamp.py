from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


@dataclass(frozen=True)
class SVAMPExample:
    question: str
    prompt: str
    target: str
    final_answer: str


def _format_answer(answer) -> str:
    """Normalise numeric answer to a clean string (drop .0 for integers)."""
    try:
        f = float(answer)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(answer).strip()


def _format_prompt(body: str, question: str) -> str:
    return (
        "You are a helpful assistant that solves math word problems. "
        "Use basic arithmetic operations (+, −, ×, ÷) and show your reasoning step by step.\n\n"
        f"Question: {body.strip()} {question.strip()}\n"
        "Answer:"
    )


def _format_target(answer) -> str:
    final = _format_answer(answer)
    return f" Let's think step by step.\n#### {final}"


def preprocess_svamp_example(body: str, question: str, answer) -> SVAMPExample:
    final = _format_answer(answer)
    return SVAMPExample(
        question=f"{body.strip()} {question.strip()}",
        prompt=_format_prompt(body, question),
        target=_format_target(answer),
        final_answer=final,
    )


def load_svamp_hf():
    """Load SVAMP. The dataset has a single split (1000 examples)."""
    try:
        return load_dataset("ChilleD/SVAMP", split="train")
    except Exception:
        return load_dataset("ChilleD/SVAMP", split="test")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
