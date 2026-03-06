from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

# Subjects used for evaluation:
#   - math-related  → test whether fine-tuning helped or hurt math
#   - world_history → proxy for general knowledge retention (forgetting check)
SUBJECTS = [
    "high_school_mathematics",
    "elementary_mathematics",
    "world_history",
]

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


@dataclass(frozen=True)
class MMLUExample:
    subject: str
    question: str
    choices: list[str]
    correct_letter: str
    prompt: str


def _format_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "The following is a multiple choice question. "
        "Answer with only the letter of the correct option (A, B, C, or D).\n",
        f"Question: {question.strip()}",
    ]
    for i, choice in enumerate(choices):
        lines.append(f"{LETTER_MAP[i]}. {choice.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def preprocess_mmlu_example(subject: str, question: str, choices: list[str], answer: int) -> MMLUExample:
    return MMLUExample(
        subject=subject,
        question=question,
        choices=choices,
        correct_letter=LETTER_MAP[answer],
        prompt=_format_prompt(question, choices),
    )


def load_mmlu_subject(subject: str, split: str = "test"):
    """Load one MMLU subject. Falls back to 'validation' if 'test' is unavailable."""
    try:
        return load_dataset("cais/mmlu", subject, split=split)
    except Exception:
        return load_dataset("cais/mmlu", subject, split="validation")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
