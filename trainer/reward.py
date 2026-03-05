from __future__ import annotations

import re


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", flags=re.MULTILINE)
_FINAL_RE = re.compile(r"final\s+answer\s*[:=]\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


def extract_final_answer_from_text(text: str) -> str:
    """
    Best-effort extraction of the final answer from a model completion.

    Priority:
      1. GSM8K-style marker  '#### 42'
      2. Natural-language cue  'Final answer: 42'
      3. Last number in the text
    """
    t = text.strip()
    m = _HASH_RE.search(t)
    if m:
        return m.group(1).strip()
    m = _FINAL_RE.search(t)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(t)
    return nums[-1].strip() if nums else ""


def normalize_answer(ans: str) -> str:
    """Normalize to a canonical numeric string for comparison."""
    ans = ans.strip().replace(",", "")
    nums = _NUM_RE.findall(ans)
    if nums:
        # Return the last number with commas stripped
        return nums[-1].replace(",", "")
    return ans.lower()


def correctness_reward(completion_text: str, ground_truth_final: str) -> float:
    """1.0 if the extracted answer matches ground truth, else 0.0."""
    pred = normalize_answer(extract_final_answer_from_text(completion_text))
    gold = normalize_answer(ground_truth_final)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0


def format_reward(completion_text: str) -> float:
    """
    0.5 if the completion contains a '#### <number>' marker (well-formed CoT),
    0.0 otherwise.

    This nudges GRPO to produce structured answers even when the numeric
    value is wrong, making future extraction more reliable.
    """
    return 0.5 if _HASH_RE.search(completion_text.strip()) else 0.0


def compute_gsm8k_reward(completion_text: str, ground_truth_final: str) -> dict[str, float]:
    """
    Combined reward used by both eval and GRPO.

    Returns
    -------
    {
        "reward":       float in {0.0, 1.0, 1.5}  – correctness + format bonus
        "correct":      float in {0.0, 1.0}
        "format":       float in {0.0, 0.5}
    }

    Scoring:
      - Correct answer with proper #### marker → 1.5
      - Correct answer without marker         → 1.0
      - Wrong answer with proper #### marker  → 0.5
      - Wrong answer without marker           → 0.0
    """
    c = correctness_reward(completion_text, ground_truth_final)
    f = format_reward(completion_text)
    return {"reward": c + f, "correct": c, "format": f}
