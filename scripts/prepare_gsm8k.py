from __future__ import annotations

from pathlib import Path
from typing import Optional

import os
import sys

import typer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep HF caches inside the repo so this works in restricted environments.
# Must be set BEFORE importing datasets / huggingface_hub.
hf_home = PROJECT_ROOT / ".hf"
os.environ.setdefault("HF_HOME", str(hf_home))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

from dataset.gsm8k import load_gsm8k_hf, preprocess_gsm8k_example, write_jsonl  # noqa: E402


app = typer.Typer(add_completion=False)


def _maybe_limit(ds, limit: Optional[int], seed: int):
    if limit is None:
        return ds
    if limit <= 0:
        raise typer.BadParameter("limit must be > 0")
    if limit >= len(ds):
        return ds
    # Shuffle then select for a more representative small subset.
    return ds.shuffle(seed=seed).select(range(limit))


@app.command()
def main(
    out_dir: str = typer.Option("dataset/processed", help="Output directory for processed jsonl files."),
    limit_train: Optional[int] = typer.Option(None, help="Optional cap on number of train examples (for smoke tests)."),
    limit_test: Optional[int] = typer.Option(None, help="Optional cap on number of test examples (for smoke tests)."),
    seed: int = typer.Option(0, help="Seed used when shuffling before applying limits."),
) -> None:
    """
    Download GSM8K (HF datasets) and write prompt/target jsonl files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds = _maybe_limit(load_gsm8k_hf("train"), limit_train, seed)
    test_ds = _maybe_limit(load_gsm8k_hf("test"), limit_test, seed)

    def convert(ds):
        rows = []
        for ex in tqdm(ds, desc="preprocess"):
            p = preprocess_gsm8k_example(ex["question"], ex["answer"])
            rows.append(
                {
                    "question": p.question,
                    "answer": p.answer,
                    "prompt": p.prompt,
                    "target": p.target,
                    "final_answer": p.final_answer,
                }
            )
        return rows

    train_rows = convert(train_ds)
    test_rows = convert(test_ds)

    train_file = out_path / "gsm8k_train.jsonl"
    test_file = out_path / "gsm8k_test.jsonl"
    write_jsonl(train_file, train_rows)
    write_jsonl(test_file, test_rows)

    typer.echo(f"Wrote {len(train_rows)} train rows to {train_file}")
    typer.echo(f"Wrote {len(test_rows)} test rows to {test_file}")


if __name__ == "__main__":
    app()

