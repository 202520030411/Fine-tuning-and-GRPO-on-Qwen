"""Download SVAMP and write prompt/target JSONL for evaluation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

hf_home = PROJECT_ROOT / ".hf"
os.environ.setdefault("HF_HOME", str(hf_home))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

from dataset.svamp import load_svamp_hf, preprocess_svamp_example, write_jsonl  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: str = typer.Option("dataset/processed", help="Output directory."),
    limit: int = typer.Option(500, help="Max examples to use (SVAMP has 1000 total)."),
) -> None:
    """Download SVAMP and write evaluation JSONL."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading SVAMP …")
    ds = load_svamp_hf()
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    rows = []
    for ex in tqdm(ds, desc="preprocess"):
        p = preprocess_svamp_example(ex["Body"], ex["Question"], ex["Answer"])
        rows.append({
            "question":     p.question,
            "prompt":       p.prompt,
            "target":       p.target,
            "final_answer": p.final_answer,
        })

    out_file = out_path / "svamp_test.jsonl"
    write_jsonl(out_file, rows)
    typer.echo(f"Wrote {len(rows)} examples to {out_file}")


if __name__ == "__main__":
    app()
