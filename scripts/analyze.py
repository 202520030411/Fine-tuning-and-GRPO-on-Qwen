"""
Analyze and visualize fine-tuning results.

Inputs
------
  --base-results   : eval JSONL from the base (untuned) model
  --sft-results    : eval JSONL from the SFT model
  --grpo-results   : eval JSONL from the GRPO model
  --sft-log        : training JSONL written by train_sft.py (step, loss)
  --grpo-log       : training JSONL written by train_grpo.py (step, reward, pg_loss, kl, accuracy)

Outputs (all saved to images/)
-------------------------------
  accuracy_comparison.png
  format_rate_comparison.png
  sft_training_curve.png
  grpo_training_curves.png
  answer_length_distribution.png
  reward_distribution.png
  error_samples.txt
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import typer

app = typer.Typer(add_completion=False)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ensure_images_dir() -> Path:
    d = PROJECT_ROOT / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
    color: str = "steelblue",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [v * 100 for v in values], color=color, edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _training_curve_sft(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps  = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, marker="o", markersize=4, color="tomato", linewidth=1.5)
    ax.set_title("SFT Training Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Avg Cross-Entropy Loss")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _training_curves_grpo(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps    = [r["step"]     for r in rows]
    rewards  = [r["reward"]   for r in rows]
    accs     = [r.get("accuracy", float("nan")) for r in rows]
    pg_loss  = [r["pg_loss"]  for r in rows]
    kls      = [r["kl"]       for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("GRPO Training Curves", fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], rewards,  "Mean Reward",     "Reward",       "darkorange"),
        (axes[0, 1], accs,     "Accuracy (train)", "Accuracy",    "mediumseagreen"),
        (axes[1, 0], pg_loss,  "Policy-Gradient Loss", "PG Loss", "royalblue"),
        (axes[1, 1], kls,      "KL Divergence",    "KL",           "mediumpurple"),
    ]
    for ax, vals, title, ylabel, color in panels:
        ax.plot(steps, vals, marker="o", markersize=3, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _answer_length_distribution(datasets: dict[str, list[dict]], out_path: Path) -> None:
    """Histogram of completion token lengths split by correct / incorrect."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (name, rows) in zip(axes, datasets.items()):
        correct_lens   = [len(r["completion"].split()) for r in rows if r.get("correct", 0) > 0]
        incorrect_lens = [len(r["completion"].split()) for r in rows if r.get("correct", 0) == 0]
        ax.hist(incorrect_lens, bins=30, alpha=0.6, label="Incorrect", color="tomato")
        ax.hist(correct_lens,   bins=30, alpha=0.6, label="Correct",   color="steelblue")
        ax.set_title(f"{name} — Answer Length", fontsize=11)
        ax.set_xlabel("Completion length (words)")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Answer Length Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _reward_distribution(datasets: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["steelblue", "darkorange", "mediumseagreen"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, rows), color in zip(datasets.items(), colors):
        rewards = [r.get("reward", r.get("correct", 0)) for r in rows]
        ax.hist(rewards, bins=10, alpha=0.6, label=name, color=color)
    ax.set_title("Reward Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _summary_table(datasets: dict[str, list[dict]]) -> None:
    header = f"{'Model':<12} {'N':>6} {'Accuracy':>10} {'Format%':>10} {'AvgReward':>11}"
    typer.echo("\n" + "=" * len(header))
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, rows in datasets.items():
        n          = len(rows)
        accuracy   = sum(r.get("correct", 0) > 0 for r in rows) / n if n else 0
        format_r   = sum(r.get("format",  0) > 0 for r in rows) / n if n else 0
        avg_reward = sum(r.get("reward",  0)      for r in rows) / n if n else 0
        typer.echo(f"{name:<12} {n:>6} {accuracy:>10.1%} {format_r:>10.1%} {avg_reward:>11.4f}")
    typer.echo("=" * len(header) + "\n")


def _write_error_samples(datasets: dict[str, list[dict]], out_path: Path, n: int = 5) -> None:
    lines = []
    for name, rows in datasets.items():
        wrong = [r for r in rows if r.get("correct", 0) == 0][:n]
        lines.append(f"\n{'='*60}\n{name} — {len(wrong)} wrong samples (of first {n} shown)\n{'='*60}")
        for i, r in enumerate(wrong, 1):
            lines.append(
                f"\n[{i}] Question: {r.get('question','')[:120]}\n"
                f"    Gold    : {r.get('gold','')}\n"
                f"    Output  : {r.get('completion','')[:200]}"
            )
    text = "\n".join(lines)
    out_path.write_text(text)
    typer.echo(f"  Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    base_results:  Optional[str] = typer.Option(None, "--base-results",  help="Eval JSONL from base model."),
    sft_results:   Optional[str] = typer.Option(None, "--sft-results",   help="Eval JSONL from SFT model."),
    grpo_results:  Optional[str] = typer.Option(None, "--grpo-results",  help="Eval JSONL from GRPO model."),
    sft_log:       Optional[str] = typer.Option(None, "--sft-log",       help="Training JSONL from train_sft.py."),
    grpo_log:      Optional[str] = typer.Option(None, "--grpo-log",      help="Training JSONL from train_grpo.py."),
    images_dir:    Optional[str] = typer.Option(None, "--images-dir",    help="Output directory for plots (default: <repo>/images)."),
) -> None:
    """Analyze eval results and training logs; produce plots and a summary table."""
    import matplotlib
    matplotlib.use("Agg")  # headless rendering — no display required

    out_dir = Path(images_dir) if images_dir else _ensure_images_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load eval datasets ──────────────────────────────────────────────────
    eval_datasets: dict[str, list[dict]] = {}
    for label, path in [("Base", base_results), ("SFT", sft_results), ("GRPO", grpo_results)]:
        if path:
            if not Path(path).exists():
                typer.echo(f"WARNING: {label} results file not found: {path} — skipping")
                continue
            typer.echo(f"Loading {label} results from {path} …")
            eval_datasets[label] = _read_jsonl(path)

    if not eval_datasets:
        typer.echo("No eval result files provided — skipping eval plots.")
    else:
        # Summary table
        _summary_table(eval_datasets)

        # Accuracy comparison
        labels = list(eval_datasets.keys())
        accuracies = [
            sum(r.get("correct", 0) > 0 for r in rows) / len(rows)
            for rows in eval_datasets.values()
        ]
        format_rates = [
            sum(r.get("format", 0) > 0 for r in rows) / len(rows)
            for rows in eval_datasets.values()
        ]
        _bar_chart(
            labels, accuracies,
            "Accuracy by Model", "Accuracy (%)",
            out_dir / "accuracy_comparison.png",
            color="steelblue",
        )
        _bar_chart(
            labels, format_rates,
            "Format Rate (#### marker) by Model", "Format Rate (%)",
            out_dir / "format_rate_comparison.png",
            color="darkorange",
        )
        _answer_length_distribution(eval_datasets, out_dir / "answer_length_distribution.png")
        _reward_distribution(eval_datasets, out_dir / "reward_distribution.png")
        _write_error_samples(eval_datasets, out_dir / "error_samples.txt")

    # ── Training curves ─────────────────────────────────────────────────────
    if sft_log:
        if not Path(sft_log).exists():
            typer.echo(f"WARNING: SFT log not found: {sft_log} — skipping")
        else:
            typer.echo(f"Loading SFT training log from {sft_log} …")
            sft_rows = _read_jsonl(sft_log)
            if sft_rows:
                _training_curve_sft(sft_rows, out_dir / "sft_training_curve.png")

    if grpo_log:
        if not Path(grpo_log).exists():
            typer.echo(f"WARNING: GRPO log not found: {grpo_log} — skipping")
        else:
            typer.echo(f"Loading GRPO training log from {grpo_log} …")
            grpo_rows = _read_jsonl(grpo_log)
            if grpo_rows:
                _training_curves_grpo(grpo_rows, out_dir / "grpo_training_curves.png")

    typer.echo(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
