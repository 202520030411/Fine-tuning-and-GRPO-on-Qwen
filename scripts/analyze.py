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


def _training_curve_sft(rows: list[dict], out_path: Path, title: str = "SFT Training Loss") -> None:
    import matplotlib.pyplot as plt

    steps  = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, marker="o", markersize=4, color="tomato", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
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


# ── Log readers ───────────────────────────────────────────────────────────────

def _read_grpo_log(path: str) -> list[dict]:
    """
    Read GRPO training log in either format:
    - Custom JSONL (one dict per line with keys: step, reward, pg_loss, kl, accuracy)
    - TRL trainer_state.json (JSON with a top-level "log_history" list)
    """
    p = Path(path)
    if p.suffix == ".json":
        with p.open() as f:
            state = json.load(f)
        raw = state.get("log_history", [])
        rows = []
        for entry in raw:
            if "step" not in entry:
                continue
            row: dict = {"step": entry["step"]}
            # TRL logs 'rewards/mean' or 'reward' depending on version
            row["reward"] = entry.get("reward", entry.get("rewards/mean", 0.0))
            row["pg_loss"] = entry.get("loss", 0.0)
            row["kl"] = entry.get("kl", 0.0)
            # TRL doesn't log accuracy; default to 0 so the plot is still drawn
            row["accuracy"] = entry.get("accuracy", 0.0)
            rows.append(row)
        return rows
    # Fall back to custom JSONL format
    return _read_jsonl(path)


# ── SVAMP / MMLU helpers ──────────────────────────────────────────────────────

def _cross_dataset_accuracy(
    gsm8k_sets: dict[str, list[dict]],
    svamp_sets: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Grouped bar chart: GSM8K vs SVAMP accuracy per model."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = list({**gsm8k_sets, **svamp_sets}.keys())
    gsm8k_acc  = [sum(r.get("correct", 0) > 0 for r in gsm8k_sets.get(m, [])) / max(1, len(gsm8k_sets.get(m, []))) for m in models]
    svamp_acc  = [sum(r.get("correct", 0) > 0 for r in svamp_sets.get(m, [])) / max(1, len(svamp_sets.get(m, []))) for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [v * 100 for v in gsm8k_acc], w, label="GSM8K", color="steelblue")
    ax.bar(x + w / 2, [v * 100 for v in svamp_acc],  w, label="SVAMP",  color="mediumseagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Math Accuracy: GSM8K vs SVAMP", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _mmlu_forgetting_chart(mmlu_sets: dict[str, list[dict]], out_path: Path) -> None:
    """Bar chart showing MMLU accuracy per subject per model (forgetting check)."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect subjects
    subjects: list[str] = []
    for rows in mmlu_sets.values():
        for r in rows:
            s = r.get("subject", "unknown")
            if s not in subjects:
                subjects.append(s)

    models = list(mmlu_sets.keys())
    x = np.arange(len(subjects))
    w = 0.8 / max(len(models), 1)
    colors = ["steelblue", "darkorange", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 3), 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        rows = mmlu_sets[model]
        accs = []
        for subj in subjects:
            subj_rows = [r for r in rows if r.get("subject") == subj]
            acc = sum(r.get("correct", 0) for r in subj_rows) / max(1, len(subj_rows))
            accs.append(acc * 100)
        offset = (i - len(models) / 2 + 0.5) * w
        bars = ax.bar(x + offset, accs, w, label=model, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    short_subjects = [s.replace("_", " ").replace("high school", "HS").replace("elementary", "Elem") for s in subjects]
    ax.set_xticks(x)
    ax.set_xticklabels(short_subjects, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MMLU Accuracy by Subject (Forgetting Check)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _dora_vs_lora_chart(eval_datasets: dict[str, list[dict]], out_path: Path) -> None:
    """Bar chart comparing LoRA (SFT) vs DoRA accuracy and format rate side by side."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(eval_datasets.keys())
    accs   = [sum(r.get("correct", 0) > 0 for r in eval_datasets[m]) / max(1, len(eval_datasets[m])) * 100 for m in models]
    fmts   = [sum(r.get("format",  0) > 0 for r in eval_datasets[m]) / max(1, len(eval_datasets[m])) * 100 for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy",    color="steelblue",   edgecolor="black")
    bars2 = ax.bar(x + w / 2, fmts, w, label="Format Rate", color="darkorange",  edgecolor="black")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Model Comparison: Accuracy & Format Rate", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _prompt_comparison_chart(
    prompt_dirs: dict[str, str],
    out_path: Path,
) -> None:
    """
    Grouped bar chart: prompt strategies (direct / cot / rules) on x-axis,
    one bar group per model.
    Reads prompt_comparison_summary.json from each directory.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_data: dict[str, dict[str, float]] = {}  # model -> strategy -> accuracy
    for model, dir_path in prompt_dirs.items():
        summary_file = Path(dir_path) / "prompt_comparison_summary.json"
        if not summary_file.exists():
            typer.echo(f"  WARNING: {summary_file} not found — skipping {model}")
            continue
        with open(summary_file) as f:
            metrics = json.load(f)
        all_data[model] = {m["strategy"]: m["accuracy"] * 100 for m in metrics}

    if not all_data:
        typer.echo("  No prompt comparison data found — skipping chart.")
        return

    strategies = ["direct", "cot", "rules"]
    strategy_labels = {"direct": "Direct\n(no guidance)", "cot": "CoT\n(step by step)", "rules": "Math Rules\n(+,−,×,÷)"}
    models = list(all_data.keys())
    x = np.arange(len(strategies))
    w = 0.8 / max(len(models), 1)
    colors = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [all_data[model].get(s, 0) for s in strategies]
        offset = (i - len(models) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=model, color=color, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([strategy_labels.get(s, s) for s in strategies], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Prompt Strategy Comparison (GSM8K)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _reasoning_comparison_chart(
    reasoning_dirs: dict[str, str],
    out_path: Path,
) -> None:
    """
    Grouped bar chart comparing reasoning validity metrics across models.
    Reads reasoning_summary.json from each directory.
    Metrics shown: step_accuracy, reasoning_validity_rate, final_answer_accuracy.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model_summaries: dict[str, dict] = {}
    for model, dir_path in reasoning_dirs.items():
        summary_file = Path(dir_path) / "reasoning_summary.json"
        if not summary_file.exists():
            typer.echo(f"  WARNING: {summary_file} not found — skipping {model}")
            continue
        with open(summary_file) as f:
            model_summaries[model] = json.load(f)

    if not model_summaries:
        typer.echo("  No reasoning summary data found — skipping chart.")
        return

    metrics = [
        ("step_accuracy",          "Step Accuracy",          "#FF9800"),
        ("reasoning_validity_rate","Reasoning Validity",     "#4CAF50"),
        ("final_answer_accuracy",  "Final Answer Accuracy",  "#2196F3"),
    ]
    models = list(model_summaries.keys())
    x = np.arange(len(models))
    w = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 5))
    for i, (key, label, color) in enumerate(metrics):
        vals = [model_summaries[m].get(key, 0) * 100 for m in models]
        offset = (i - len(metrics) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Step-by-Step Reasoning Validity by Model", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    typer.echo(f"  Saved {out_path}")


def _mmlu_summary_table(mmlu_sets: dict[str, list[dict]]) -> None:
    subjects: list[str] = []
    for rows in mmlu_sets.values():
        for r in rows:
            s = r.get("subject", "unknown")
            if s not in subjects:
                subjects.append(s)

    header = f"{'Model':<8}  " + "  ".join(f"{s[:22]:<22}" for s in subjects) + "  Overall"
    typer.echo("\n" + "=" * len(header))
    typer.echo("MMLU Results")
    typer.echo("=" * len(header))
    typer.echo(header)
    typer.echo("-" * len(header))
    for model, rows in mmlu_sets.items():
        parts = []
        total_c, total_n = 0, 0
        for subj in subjects:
            sr = [r for r in rows if r.get("subject") == subj]
            c = sum(r.get("correct", 0) for r in sr)
            n = len(sr)
            total_c += c; total_n += n
            parts.append(f"{c}/{n}={c/max(1,n):.0%}")
        overall = f"{total_c/max(1,total_n):.0%}"
        typer.echo(f"{model:<8}  " + "  ".join(f"{p:<22}" for p in parts) + f"  {overall}")
    typer.echo("=" * len(header))


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    # ── GSM8K eval ────────────────────────────────────────────────────────
    base_results:       Optional[str] = typer.Option(None, "--base-results",       help="GSM8K eval JSONL from base model."),
    lora_sft_results:   Optional[str] = typer.Option(None, "--lora-sft-results",   help="GSM8K eval JSONL from LoRA-SFT model (comparison)."),
    sft_results:        Optional[str] = typer.Option(None, "--sft-results",        help="GSM8K eval JSONL from main SFT model (DoRA-SFT in full DoRA pipeline)."),
    grpo_results:       Optional[str] = typer.Option(None, "--grpo-results",       help="GSM8K eval JSONL from GRPO model."),
    dora_results:       Optional[str] = typer.Option(None, "--dora-results",       help="GSM8K eval JSONL from standalone DoRA model (optional)."),
    # ── SVAMP eval ────────────────────────────────────────────────────────
    svamp_base:         Optional[str] = typer.Option(None, "--svamp-base",         help="SVAMP eval JSONL from base model."),
    svamp_sft:          Optional[str] = typer.Option(None, "--svamp-sft",          help="SVAMP eval JSONL from SFT model."),
    svamp_grpo:         Optional[str] = typer.Option(None, "--svamp-grpo",         help="SVAMP eval JSONL from GRPO model."),
    # ── MMLU eval ─────────────────────────────────────────────────────────
    mmlu_base:          Optional[str] = typer.Option(None, "--mmlu-base",          help="MMLU eval JSONL from base model."),
    mmlu_sft:           Optional[str] = typer.Option(None, "--mmlu-sft",           help="MMLU eval JSONL from SFT model."),
    mmlu_grpo:          Optional[str] = typer.Option(None, "--mmlu-grpo",          help="MMLU eval JSONL from GRPO model."),
    # ── Training logs ─────────────────────────────────────────────────────
    sft_log:            Optional[str] = typer.Option(None, "--sft-log",            help="Training JSONL from DoRA-SFT (train_sft.py --use-dora)."),
    grpo_log:           Optional[str] = typer.Option(None, "--grpo-log",           help="trainer_state.json from TRL GRPOTrainer."),
    dora_log:           Optional[str] = typer.Option(None, "--lora-sft-log",       help="Training JSONL from LoRA-SFT (train_sft.py without --use-dora).")  ,
    # ── Prompt comparison dirs ────────────────────────────────────────────
    prompt_base_dir:    Optional[str] = typer.Option(None, "--prompt-base-dir",    help="Output dir of eval_prompts.py run on base model."),
    prompt_sft_dir:     Optional[str] = typer.Option(None, "--prompt-sft-dir",     help="Output dir of eval_prompts.py run on SFT model."),
    # ── Reasoning validity dirs ───────────────────────────────────────────
    reasoning_base_dir: Optional[str] = typer.Option(None, "--reasoning-base-dir", help="Output dir of eval_reasoning.py run on base model."),
    reasoning_sft_dir:  Optional[str] = typer.Option(None, "--reasoning-sft-dir",  help="Output dir of eval_reasoning.py run on SFT model."),
    reasoning_grpo_dir: Optional[str] = typer.Option(None, "--reasoning-grpo-dir", help="Output dir of eval_reasoning.py run on GRPO model."),
    reasoning_dora_dir: Optional[str] = typer.Option(None, "--reasoning-dora-dir", help="Output dir of eval_reasoning.py run on DoRA model."),
    # ── Output ────────────────────────────────────────────────────────────
    images_dir:         Optional[str] = typer.Option(None, "--images-dir",         help="Output directory for plots."),
) -> None:
    """Analyze eval results and training logs; produce plots and a summary table."""
    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(images_dir) if images_dir else _ensure_images_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── GSM8K eval (Base / LoRA-SFT / DoRA-SFT / GRPO) ──────────────────
    eval_datasets: dict[str, list[dict]] = {}
    for label, path in [("Base", base_results), ("LoRA-SFT", lora_sft_results),
                         ("SFT", sft_results), ("GRPO", grpo_results),
                         ("DoRA", dora_results)]:
        if path:
            if not Path(path).exists():
                typer.echo(f"WARNING: {label} results file not found: {path} — skipping")
                continue
            typer.echo(f"Loading {label} results from {path} …")
            eval_datasets[label] = _read_jsonl(path)

    if not eval_datasets:
        typer.echo("No GSM8K eval result files provided — skipping GSM8K plots.")
    else:
        _summary_table(eval_datasets)
        labels = list(eval_datasets.keys())
        accuracies   = [sum(r.get("correct", 0) > 0 for r in rows) / len(rows) for rows in eval_datasets.values()]
        format_rates = [sum(r.get("format",  0) > 0 for r in rows) / len(rows) for rows in eval_datasets.values()]
        _bar_chart(labels, accuracies,   "GSM8K Accuracy by Model",            "Accuracy (%)",    out_dir / "accuracy_comparison.png",   color="steelblue")
        _bar_chart(labels, format_rates, "Format Rate (#### marker) by Model", "Format Rate (%)", out_dir / "format_rate_comparison.png", color="darkorange")
        _answer_length_distribution(eval_datasets, out_dir / "answer_length_distribution.png")
        _reward_distribution(eval_datasets, out_dir / "reward_distribution.png")
        _write_error_samples(eval_datasets, out_dir / "error_samples.txt")
        # Combined accuracy + format chart (useful when DoRA is included)
        if len(eval_datasets) > 1:
            _dora_vs_lora_chart(eval_datasets, out_dir / "model_comparison.png")

    # ── SVAMP eval ──────────────────────────────────────────────────────────
    svamp_datasets: dict[str, list[dict]] = {}
    for label, path in [("Base", svamp_base), ("SFT", svamp_sft), ("GRPO", svamp_grpo)]:
        if path:
            if not Path(path).exists():
                typer.echo(f"WARNING: SVAMP {label} not found: {path} — skipping")
                continue
            typer.echo(f"Loading SVAMP {label} from {path} …")
            svamp_datasets[label] = _read_jsonl(path)

    if svamp_datasets:
        svamp_labels = list(svamp_datasets.keys())
        svamp_accs   = [sum(r.get("correct", 0) > 0 for r in rows) / max(1, len(rows)) for rows in svamp_datasets.values()]
        _bar_chart(svamp_labels, svamp_accs, "SVAMP Accuracy by Model", "Accuracy (%)", out_dir / "svamp_accuracy.png", color="mediumseagreen")
        if eval_datasets:
            _cross_dataset_accuracy(eval_datasets, svamp_datasets, out_dir / "gsm8k_vs_svamp.png")

    # ── MMLU eval ───────────────────────────────────────────────────────────
    mmlu_datasets: dict[str, list[dict]] = {}
    for label, path in [("Base", mmlu_base), ("SFT", mmlu_sft), ("GRPO", mmlu_grpo)]:
        if path:
            if not Path(path).exists():
                typer.echo(f"WARNING: MMLU {label} not found: {path} — skipping")
                continue
            typer.echo(f"Loading MMLU {label} from {path} …")
            mmlu_datasets[label] = _read_jsonl(path)

    if mmlu_datasets:
        _mmlu_summary_table(mmlu_datasets)
        _mmlu_forgetting_chart(mmlu_datasets, out_dir / "mmlu_forgetting.png")

    # ── Training curves ─────────────────────────────────────────────────────
    if sft_log:
        if not Path(sft_log).exists():
            typer.echo(f"WARNING: DoRA-SFT log not found: {sft_log} — skipping")
        else:
            typer.echo(f"Loading DoRA-SFT training log from {sft_log} …")
            sft_rows = _read_jsonl(sft_log)
            if sft_rows:
                _training_curve_sft(sft_rows, out_dir / "dora_sft_training_curve.png", title="DoRA-SFT Training Loss")

    if dora_log:
        if not Path(dora_log).exists():
            typer.echo(f"WARNING: LoRA-SFT log not found: {dora_log} — skipping")
        else:
            typer.echo(f"Loading LoRA-SFT training log from {dora_log} …")
            dora_rows = _read_jsonl(dora_log)
            if dora_rows:
                _training_curve_sft(dora_rows, out_dir / "lora_sft_training_curve.png", title="LoRA-SFT Training Loss")

    if grpo_log:
        if not Path(grpo_log).exists():
            typer.echo(f"WARNING: GRPO log not found: {grpo_log} — skipping")
        else:
            typer.echo(f"Loading GRPO training log from {grpo_log} …")
            grpo_rows = _read_grpo_log(grpo_log)
            if grpo_rows:
                _training_curves_grpo(grpo_rows, out_dir / "grpo_training_curves.png")

    # ── Prompt comparison ───────────────────────────────────────────────────
    prompt_dirs: dict[str, str] = {}
    for label, d in [("Base", prompt_base_dir), ("SFT", prompt_sft_dir)]:
        if d and Path(d).exists():
            prompt_dirs[label] = d
        elif d:
            typer.echo(f"WARNING: prompt dir not found: {d} — skipping {label}")
    if prompt_dirs:
        typer.echo("Generating prompt comparison chart …")
        _prompt_comparison_chart(prompt_dirs, out_dir / "prompt_comparison.png")

    # ── Reasoning validity ──────────────────────────────────────────────────
    reasoning_dirs: dict[str, str] = {}
    for label, d in [
        ("Base",  reasoning_base_dir),
        ("SFT",   reasoning_sft_dir),
        ("GRPO",  reasoning_grpo_dir),
        ("DoRA",  reasoning_dora_dir),
    ]:
        if d and Path(d).exists():
            reasoning_dirs[label] = d
        elif d:
            typer.echo(f"WARNING: reasoning dir not found: {d} — skipping {label}")
    if reasoning_dirs:
        typer.echo("Generating reasoning validity chart …")
        _reasoning_comparison_chart(reasoning_dirs, out_dir / "reasoning_comparison.png")

    typer.echo(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    app()
