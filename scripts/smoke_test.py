"""
Smoke test — verifies the full pipeline runs end-to-end without errors.

Runs with minimal settings (3 steps, 5 examples) so it completes in ~2 min
on a GPU and can catch import errors, shape mismatches, and CLI mistakes
before committing to a 4-5 hour full training run.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --keep-tmp   # don't delete temp outputs

What is tested:
    1.  train_sft.py  --use-dora   (DoRA-SFT,  3 steps)
    2.  train_sft.py               (LoRA-SFT,  3 steps)
    3.  train_grpo.py              (DoRA-GRPO, 3 steps, group_size=2)
    4.  eval.py                    (base model, 5 examples)
    5.  eval.py                    (SFT adapter, 5 examples)
    6.  eval.py                    (GRPO merged model, 5 examples)
    7.  eval_mmlu.py               (base model, 5 examples per subject)
    8.  eval_prompts.py            (base model, 5 examples)
    9.  eval_reasoning.py          (from existing eval results)
    10. analyze.py                 (all results + plots)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(add_completion=False)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def _run(label: str, cmd: list[str], cwd: Path) -> bool:
    typer.echo(f"\n── {label} {'─' * max(0, 55 - len(label))}")
    typer.echo("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if result.returncode == 0:
        typer.echo(f"  {PASS} passed")
        return True
    else:
        typer.echo(f"  {FAIL} FAILED (exit {result.returncode})")
        return False


def _check(label: str, path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    status = PASS if ok else FAIL
    typer.echo(f"  {status} {label}: {path}")
    return ok


@app.command()
def main(
    model: str = typer.Option("Qwen/Qwen3-0.6B-Base", help="Model to test with."),
    keep_tmp: bool = typer.Option(False, "--keep-tmp", help="Keep temp output directory after test."),
    data_dir: str = typer.Option("dataset/processed", help="Directory with gsm8k_train.jsonl and gsm8k_test.jsonl."),
) -> None:
    """Run a quick end-to-end smoke test of the full pipeline."""
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    typer.echo(f"Temp directory: {tmp}")

    results: list[tuple[str, bool]] = []

    def step(label: str, cmd: list[str]) -> bool:
        ok = _run(label, cmd, cwd=PROJECT_ROOT)
        results.append((label, ok))
        return ok

    py = sys.executable
    train_path  = str(PROJECT_ROOT / data_dir / "gsm8k_train.jsonl")
    test_path   = str(PROJECT_ROOT / data_dir / "gsm8k_test.jsonl")
    svamp_path  = str(PROJECT_ROOT / data_dir / "svamp_test.jsonl")

    dora_sft_dir  = str(tmp / "dora_sft")
    lora_sft_dir  = str(tmp / "lora_sft")
    grpo_dir      = str(tmp / "grpo")
    images_dir    = str(tmp / "images")

    # ── 1. DoRA-SFT ─────────────────────────────────────────────────────────
    step("1. DoRA-SFT (3 steps)", [
        py, "scripts/train_sft.py",
        "--model-name-or-path", model,
        "--train-path", train_path,
        "--output-dir", dora_sft_dir,
        "--train-log-path", str(tmp / "dora_sft_log.jsonl"),
        "--max-steps", "3",
        "--per-device-batch-size", "1",
        "--grad-accum", "1",
        "--max-length", "128",
        "--fp16",
        "--use-dora",
        "--log-every", "1",
    ])

    # ── 2. LoRA-SFT ─────────────────────────────────────────────────────────
    step("2. LoRA-SFT (3 steps)", [
        py, "scripts/train_sft.py",
        "--model-name-or-path", model,
        "--train-path", train_path,
        "--output-dir", lora_sft_dir,
        "--train-log-path", str(tmp / "lora_sft_log.jsonl"),
        "--max-steps", "3",
        "--per-device-batch-size", "1",
        "--grad-accum", "1",
        "--max-length", "128",
        "--fp16",
        "--log-every", "1",
    ])

    # ── 3. DoRA-GRPO ────────────────────────────────────────────────────────
    step("3. DoRA-GRPO (3 steps, group_size=2)", [
        py, "scripts/train_grpo.py",
        "--model-name-or-path", model,
        "--ref-model-path", dora_sft_dir,
        "--train-path", train_path,
        "--output-dir", grpo_dir,
        "--max-steps", "3",
        "--group-size", "2",
        "--kl-coef", "0.01",
        "--max-completion-length", "64",
    ])

    # ── 4–6. Eval ────────────────────────────────────────────────────────────
    for label, extra in [
        ("4. Eval base",    []),
        ("5. Eval DoRA-SFT", ["--adapter-path", dora_sft_dir]),
        ("6. Eval GRPO",    ["--base-model", grpo_dir]),
    ]:
        base = grpo_dir if "GRPO" in label else model
        step(label, [
            py, "scripts/eval.py",
            "--base-model", base,
            "--test-path", test_path,
            "--output-path", str(tmp / f"eval_{label.split('.')[1].strip().lower().replace(' ', '_')}.jsonl"),
            "--max-new-tokens", "64",
            "--max-examples", "5",
            "--batch-size", "1",
        ] + (extra if "GRPO" not in label else []))

    # ── 7. MMLU eval ────────────────────────────────────────────────────────
    step("7. MMLU eval base (5 examples)", [
        py, "scripts/eval_mmlu.py",
        "--base-model", model,
        "--output-path", str(tmp / "eval_mmlu.jsonl"),
        "--subjects", "high_school_mathematics",
        "--max-examples-per-subject", "5",
        "--batch-size", "1",
    ])

    # ── 8. Prompt comparison ─────────────────────────────────────────────────
    step("8. Prompt comparison (5 examples)", [
        py, "scripts/eval_prompts.py",
        "--base-model", model,
        "--test-path", test_path,
        "--output-dir", str(tmp / "prompt_base"),
        "--max-examples", "5",
    ])

    # ── 9. Reasoning validity ────────────────────────────────────────────────
    step("9. Reasoning validity", [
        py, "scripts/eval_reasoning.py",
        "--results-path", str(tmp / "eval_base.jsonl"),
        "--output-dir", str(tmp / "reasoning_base"),
    ])

    # ── 10. Analyze ──────────────────────────────────────────────────────────
    step("10. Analyze + plots", [
        py, "scripts/analyze.py",
        "--base-results",         str(tmp / "eval_base.jsonl"),
        "--lora-sft-results",     str(tmp / "eval_lora_sft.jsonl"),
        "--sft-results",          str(tmp / "eval_dora_sft.jsonl"),
        "--grpo-results",         str(tmp / "eval_grpo.jsonl"),
        "--sft-log",              str(tmp / "dora_sft_log.jsonl"),
        "--lora-sft-log",         str(tmp / "lora_sft_log.jsonl"),
        "--grpo-log",             str(tmp / "grpo" / "trainer_state.json"),
        "--mmlu-base",            str(tmp / "eval_mmlu.jsonl"),
        "--prompt-base-dir",      str(tmp / "prompt_base"),
        "--reasoning-base-dir",   str(tmp / "reasoning_base"),
        "--images-dir",           images_dir,
    ])

    # ── Summary ─────────────────────────────────────────────────────────────
    typer.echo("\n" + "=" * 60)
    typer.echo("SMOKE TEST SUMMARY")
    typer.echo("=" * 60)
    passed = sum(ok for _, ok in results)
    for label, ok in results:
        typer.echo(f"  {PASS if ok else FAIL}  {label}")
    typer.echo("=" * 60)
    typer.echo(f"  {passed}/{len(results)} steps passed")

    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        typer.echo(f"  Temp directory cleaned up.")
    else:
        typer.echo(f"  Outputs kept at: {tmp}")

    if passed < len(results):
        raise typer.Exit(1)
    typer.echo("\n  All checks passed — safe to run the full pipeline.")


if __name__ == "__main__":
    app()
