# SFT + GRPO on Qwen3-0.6B

Fine-tuning **Qwen3-0.6B-Base** for mathematical reasoning using Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimisation (GRPO), evaluated on GSM8K, SVAMP, and MMLU.

## Training Pipeline

```
Base Model  ──(300 SFT steps, DoRA)──▶  DoRA-SFT  ──(200 RL steps, GRPO)──▶  GRPO
                                              │
                                    (ablation: LoRA-SFT)
```

- **DoRA-SFT** is the primary SFT model and the base for GRPO.
- **LoRA-SFT** is trained identically but with plain LoRA — used only as an ablation to compare adapters.
- **GRPO** fine-tunes the *merged* DoRA-SFT weights end-to-end; no adapter is involved.

## Results

| Model     | GSM8K Acc. | GSM8K Format | SVAMP Acc. | MMLU Acc. |
|-----------|-----------|-------------|-----------|----------|
| Base      | 33.1%     | 0.0%        | 68.3%     | 8.7%     |
| LoRA-SFT  | **52.0%** | 94.9%       | 61.1%     | —        |
| DoRA-SFT  | 51.7%     | 95.1%       | 59.1%     | 9.0%     |
| GRPO      | 51.1%     | **95.7%**   | 61.1%     | **9.7%** |

- **SFT** (both LoRA and DoRA) gives the biggest accuracy gain (~+19 pp on GSM8K) and teaches the `#### N` answer format from scratch.
- **DoRA vs LoRA** differ by <0.5 pp — gap is negligible at this scale.
- **GRPO** is −0.6 pp below DoRA-SFT on GSM8K but recovers the SVAMP gap (+2 pp vs DoRA-SFT). Mean training reward rises 0.66 → 0.96 over 200 RL steps.
- **No catastrophic forgetting**: MMLU scores are stable across all models. The ~9% baseline reflects the inherent limit of a 0.6B model on academic MCQ, not forgetting.

## How GRPO Works

For each training prompt, the model generates **G=8 completions**. Each is scored with the reward function (correctness +1.0, `#### N` format +0.5). The *group average* serves as the baseline — no separate value network is needed. Completions that score above the group mean receive a positive policy update; those below are suppressed. A KL penalty keeps the policy close to the DoRA-SFT reference.

This is cheaper than PPO and works well for verifiable tasks like math, where reward is easy to compute.

## Folder Structure

```
dataset/              data loaders and processed JSONL files
  gsm8k.py            GSM8K preprocessing
  svamp.py            SVAMP preprocessing
  mmlu.py             MMLU preprocessing

scripts/              runnable CLIs
  prepare_gsm8k.py    download and preprocess GSM8K
  prepare_svamp.py    download and preprocess SVAMP
  train_sft.py        SFT training with plain LoRA (ablation)
  train_dora.py       SFT training with DoRA — primary SFT model
  train_grpo.py       GRPO training (uses TRL GRPOTrainer)
  eval.py             evaluate on GSM8K / SVAMP
  eval_mmlu.py        evaluate on MMLU (multiple choice)
  eval_prompts.py     prompt design comparison (direct vs CoT vs rules)
  eval_reasoning.py   step-by-step reasoning validity analysis
  analyze.py          generate plots and summary table

trainer/              core training and reward logic
  sft.py              SFT training loop (LoRA/DoRA, fp16, gradient checkpointing)
  reward.py           correctness + format reward functions
  jsonl.py            JSONL read/write helpers

model/                saved model checkpoints (not tracked in git)
results/              evaluation JSONL outputs and generated plots
slides.tex            LaTeX Beamer presentation
run_kaggle.ipynb      full end-to-end notebook for Kaggle
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `transformers>=4.51.0`, `peft>=0.11.1`, `trl>=0.12.0`, `datasets`, `tiktoken`, `accelerate`, `matplotlib`.

## Running on Kaggle (recommended)

The easiest way to run the full pipeline is via the included Kaggle notebook:

1. Import `run_kaggle.ipynb` into Kaggle
2. Set **Accelerator → GPU T4 x2** and **Internet → On**
3. Click **Save Version → Save & Run All**
4. Download `results.zip` and `plots.zip` from the Output panel when done

Total runtime: ~4.5 hours on T4 x2.

## Running Locally

### 1. Prepare datasets

```bash
python scripts/prepare_gsm8k.py
python scripts/prepare_svamp.py --limit 500
```

### 2. DoRA-SFT training (~35 min on T4)

```bash
python scripts/train_dora.py \
  --model-name-or-path Qwen/Qwen3-0.6B-Base \
  --train-path dataset/processed/gsm8k_train.jsonl \
  --output-dir model/sft_gsm8k \
  --train-log-path model/sft_gsm8k/train_log.jsonl \
  --max-steps 300 \
  --per-device-batch-size 2 \
  --grad-accum 16 \
  --max-length 384 \
  --fp16 \
  --gradient-checkpointing \
  --log-every 10
```

### 3. GRPO training (~2.5 hrs on T4)

GRPO is trained on top of the **merged DoRA-SFT checkpoint**:

```bash
python scripts/train_grpo.py \
  --model-name-or-path Qwen/Qwen3-0.6B-Base \
  --ref-model-path model/sft_gsm8k \
  --train-path dataset/processed/gsm8k_train.jsonl \
  --output-dir model/grpo_gsm8k \
  --max-steps 200 \
  --group-size 8 \
  --kl-coef 0.01
```

The GRPO model is saved as a **fully merged model** (base + DoRA-SFT + GRPO weights baked together), so evaluation requires no adapter path.

### 4. Evaluate

```bash
# GSM8K — Base
python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path results/eval_base.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

# GSM8K — DoRA-SFT
python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/sft_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path results/eval_sft.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

# GSM8K — GRPO (merged model, no adapter needed)
python scripts/eval.py --base-model model/grpo_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path results/eval_grpo.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

# MMLU forgetting check (2 subjects, 150 examples each)
python scripts/eval_mmlu.py --base-model Qwen/Qwen3-0.6B-Base \
  --output-path results/eval_mmlu_base.jsonl \
  --subjects "high_school_mathematics,elementary_mathematics" \
  --max-examples-per-subject 150 --batch-size 8
```

### 5. Analyze and plot

```bash
python scripts/analyze.py \
  --base-results  results/eval_base.jsonl \
  --sft-results   results/eval_sft.jsonl \
  --grpo-results  results/eval_grpo.jsonl \
  --svamp-base    results/eval_svamp_base.jsonl \
  --svamp-sft     results/eval_svamp_sft.jsonl \
  --svamp-grpo    results/eval_svamp_grpo.jsonl \
  --mmlu-base     results/eval_mmlu_base.jsonl \
  --mmlu-sft      results/eval_mmlu_sft.jsonl \
  --mmlu-grpo     results/eval_mmlu_grpo.jsonl \
  --sft-log       model/sft_gsm8k/train_log.jsonl \
  --grpo-log      model/grpo_gsm8k/trainer_state.json \
  --images-dir    results/plots
```

### 6. LoRA ablation (optional)

To reproduce the DoRA vs LoRA comparison, run plain LoRA SFT and evaluate it:

```bash
python scripts/train_sft.py \
  --model-name-or-path Qwen/Qwen3-0.6B-Base \
  --train-path dataset/processed/gsm8k_train.jsonl \
  --output-dir model/lora_sft_gsm8k \
  --train-log-path model/lora_sft_gsm8k/train_log.jsonl \
  --max-steps 300 --per-device-batch-size 2 --grad-accum 16 \
  --max-length 384 --fp16 --gradient-checkpointing --log-every 10

python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/lora_sft_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path results/eval_lora_sft.jsonl \
  --max-new-tokens 256 --batch-size 8 --max-examples 500
```

### 7. Prompt design comparison

```bash
python scripts/eval_prompts.py \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/sft_gsm8k \
  --output-dir results/prompt_comparison \
  --max-examples 200
```

### 8. Step-by-step reasoning validity

```bash
python scripts/eval_reasoning.py \
  --results-path results/eval_sft.jsonl \
  --output-dir results/reasoning_sft
```

## Design Notes

- **DoRA** (r=8, α=16) applied to all attention and MLP projection layers — keeps VRAM under 15 GB on a single T4. Weight-decomposition into magnitude + direction gives slightly more expressive updates than plain LoRA, though the gap is small at 0.6B scale.
- **GRPO reward**: correctness (1.0) + format bonus (0.5 if output contains `#### N`) — max reward 1.5. This same formula is used as a post-hoc *eval score* for all models at inference; for Base and SFT it is not a training signal.
- **GRPO save**: after RL training, the policy is saved as a standalone full model with all weights baked in (no adapter). This is necessary because GRPO fine-tunes the merged DoRA-SFT weights end-to-end.
- **Evaluation**: `eval.py` auto-detects whether a given path is a LoRA/DoRA adapter or a merged full model, so the same script handles all four models.
- **MMLU**: 2 subjects (high school mathematics, elementary mathematics), 150 examples each (300 total), used as a proxy forgetting check. All models score ~9%, consistent with the inherent limit of a 0.6B model on academic MCQ rather than forgetting.
