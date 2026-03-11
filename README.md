# SFT + GRPO on Qwen3-0.6B

Fine-tuning **Qwen3-0.6B-Base** for mathematical reasoning using Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimisation (GRPO), evaluated on GSM8K, SVAMP, and MMLU.

## Results

| Model | GSM8K Acc. | GSM8K Format | SVAMP Acc. | MMLU Acc. |
|-------|-----------|-------------|-----------|----------|
| Base  | 37.2%     | 0.0%        | 71.4%     | 8.7%     |
| SFT   | **51.6%** | 95.4%       | 62.2%     | 8.7%     |
| GRPO  | 49.2%     | **96.0%**   | 61.8%     | 9.0%     |

- **SFT** gives the biggest accuracy gain (+14 pp on GSM8K) and teaches the model the `#### N` answer format.
- **GRPO** consolidates the SFT baseline, with reward rising from 0.86 → 1.06 over 200 RL steps and format rate improving further.
- **No catastrophic forgetting** observed on MMLU — scores are identical before and after fine-tuning.

## Folder Structure

```
dataset/        data loaders and processed JSONL files
  gsm8k.py      GSM8K preprocessing
  svamp.py      SVAMP preprocessing
  mmlu.py       MMLU preprocessing

scripts/        runnable CLIs
  prepare_gsm8k.py    download and preprocess GSM8K
  prepare_svamp.py    download and preprocess SVAMP
  train_sft.py        SFT training (LoRA)
  train_dora.py       DoRA training (same interface as train_sft.py)
  train_grpo.py       GRPO training (uses TRL GRPOTrainer)
  eval.py             evaluate on GSM8K / SVAMP
  eval_mmlu.py        evaluate on MMLU (multiple choice)
  eval_prompts.py     prompt design comparison (direct vs CoT vs rules)
  eval_reasoning.py   step-by-step reasoning validity analysis
  analyze.py          generate plots and summary table

trainer/        core training and reward logic
  sft.py        SFT training loop (LoRA, fp16, gradient checkpointing)
  reward.py     correctness + format reward functions
  jsonl.py      JSONL read/write helpers

model/          saved model checkpoints (not tracked in git)
images/         generated plots (not tracked in git)
slides.tex      LaTeX Beamer presentation
run_kaggle.ipynb  full end-to-end notebook for Kaggle
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

### 2. SFT training (~35 min on T4)

```bash
python scripts/train_sft.py \
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

The GRPO model is saved as a **fully merged model** (base + SFT + GRPO weights baked together), so evaluation requires no adapter path.

### 4. Evaluate

```bash
# GSM8K
python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path model/eval_base.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/sft_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path model/eval_sft.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

python scripts/eval.py --base-model model/grpo_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path model/eval_grpo.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500

# MMLU (forgetting check)
python scripts/eval_mmlu.py --base-model Qwen/Qwen3-0.6B-Base \
  --output-path model/eval_mmlu_base.jsonl \
  --subjects "high_school_mathematics,elementary_mathematics,world_history" \
  --max-examples-per-subject 150 --batch-size 8
```

### 5. Analyze and plot

```bash
python scripts/analyze.py \
  --base-results  model/eval_base.jsonl \
  --sft-results   model/eval_sft.jsonl \
  --grpo-results  model/eval_grpo.jsonl \
  --svamp-base    model/eval_svamp_base.jsonl \
  --svamp-sft     model/eval_svamp_sft.jsonl \
  --svamp-grpo    model/eval_svamp_grpo.jsonl \
  --mmlu-base     model/eval_mmlu_base.jsonl \
  --mmlu-sft      model/eval_mmlu_sft.jsonl \
  --mmlu-grpo     model/eval_mmlu_grpo.jsonl \
  --sft-log       model/sft_gsm8k/train_log.jsonl \
  --grpo-log      model/grpo_gsm8k/trainer_state.json \
  --images-dir    images
```

### 6. DoRA comparison (~35 min on T4)

```bash
python scripts/train_dora.py \
  --model-name-or-path Qwen/Qwen3-0.6B-Base \
  --train-path dataset/processed/gsm8k_train.jsonl \
  --output-dir model/dora_gsm8k \
  --train-log-path model/dora_gsm8k/train_log.jsonl \
  --max-steps 300 \
  --per-device-batch-size 2 \
  --grad-accum 16 \
  --max-length 384 \
  --fp16 \
  --gradient-checkpointing \
  --log-every 10

python scripts/eval.py --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/dora_gsm8k \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --output-path model/eval_dora.jsonl --max-new-tokens 256 --batch-size 8 --max-examples 500
```

### 7. Prompt design comparison

```bash
python scripts/eval_prompts.py \
  --test-path dataset/processed/gsm8k_test.jsonl \
  --base-model Qwen/Qwen3-0.6B-Base \
  --adapter-path model/sft_gsm8k \
  --output-dir model/prompt_comparison \
  --max-examples 200
```

### 8. Step-by-step reasoning validity

```bash
python scripts/eval_reasoning.py \
  --results-path model/eval_sft.jsonl \
  --output-dir model/reasoning_sft
```

## Design Notes

- **LoRA** (r=8, α=16) applied to all attention and MLP projection layers — keeps VRAM under 15 GB on a single T4.
- **GRPO reward**: correctness (1.0) + format bonus (0.5 if output contains `#### N`) — max reward 1.5.
- **GRPO save**: after training, the LoRA is merged into the SFT-initialised weights and saved as a standalone full model. This is necessary because the GRPO LoRA learns residuals relative to SFT weights, not the raw base.
- **Evaluation**: `eval.py` auto-detects whether a given path is a LoRA adapter or a merged full model, so `--adapter-path` and `--base-model` both work for the GRPO model.
- **MMLU**: uses 3 subjects (high school math, elementary math, world history) as a proxy forgetting check. The model scores ~9% on all three regardless of fine-tuning — this reflects the size limit of a 0.6B model on academic MCQ, not catastrophic forgetting.

