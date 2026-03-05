# GSM8K SFT + GRPO (minimal project)

This repo contains a small, practical pipeline to:
- prepare **GSM8K** into prompt/target format
- run **SFT** (supervised fine-tuning)
- run **GRPO** (group-relative policy optimization) with a simple verified reward (final answer match)

## Folders
- `dataset/`: data code + saved jsonl files
- `scripts/`: runnable CLIs
- `trainer/`: training + reward logic
- `model/`: output checkpoints
- `images/`: plots (optional)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Prepare GSM8K

```bash
python scripts/prepare_gsm8k.py --out_dir dataset/processed
```

This creates:
- `dataset/processed/gsm8k_train.jsonl`
- `dataset/processed/gsm8k_test.jsonl`

## 2) SFT

Example (LoRA; small default model for convenience):

```bash
python scripts/train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_path dataset/processed/gsm8k_train.jsonl \
  --eval_path dataset/processed/gsm8k_test.jsonl \
  --output_dir model/sft_gsm8k \
  --max_steps 200 \
  --per_device_batch_size 1 \
  --grad_accum 16
```

## 3) GRPO

```bash
python scripts/train_grpo.py \
  --model_name_or_path model/sft_gsm8k \
  --train_path dataset/processed/gsm8k_train.jsonl \
  --eval_path dataset/processed/gsm8k_test.jsonl \
  --output_dir model/grpo_gsm8k \
  --outer_steps 50 \
  --prompts_per_step 8 \
  --group_size 4 \
  --max_new_tokens 128
```

## 4) Evaluate

```bash
python scripts/eval_gsm8k.py --model_name_or_path model/grpo_gsm8k --data_path dataset/processed/gsm8k_test.jsonl
```

## Notes
- GRPO here uses a **binary verified reward**: 1 if the extracted final answer matches GSM8K ground truth, else 0.
- This is a minimal educational implementation (single-process). For performance, use multi-GPU + faster generation backends.

