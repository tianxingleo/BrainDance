#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="${1:-ai_engine/finetune_qwen3/outputs/qwen3_0p6b_lora_sft_round1}"
TRAIN_FILE="${2:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl}"
VAL_FILE="${3:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/train_lora_sft.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --cutoff_len 1536 \
  --num_train_epochs 2 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
