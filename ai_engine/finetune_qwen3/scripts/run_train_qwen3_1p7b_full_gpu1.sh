#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="${1:-ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_round1_gpu1}"
TRAIN_FILE="${2:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl}"
VAL_FILE="${3:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/train_full_sft.py \
  --model_name "Qwen/Qwen3-1.7B" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --cutoff_len 1536 \
  --num_train_epochs 1 \
  --learning_rate 8e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10
