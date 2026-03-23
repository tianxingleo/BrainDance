#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="${1:-ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_round1}"
TRAIN_FILE="${2:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl}"
VAL_FILE="${3:-ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl}"
LEARNING_RATE="${LEARNING_RATE:-${4:-1e-5}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${5:-1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${6:-2}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${7:-2}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${8:-8}}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/train_full_sft.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --cutoff_len 1536 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --logging_steps 10
