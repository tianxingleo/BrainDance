#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/train_lora_sft.py \
  --model_name "Qwen/Qwen3-1.7B" \
  --train_file "ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl" \
  --val_file "ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl" \
  --output_dir "ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3" \
  --cutoff_len 1536 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
