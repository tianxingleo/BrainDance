#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

ADAPTER_PATH="${1:-ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/run_smoke_eval.py \
  --model_name "Qwen/Qwen3-1.7B" \
  --adapter_path "$ADAPTER_PATH"
