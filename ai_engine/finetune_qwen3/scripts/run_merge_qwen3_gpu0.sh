#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BASE_MODEL="${1:-Qwen/Qwen3-1.7B}"
ADAPTER_PATH="${2:-ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed}"
OUTPUT_DIR="${3:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype auto_bf16 \
  --device_map cpu \
  --safe_serialization
