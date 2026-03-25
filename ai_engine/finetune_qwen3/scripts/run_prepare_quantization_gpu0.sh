#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MERGED_MODEL_DIR="${1:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged}"
OUTPUT_DIR="${2:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized}"
LLAMA_CPP_DIR="${3:-}"
QUANT_TYPE="${4:-Q4_K_M}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py \
  --merged_model_dir "$MERGED_MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --llama_cpp_dir "$LLAMA_CPP_DIR" \
  --quant_type "$QUANT_TYPE"
