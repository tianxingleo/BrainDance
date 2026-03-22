#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MERGED_MODEL_DIR="${1:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0}"
OUTPUT_FILE="${2:-ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_merged_round4_1_patch_mixed_gpu1.json}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/evaluate_benchmark.py \
  --model_name "$MERGED_MODEL_DIR" \
  --benchmark_file "ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl" \
  --output_file "$OUTPUT_FILE"
