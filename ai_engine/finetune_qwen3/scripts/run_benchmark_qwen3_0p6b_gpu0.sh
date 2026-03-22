#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

ADAPTER_PATH="${1:-ai_engine/finetune_qwen3/outputs/qwen3_0p6b_lora_sft_round1}"
OUTPUT_FILE="${2:-ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_round1_gpu0.json}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/evaluate_benchmark.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --adapter_path "$ADAPTER_PATH" \
  --benchmark_file "ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl" \
  --output_file "$OUTPUT_FILE"
