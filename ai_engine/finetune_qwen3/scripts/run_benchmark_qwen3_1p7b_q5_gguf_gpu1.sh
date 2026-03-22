#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${1:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/model-f16-q5_k_m.gguf}"
OUTPUT_FILE="${2:-ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_q5_gguf_round4_1_patch_mixed_gpu1.json}"
LLAMA_CLI_PATH="${3:-ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda/bin/llama-cli}"
DEVICE="${4:-CUDA0}"

conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/evaluate_gguf_benchmark.py \
  --model_path "$MODEL_PATH" \
  --llama_cli_path "$LLAMA_CLI_PATH" \
  --benchmark_file "ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl" \
  --output_file "$OUTPUT_FILE" \
  --device "$DEVICE" \
  --batch_size 128 \
  --ubatch_size 64 \
  --threads 8 \
  --retries 2
