#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${1:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/model-f16.gguf}"
OUTPUT_DIR="${2:-ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/imatrix}"
CORPUS_PATH="${3:-$OUTPUT_DIR/calibration_corpus_chat_256.txt}"
IMATRIX_PATH="${4:-$OUTPUT_DIR/imatrix_chat_256.gguf}"
CHUNKS="${5:-128}"
LLAMA_BIN_DIR="${6:-ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda/bin}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CORPUS_PATH" ]]; then
  conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/export_imatrix_corpus.py \
    --inputs \
      ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl \
      ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark_strict_no_leak_ood_20260322_v3.jsonl \
      ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl \
    --output "$CORPUS_PATH" \
    --max_records 256 \
    --mode chat
fi

conda run -n qwen3_ft "$LLAMA_BIN_DIR/llama-imatrix" \
  -m "$MODEL_PATH" \
  -f "$CORPUS_PATH" \
  -o "$IMATRIX_PATH" \
  --output-frequency 16 \
  --chunks "$CHUNKS" \
  --no-ppl \
  --device CUDA0 \
  -ngl 999

conda run -n qwen3_ft "$LLAMA_BIN_DIR/llama-quantize" \
  --imatrix "$IMATRIX_PATH" \
  "$MODEL_PATH" \
  "$OUTPUT_DIR/model-f16-q4_k_m-imatrix.gguf" \
  Q4_K_M

conda run -n qwen3_ft "$LLAMA_BIN_DIR/llama-quantize" \
  --imatrix "$IMATRIX_PATH" \
  "$MODEL_PATH" \
  "$OUTPUT_DIR/model-f16-q5_k_m-imatrix.gguf" \
  Q5_K_M
