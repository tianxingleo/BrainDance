#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

CASES_FILE="${1:-ai_engine/finetune_qwen3/data/real_chain_debug_cases_part12.json}"
LOG_DIR="${2:-ai_engine/finetune_qwen3/logs}"
RUN_SCRIPT="ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh"

labels=(
  "round3"
  "round4_patch"
  "round4_1_patch_mixed"
)

adapters=(
  "ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3"
  "ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_patch"
  "ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed"
)

mkdir -p "$LOG_DIR"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  adapter_path="${adapters[$index]}"

  echo "[Part12] running ${label} with cases ${CASES_FILE}"
  bash "$RUN_SCRIPT" lora_round3 \
    --adapter_path "$adapter_path" \
    --cases_file "$CASES_FILE" \
    --output_file "${LOG_DIR}/real_chain_part12_${label}_cases.jsonl" \
    --summary_file "${LOG_DIR}/real_chain_part12_${label}_summary.json" \
    --overwrite_output
done
