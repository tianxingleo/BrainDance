#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "Unable to initialize conda." >&2
  exit 1
fi

conda activate qwen3_ft

python ai_engine/finetune_qwen3/scripts/backfill_interactive_session_fields.py \
  --input ai_engine/finetune_qwen3/logs/interactive_sessions \
  "$@"

python ai_engine/finetune_qwen3/scripts/summarize_interactive_debug_routes.py \
  --input ai_engine/finetune_qwen3/logs/interactive_sessions
