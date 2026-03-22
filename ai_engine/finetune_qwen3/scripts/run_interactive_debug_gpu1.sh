#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Fallback for shells where `conda` is not pre-initialized.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "Unable to initialize conda." >&2
  exit 1
fi

conda activate qwen3_ft
exec python ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py "$@"
