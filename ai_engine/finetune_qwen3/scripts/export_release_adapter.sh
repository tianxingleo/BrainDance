#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ai_engine/finetune_qwen3/scripts/export_release_adapter.sh <source_output_dir> <release_name> [base_model]

Example:
  ai_engine/finetune_qwen3/scripts/export_release_adapter.sh \
    ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3 \
    qwen3_1p7b_braindance_round3 \
    Qwen/Qwen3-1.7B
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

SOURCE_DIR="${1%/}"
RELEASE_NAME="$2"
BASE_MODEL="${3:-Qwen/Qwen3-1.7B}"
TARGET_DIR="ai_engine/finetune_qwen3/releases/$RELEASE_NAME"

required_files=(
  "adapter_model.safetensors"
  "adapter_config.json"
)

optional_files=(
  "final_metrics.json"
  "training_spec.json"
  "README.md"
  "chat_template.jinja"
  "merge_metadata.json"
  "quantization_plan.json"
)

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

for file in "${required_files[@]}"; do
  if [[ ! -f "$SOURCE_DIR/$file" ]]; then
    echo "Missing required file: $SOURCE_DIR/$file" >&2
    exit 1
  fi
done

if [[ -e "$TARGET_DIR" ]] && find "$TARGET_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Target directory is not empty: $TARGET_DIR" >&2
  echo "Choose another release name or remove the existing directory first." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

for file in "${required_files[@]}" "${optional_files[@]}"; do
  if [[ -f "$SOURCE_DIR/$file" ]]; then
    cp "$SOURCE_DIR/$file" "$TARGET_DIR/$file"
  fi
done

cat > "$TARGET_DIR/release_metadata.json" <<EOF
{
  "release_name": "$RELEASE_NAME",
  "base_model": "$BASE_MODEL",
  "source_output_dir": "$SOURCE_DIR",
  "exported_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "notes": [
    "This release bundle contains the LoRA adapter and minimal metadata.",
    "Load the tokenizer from the base model unless you intentionally publish a tokenizer override."
  ]
}
EOF

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$TARGET_DIR"
    sha256sum \
      adapter_model.safetensors \
      adapter_config.json \
      release_metadata.json \
      $(for file in "${optional_files[@]}"; do [[ -f "$file" ]] && printf '%s ' "$file"; done) \
      > SHA256SUMS
  )
fi

echo "Exported release bundle to: $TARGET_DIR"
echo "Tracked by regular Git: adapter_config.json, metadata, metrics"
echo "Tracked by Git LFS: adapter_model.safetensors"
