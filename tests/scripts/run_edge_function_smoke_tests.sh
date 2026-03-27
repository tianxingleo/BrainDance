#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTTP_DIR="$ROOT_DIR/tests/http"
OUTPUT_DIR="$ROOT_DIR/tests/output/edge"

if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <search-models|agent-recall|confirm-text-image|text-to-image|agent-preview-execute>" >&2
  exit 1
fi

echo "[edge-smoke] target=$TARGET"
mkdir -p "$OUTPUT_DIR"

case "$TARGET" in
  search-models)
    SCRIPT="$HTTP_DIR/search_models_smoke.sh"
    OUTPUT_FILE="$OUTPUT_DIR/search_models_response.json"
    ;;
  agent-recall)
    SCRIPT="$HTTP_DIR/agent_recall_stream_smoke.sh"
    OUTPUT_FILE="$OUTPUT_DIR/agent_recall_stream.jsonl"
    ;;
  confirm-text-image)
    SCRIPT="$HTTP_DIR/confirm_text_image_smoke.sh"
    OUTPUT_FILE="$OUTPUT_DIR/confirm_text_image_response.json"
    ;;
  text-to-image|agent-preview-execute)
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[edge-smoke] dry-run enabled, skip specialized target=$TARGET"
      exit 0
    fi
    echo "[edge-smoke] TODO: 为 $TARGET 补充专用执行脚本"
    exit 0
    ;;
  *)
    echo "[edge-smoke] unsupported target: $TARGET" >&2
    exit 1
    ;;
esac

"$SCRIPT" "$OUTPUT_FILE"
