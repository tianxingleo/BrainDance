#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"

if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <search-models|agent-recall|confirm-text-image|text-to-image|agent-preview-execute>" >&2
  exit 1
fi

echo "[edge-smoke] target=$TARGET"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[edge-smoke] dry-run enabled, skip HTTP calls"
  exit 0
fi

echo "[edge-smoke] TODO: 调用 tests/http 或等效请求脚本并保存原始响应"
