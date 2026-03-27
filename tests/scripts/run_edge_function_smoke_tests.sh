#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <search-models|agent-recall|confirm-text-image|text-to-image|agent-preview-execute>" >&2
  exit 1
fi

echo "[edge-smoke] target=$TARGET"
echo "[edge-smoke] TODO: 调用 tests/http 或等效请求脚本并保存原始响应"
