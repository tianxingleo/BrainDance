#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-}"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"

if [[ -z "$OUTPUT_FILE" ]]; then
  echo "usage: $0 <output-file>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

if [[ "$DRY_RUN" == "1" ]]; then
  cat > "$OUTPUT_FILE" <<'EOF'
{"target":"search-models","mode":"dry-run","status":"skipped"}
EOF
  echo "[http-search-models] dry-run wrote $OUTPUT_FILE"
  exit 0
fi

echo "[http-search-models] TODO: 请求 search-models 并写入 $OUTPUT_FILE"
