#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_common.sh"

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

bd_require_local_supabase

HTTP_CODE="$(
  curl -sS \
    -o "$OUTPUT_FILE" \
    -w "%{http_code}" \
    -X POST "$(bd_api_url)/functions/v1/search-models" \
    -H "Authorization: Bearer $(bd_service_role_key)" \
    -H "Content-Type: application/json" \
    -d '{}'
)"

if [[ "$HTTP_CODE" != "400" ]]; then
  echo "[http-search-models] unexpected status=$HTTP_CODE" >&2
  cat "$OUTPUT_FILE" >&2
  exit 1
fi

if ! grep -q '"success":false' "$OUTPUT_FILE"; then
  echo "[http-search-models] response missing success=false" >&2
  cat "$OUTPUT_FILE" >&2
  exit 1
fi

if ! grep -q "query" "$OUTPUT_FILE"; then
  echo "[http-search-models] response missing query validation error" >&2
  cat "$OUTPUT_FILE" >&2
  exit 1
fi

echo "[http-search-models] wrote $OUTPUT_FILE status=$HTTP_CODE"
