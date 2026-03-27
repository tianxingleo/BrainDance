#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_common.sh"

OUTPUT_FILE="${1:-}"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
STREAM_FORMAT="${BRAINDANCE_IT_AGENT_STREAM_FORMAT:-ndjson}"

if [[ -z "$OUTPUT_FILE" ]]; then
  echo "usage: $0 <output-file>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

if [[ "$DRY_RUN" == "1" ]]; then
  cat > "$OUTPUT_FILE" <<'EOF'
{"event":"status","data":{"target":"agent-recall","mode":"dry-run","status":"skipped"}}
EOF
  echo "[http-agent-recall] dry-run wrote $OUTPUT_FILE"
  exit 0
fi

bd_require_local_supabase

case "$STREAM_FORMAT" in
  ndjson)
    HTTP_CODE="$(
      curl -sS -N \
        -o "$OUTPUT_FILE" \
        -w "%{http_code}" \
        -X POST "$(bd_api_url)/functions/v1/agent-recall?stream=1" \
        -H "Authorization: Bearer $(bd_service_role_key)" \
        -H "Content-Type: application/json" \
        -H "Accept: application/x-ndjson" \
        -d '{"query":"你是谁"}'
    )"

    if [[ "$HTTP_CODE" != "200" ]]; then
      echo "[http-agent-recall] unexpected status=$HTTP_CODE" >&2
      sed -n '1,40p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '"event":"ping"' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] stream missing ping event" >&2
      sed -n '1,40p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '"event":"status"' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] stream missing status event" >&2
      sed -n '1,40p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '"event":"done"' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] stream missing done event" >&2
      sed -n '1,80p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q 'BrainDance 的空间记忆智能管理助手' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] stream missing expected assistant answer" >&2
      sed -n '1,80p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi
    ;;
  sse)
    HTTP_CODE="$(
      curl -sS -N \
        -o "$OUTPUT_FILE" \
        -w "%{http_code}" \
        -X POST "$(bd_api_url)/functions/v1/agent-recall?stream=1" \
        -H "Authorization: Bearer $(bd_service_role_key)" \
        -H "Content-Type: application/json" \
        -H "Accept: text/event-stream" \
        -d '{"query":"你是谁"}'
    )"

    if [[ "$HTTP_CODE" != "200" ]]; then
      echo "[http-agent-recall] unexpected status=$HTTP_CODE" >&2
      sed -n '1,60p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '^event: ping$' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] sse missing ping event" >&2
      sed -n '1,60p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '^event: status$' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] sse missing status event" >&2
      sed -n '1,60p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q '^event: done$' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] sse missing done event" >&2
      sed -n '1,120p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi

    if ! grep -q 'BrainDance 的空间记忆智能管理助手' "$OUTPUT_FILE"; then
      echo "[http-agent-recall] sse missing expected assistant answer" >&2
      sed -n '1,120p' "$OUTPUT_FILE" >&2 || true
      exit 1
    fi
    ;;
  *)
    echo "[http-agent-recall] unsupported stream format: $STREAM_FORMAT" >&2
    exit 1
    ;;
esac

echo "[http-agent-recall] wrote $OUTPUT_FILE status=$HTTP_CODE format=$STREAM_FORMAT"
