#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

TASK_ID=""
STATUS=""
APPEND_LOG=""
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)
      TASK_ID="$2"
      shift 2
      ;;
    --status)
      STATUS="$2"
      shift 2
      ;;
    --append-log)
      APPEND_LOG="$2"
      shift 2
      ;;
    *)
      echo "[mutate-task] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TASK_ID" || -z "$STATUS" ]]; then
  echo "usage: $0 --task-id <uuid> --status <status> [--append-log <text>]" >&2
  exit 1
fi

echo "[mutate-task] task_id=$TASK_ID status=$STATUS append_log=$APPEND_LOG"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[mutate-task] dry-run enabled, skip task status mutation"
  exit 0
fi

bd_require_local_supabase

UPDATE_RESULT="$(
  bd_psql -At \
    -v task_id="$TASK_ID" \
    -v new_status="$STATUS" \
    -v append_log="$APPEND_LOG" <<'SQL'
with updated as (
  update public.processing_tasks
  set
    status = :'new_status',
    updated_at = now(),
    logs = case
      when :'append_log' = '' then coalesce(logs, '[]'::jsonb)
      else coalesce(logs, '[]'::jsonb) || jsonb_build_array(
        jsonb_build_object(
          'msg', :'append_log',
          'status', :'new_status',
          'source', 'integration_test'
        )
      )
    end
  where id = :'task_id'::uuid
  returning id, status, jsonb_array_length(coalesce(logs, '[]'::jsonb)) as log_count
)
select id::text || '|' || status || '|' || log_count::text from updated;
SQL
)"

if [[ -z "$UPDATE_RESULT" ]]; then
  echo "[mutate-task] task not found: $TASK_ID" >&2
  exit 1
fi

echo "[mutate-task] updated=$UPDATE_RESULT"
