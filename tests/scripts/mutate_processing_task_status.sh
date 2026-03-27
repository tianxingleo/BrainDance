#!/usr/bin/env bash
set -euo pipefail

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

echo "[mutate-task] TODO: 通过 psql 或 service-role API 更新 processing_tasks"
