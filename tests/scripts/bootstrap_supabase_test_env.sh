#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"

echo "[bootstrap] root=$ROOT_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[bootstrap] dry-run enabled, skip supabase start"
  echo "[bootstrap] TODO: 检查 braindance-assets / braindance-models bucket"
  echo "[bootstrap] TODO: 校验 .env.test / dart-defines / Edge Function 可用性"
  exit 0
fi

echo "[bootstrap] starting local supabase stack"
(
  cd "$SUPABASE_DIR"
  supabase start
)

bd_require_local_supabase
bd_psql <<'SQL'
insert into storage.buckets (id, name, public)
values
  ('braindance-assets', 'braindance-assets', true),
  ('braindance-models', 'braindance-models', true)
on conflict (id) do nothing;
SQL

echo "[bootstrap] ensured buckets: braindance-assets, braindance-models"
echo "[bootstrap] api_url=$(bd_api_url)"
echo "[bootstrap] db_url=$(bd_db_url)"
