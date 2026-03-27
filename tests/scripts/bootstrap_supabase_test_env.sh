#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPABASE_DIR="$ROOT_DIR/supabase"
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

echo "[bootstrap] TODO: 检查 braindance-assets / braindance-models bucket"
echo "[bootstrap] TODO: 校验 .env.test / dart-defines / Edge Function 可用性"
