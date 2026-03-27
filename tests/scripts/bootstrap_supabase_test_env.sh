#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPABASE_DIR="$ROOT_DIR/supabase"

echo "[bootstrap] root=$ROOT_DIR"
echo "[bootstrap] starting local supabase stack"
(
  cd "$SUPABASE_DIR"
  supabase start
)

echo "[bootstrap] TODO: 检查 braindance-assets / braindance-models bucket"
echo "[bootstrap] TODO: 校验 .env.test / dart-defines / Edge Function 可用性"
