#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
FIXTURE_FILE="$ROOT_DIR/tests/fixtures/cleanup_integration.sql"

echo "[cleanup] fixture=$FIXTURE_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[cleanup] dry-run enabled, skip cleanup mutations"
  exit 0
fi

bd_require_local_supabase
bd_delete_storage_prefix "braindance-assets" "it_"
bd_delete_storage_prefix "braindance-models" "it_"
bd_psql -f "$FIXTURE_FILE"
echo "[cleanup] removed integration rows and storage metadata for prefix it_"
