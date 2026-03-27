#!/usr/bin/env bash
set -euo pipefail

MODE="local"
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "[suite] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/tests/output"

echo "[suite] mode=$MODE root=$ROOT_DIR"
mkdir -p "$OUTPUT_DIR/flutter" "$OUTPUT_DIR/edge" "$OUTPUT_DIR/sql" "$OUTPUT_DIR/storage"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[suite] dry-run enabled, downstream scripts will run in no-op mode"
fi
"$ROOT_DIR/tests/scripts/cleanup_supabase_test_data.sh" || true
"$ROOT_DIR/tests/scripts/bootstrap_supabase_test_env.sh"
"$ROOT_DIR/tests/scripts/seed_supabase_test_data.sh" --profile minimal
"$ROOT_DIR/tests/scripts/seed_supabase_test_data.sh" --profile realtime
"$ROOT_DIR/tests/scripts/seed_supabase_test_data.sh" --profile agent

for group in auth task recall realtime community local_ai edge; do
  "$ROOT_DIR/tests/scripts/run_flutter_integration_tests.sh" --group "$group" || true
done

for target in search-models agent-recall confirm-text-image; do
  "$ROOT_DIR/tests/scripts/run_edge_function_smoke_tests.sh" "$target" || true
done

echo "[suite] TODO: 汇总测试报告并输出到 tests/output/"
