#!/usr/bin/env bash
set -euo pipefail

PROFILE="minimal"
PHASE=""
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURES_DIR="$ROOT_DIR/tests/fixtures"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    *)
      echo "[seed] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$PROFILE" in
  minimal|realtime|agent)
    ;;
  *)
    echo "[seed] unsupported profile: $PROFILE" >&2
    exit 1
    ;;
esac

echo "[seed] profile=$PROFILE phase=$PHASE"
case "$PROFILE" in
  minimal) FIXTURE_FILE="$FIXTURES_DIR/supabase_seed_minimal.sql" ;;
  realtime) FIXTURE_FILE="$FIXTURES_DIR/supabase_seed_realtime.sql" ;;
  agent) FIXTURE_FILE="$FIXTURES_DIR/supabase_seed_agent.sql" ;;
esac
echo "[seed] fixture=$FIXTURE_FILE"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[seed] dry-run enabled, skip data mutations"
  exit 0
fi

echo "[seed] TODO: 创建测试用户、插入 processing_tasks/model_assets/community_posts/memory_poses"
echo "[seed] TODO: 上传 braindance-assets / braindance-models 测试文件"
