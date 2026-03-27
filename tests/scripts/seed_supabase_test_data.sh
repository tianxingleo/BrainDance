#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROFILE="minimal"
PHASE=""
DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
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

bd_require_local_supabase
bd_psql -f "$FIXTURE_FILE"

if [[ "$PROFILE" == "minimal" || "$PROFILE" == "agent" ]]; then
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' EXIT
  printf 'ply placeholder for %s\n' "$PROFILE" > "$temp_dir/point_cloud.ply"
  printf '{"poses":[{"image":"it_frame_001.png"}]}\n' > "$temp_dir/webgl_poses.json"
  printf 'preview placeholder for %s\n' "$PROFILE" > "$temp_dir/preview.txt"

  if [[ "$PROFILE" == "minimal" ]]; then
    base_prefix="it_user_a/it_minimal_scene_001/output"
    bd_upload_storage_object "braindance-assets" "${base_prefix}/point_cloud.ply" "$temp_dir/point_cloud.ply" "application/octet-stream"
    bd_upload_storage_object "braindance-assets" "${base_prefix}/webgl_poses.json" "$temp_dir/webgl_poses.json" "application/json"
    bd_upload_storage_object "braindance-assets" "${base_prefix}/preview.txt" "$temp_dir/preview.txt" "text/plain"
  else
    base_prefix="it_user_a/it_agent_scene_001/output"
    bd_upload_storage_object "braindance-assets" "${base_prefix}/point_cloud.ply" "$temp_dir/point_cloud.ply" "application/octet-stream"
    bd_upload_storage_object "braindance-assets" "${base_prefix}/preview.txt" "$temp_dir/preview.txt" "text/plain"
  fi
fi

echo "[seed] applied profile=$PROFILE"
