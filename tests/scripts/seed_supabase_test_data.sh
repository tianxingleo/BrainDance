#!/usr/bin/env bash
set -euo pipefail

PROFILE="minimal"
PHASE=""

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

echo "[seed] profile=$PROFILE phase=$PHASE"
echo "[seed] TODO: 创建测试用户、插入 processing_tasks/model_assets/community_posts/memory_poses"
echo "[seed] TODO: 上传 braindance-assets / braindance-models 测试文件"
