#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${BRAINDANCE_IT_DRY_RUN:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_FILE="$ROOT_DIR/tests/fixtures/cleanup_integration.sql"

echo "[cleanup] fixture=$FIXTURE_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[cleanup] dry-run enabled, skip cleanup mutations"
  exit 0
fi

echo "[cleanup] TODO: 删除 it_* 前缀测试数据"
echo "[cleanup] TODO: 清理 Storage 测试目录"
echo "[cleanup] TODO: 恢复被重命名或改元数据的测试样本"
