#!/usr/bin/env bash
set -euo pipefail

GROUP=""
ENV_NAME="rls"
FAULT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      GROUP="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --fault)
      FAULT="$2"
      shift 2
      ;;
    *)
      echo "[flutter-it] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$GROUP" ]]; then
  echo "usage: $0 --group <auth|task|recall|realtime|community|edge|local_ai> [--env <rls|admin>] [--fault <name>]" >&2
  exit 1
fi

case "$GROUP" in
  auth) TARGET="integration_test/auth_flow_test.dart" ;;
  task) TARGET="integration_test/task_submission_test.dart" ;;
  recall) TARGET="integration_test/recall_flow_test.dart" ;;
  realtime) TARGET="integration_test/realtime_flow_test.dart" ;;
  community) TARGET="integration_test/community_flow_test.dart" ;;
  edge) TARGET="integration_test/edge_function_flow_test.dart" ;;
  local_ai) TARGET="integration_test/local_ai_catalog_test.dart" ;;
  *)
    echo "[flutter-it] unsupported group: $GROUP" >&2
    exit 1
    ;;
esac

echo "[flutter-it] group=$GROUP env=$ENV_NAME fault=$FAULT target=$TARGET"
echo "[flutter-it] TODO: 注入 dart-defines / .env.test 并执行 flutter test $TARGET"
