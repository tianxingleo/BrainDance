#!/usr/bin/env bash
set -euo pipefail

# 自动化创建 PR 的脚本（不会自动 push，除非你显式允许）
# 使用方法：
#  1) 在本机检查并确认改动： git status && git --no-pager diff --stat
#  2) 如果确认将要推送并创建 PR，请以下面方式运行：
#     ALLOW_PUSH=true ./create_pr.sh
#    （脚本会在 ALLOW_PUSH!=true 时退出，避免误推）

BRANCH=$(git branch --show-current)
if [ -z "$BRANCH" ]; then
  echo "无法获取当前分支名，请在仓库根目录运行此脚本。"
  exit 1
fi

if [ "${ALLOW_PUSH:-false}" != "true" ]; then
  echo "安全开关检测：未设置 ALLOW_PUSH=true，已停止执行 push/pr 创建。"
  echo "若要推送并创建 PR，请再次运行： ALLOW_PUSH=true ./create_pr.sh"
  echo
  echo "下面是将要执行的命令（仅作参考）："
  echo "  git push -u origin $BRANCH"
  echo "  gh pr create --title \"feat(3dgs): 统一 BasePipeline，集中 RAG 分析与上传逻辑并修复 RAG 写入目标\" --body \"$(sed -n '1,300p' PR_BODY.md | sed 's/"/'\\"'/g')\" --base main --head $BRANCH"
  exit 0
fi

echo "ALLOW_PUSH=true 已设置，开始推送分支并创建 PR（谨慎执行）"
echo "当前分支: $BRANCH"

echo "1/2 推送分支到远端..."
git push -u origin "$BRANCH"

echo "2/2 使用 gh 创建 PR..."
PR_TITLE="feat(3dgs): 统一 BasePipeline，集中 RAG 分析与上传逻辑并修复 RAG 写入目标"
PR_BODY=$(sed -n '1,1000p' PR_BODY.md)
gh pr create --title "$PR_TITLE" --body "$PR_BODY" --base main --head "$BRANCH"

echo "PR 创建完成。"
