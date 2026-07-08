#!/bin/bash

# 0. 检查是否输入了说明文字，如果没有则默认使用时间戳
if [ -z "$1" ]; then
  MIGRATION_NAME="update_$(date +%Y%m%d_%H%M%S)"
else
  MIGRATION_NAME="$1"
fi

echo "🚀 开始一键同步流程..."

# 1. 【数据库结构】检测插件(vector/deno)、表结构的变动，生成迁移文件
echo "📜 1. 正在生成数据库结构差异 (Migrations)..."
# 注意：如果没有任何改动，Supabase可能会警告，这是正常的
supabase db diff -f "$MIGRATION_NAME"

# 2. 【数据备份】将当前数据库的数据导出为种子数据 (Seed)
echo "🌱 2. 正在更新测试数据 (seed.sql)..."
supabase db dump --data-only > supabase/seed.sql

# 3. 【代码整理】格式化 Deno Functions 代码 (可选，为了代码美观)
if [ -d "supabase/functions" ]; then
    echo "💅 3. 正在格式化 Deno 代码..."
    deno fmt supabase/functions > /dev/null 2>&1
fi

# 4. 【Git 提交】提交到本地 Git 仓库
echo "📦 4. 正在提交到 Git..."
git add .
git commit -m "chore: $MIGRATION_NAME"

# 5. 【推送】(可选) 推送到 GitHub，如果你不想自动推，把下面这行注释掉
# 注意：根据规则，此步骤已被注释，请手动执行 git push
# echo "☁️  5. 正在推送到 GitHub..."
# git push

echo "✅ 全部完成！你的队友现在可以 Clone 并使用了。"
echo "⚠️  注意：请手动执行 'git push' 推送到远程仓库。"
