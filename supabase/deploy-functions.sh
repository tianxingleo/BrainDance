#!/bin/bash
# Supabase Edge Functions 本地部署脚本
# 功能：在本地环境中测试和部署 Edge Functions

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Supabase Edge Functions 部署脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查是否在项目根目录
if [ ! -f "supabase/config.toml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 1. 启动 Supabase 本地服务（如果尚未启动）
echo -e "\n${GREEN}📦 步骤 1: 检查 Supabase 本地服务状态${NC}"
if ! supabase status &>/dev/null; then
    echo -e "${BLUE}启动 Supabase 本地服务...${NC}"
    supabase start
else
    echo -e "${GREEN}✓ Supabase 本地服务已在运行${NC}"
fi

# 2. 显示当前函数列表
echo -e "\n${GREEN}📋 步骤 2: 当前 Edge Functions 列表${NC}"
supabase functions list

# 3. 部署单个函数到本地环境
FUNCTION_NAME="search-models"
echo -e "\n${GREEN}🚀 步骤 3: 部署函数到本地环境${NC}"
echo -e "${BLUE}部署函数: ${FUNCTION_NAME}${NC}"

# 部署到本地（不需要 --project-ref，本地模式自动检测）
supabase functions deploy "${FUNCTION_NAME}" --no-verify-jwt

# 4. 检查函数是否成功部署
echo -e "\n${GREEN}✅ 步骤 4: 验证部署状态${NC}"
supabase functions list | grep "${FUNCTION_NAME}"

# 5. 显示函数的本地访问 URL
echo -e "\n${GREEN}🌐 本地访问信息${NC}"
echo -e "${BLUE}函数 URL: http://127.0.0.1:54321/functions/v1/${FUNCTION_NAME}${NC}"
echo -e "${BLUE}Studio URL: http://127.0.0.1:54323${NC}"

# 6. 提供测试命令
echo -e "\n${GREEN}🧪 测试命令${NC}"
echo -e "${BLUE}curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/${FUNCTION_NAME}' \${NC}"
echo -e "${BLUE}  --header 'Content-Type: application/json' \${NC}"
echo -e "${BLUE}  --data '{\"query\":\"搜索测试\"}'${NC}"

echo -e "\n${GREEN}✨ 部署完成！${NC}"
echo -e "${BLUE}提示: 如需查看实时日志，运行: supabase functions logs ${FUNCTION_NAME} --tail${NC}"
