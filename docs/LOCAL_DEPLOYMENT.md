# 本地部署指南 (LOCAL_DEPLOYMENT)

> BrainDance Supabase Edge Function - 本地开发与测试指南

## 目录

- [前提条件](#前提条件)
- [环境配置](#环境配置)
- [启动本地 Supabase](#启动本地-supabase)
- [运行边缘函数](#运行边缘函数)
- [测试验证](#测试验证)
- [故障排除](#故障排除)

---

## 前提条件

### 1. 安装必要工具

```bash
# 安装 Supabase CLI
brew install supabase/tap/supabase  # macOS
# 或通过 npm: npm install -g @supabase/cli

# 验证安装
supabase --version
```

### 2. 安装 Deno (用于本地开发)

```bash
# macOS
brew install deno

# 或通过 curl (Linux/macOS)
curl -fsSL https://deno.land/x/install/install.sh | sh

# 验证
deno --version
```

---

## 环境配置

### 1. 配置环境变量

编辑 `supabase/functions/search-models/.env.local`：

```bash
# 复制示例文件
cp supabase/functions/search-models/.env.local.example supabase/functions/search-models/.env.local

# 编辑填入你的 DashScope API Key
# DASHSCOPE_API_KEY=sk-your-actual-api-key-here
```

**获取 DashScope API Key**: 
- 访问 https://dashscope.console.aliyun.com/
- 创建 API Key 并复制

### 2. 确保 Supabase 本地配置正确

检查 `supabase/config.toml` 中的服务端口：

```toml
[api]
port = 54321
```

---

## 启动本地 Supabase

### 1. 启动所有服务

```bash
cd supabase
supabase start
```

**预期输出**:
```
Started local Supabase development setup. 

         API URL: http://127.0.0.1:54321
     GraphQL URL: http://127.0.0.1:54321/graphql/v1
          DB URL: postgresql://postgres:postgres@127.0.0.1:54322/postgres
        Studio URL: http://127.0.0.1:54323
    InBucket URL: http://127.0.0.1:54324
      JWT SECRET: your-jwt-secret
        ANON KEY: your-anon-key
SERVICE ROLE KEY: your-service-role-key
```

**记录** `SERVICE ROLE KEY`，稍后需要在 `.env.local` 中配置（或通过 CLI 自动注入）。

### 2. 验证数据库迁移

确认 `match_model_assets` RPC 函数已存在：

```bash
# 进入 Supabase Docker 容器
docker exec -it supabase-db psql -U postgres -d postgres

# 检查函数是否存在
\df public.match_model_assets
```

---

## 运行边缘函数

### 1. 启动 Edge Function 开发服务器

```bash
cd supabase/functions/search-models

# 启动服务 (自动加载 .env.local)
supabase functions serve search-models --no-verify-jwt --env-file .env.local
```

**预期输出**:
```
Serving functions at:
- http://127.0.0.1:54321/functions/v1/search-models
```

### 2. 后台运行 (可选)

```bash
# 使用 nohup 或 tmux
nohup supabase functions serve search-models --no-verify-jwt --env-file .env.local > edge-function.log 2>&1 &
echo $! > edge-function.pid
```

---

## 测试验证

### 1. 基本功能测试

```bash
# 测试搜索接口
curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  --header 'Content-Type: application/json' \
  --data '{
    "query": "红色的杯子"
  }'
```

**预期响应**:
```json
{
  "success": true,
  "intent": {
    "original_query": "红色的杯子",
    "parsed_search_text": "红色杯子",
    "filter_start": null,
    "filter_end": null
  },
  "results": []
}
```

### 2. 带时间过滤的搜索

```bash
curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  --header 'Content-Type: application/json' \
  --data '{
    "query": "找一下上周拍的红色杯子"
  }'
```

### 3. 运行自动化测试

```bash
deno test --allow-all supabase/functions/search-models/test.ts
```

---

## 故障排除

### 问题 1: `DASHSCOPE_API_KEY` 未配置

**错误**: `未配置 DASHSCOPE_API_KEY`

**解决**:
```bash
# 确认环境变量已加载
cat supabase/functions/search-models/.env.local

# 重新启动函数服务
supabase functions serve search-models --no-verify-jwt --env-file .env.local
```

### 问题 2: RPC 函数不存在

**错误**: `数据库查询失败: relation "match_model_assets" does not exist`

**解决**:
```bash
# 检查迁移是否应用
supabase db diff

# 或手动运行迁移
supabase migration up
```

### 问题 3: CORS 错误

**错误**: 浏览器中跨域请求被阻止

**解决**: 边缘函数已配置 CORS 头部。确保请求头中包含：
```http
Origin: http://localhost:3000
```

### 问题 4: 向量维度不匹配

**错误**: 搜索结果为空或异常

**解决**: 
- 确认使用 `text-embedding-v2` 模型（与入库时一致）
- 确认数据库向量维度为 1536

### 问题 5: Supabase 服务未运行

**错误**: `Connection refused`

**解决**:
```bash
supabase status
supabase start
```

---

## 下一步

完成本地测试后，可以：

1. **部署到云端**: `supabase functions deploy search-models`
2. **切换权限**: 参考 `docs/代办/TODO_AUTH_MIGRATION.md` 迁移到用户认证
3. **集成前端**: 在 Flutter 应用中调用此接口

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 启动 Supabase | `supabase start` |
| 停止 Supabase | `supabase stop` |
| 启动函数 | `supabase functions serve search-models --no-verify-jwt --env-file .env.local` |
| 部署函数 | `supabase functions deploy search-models` |
| 查看日志 | `supabase functions logs search-models` |
| 运行测试 | `deno test --allow-all supabase/functions/search-models/test.ts` |
