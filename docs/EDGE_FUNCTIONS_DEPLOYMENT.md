# Supabase Edge Functions 部署指南

## 📖 目录
- [本地部署](#本地部署)
- [远程部署](#远程部署)
- [常见问题](#常见问题)

---

## 🖥️ 本地部署

### 方法 1: 使用自动化脚本（推荐）

```bash
# 运行部署脚本
cd /home/ltx/projects/BrainDance
./supabase/deploy-functions.sh
```

### 方法 2: 手动部署

#### 1️⃣ 启动 Supabase 本地服务

```bash
# 启动所有本地服务（数据库、API、Storage、Functions）
supabase start

# 查看服务状态
supabase status
```

#### 2️⃣ 部署 Edge Function 到本地

```bash
# 部署单个函数
supabase functions deploy search-models --no-verify-jwt

# 部署所有函数
supabase functions deploy --no-verify-jwt
```

#### 3️⃣ 测试本地函数

```bash
# 测试 search-models 函数
curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  --header 'Content-Type: application/json' \
  --data '{"query":"搜索红色杯子"}'
```

#### 4️⃣ 查看函数日志

```bash
# 实时查看函数日志
supabase functions logs search-models --tail

# 查看所有函数日志
supabase functions logs --tail
```

---

## 🌐 远程部署

### 前置准备

1. **登录 Supabase 账户**
```bash
supabase login
```

2. **链接到远程项目**
```bash
# 方法 1: 通过项目 ID 链接
supabase link --project-ref YOUR_PROJECT_ID

# 方法 2: 通过项目选择（会列出所有项目）
supabase link
```

### 部署到远程 Supabase

#### 1️⃣ 设置环境变量（远程）

```bash
# 在 Supabase Dashboard 中设置环境变量
# 访问: https://app.supabase.com/project/YOUR_PROJECT_ID/functions/secrets

# 或通过 CLI 设置（需要先链接项目）
supabase secrets set DASHSCOPE_API_KEY=your_api_key_here
```

#### 2️⃣ 部署函数到远程

```bash
# 部署单个函数到远程
supabase functions deploy search-models --project-ref YOUR_PROJECT_ID

# 如果已链接项目，可以省略 --project-ref
supabase functions deploy search-models
```

#### 3️⃣ 验证远程部署

```bash
# 列出远程项目的所有函数
supabase functions list --project-ref YOUR_PROJECT_ID

# 查看函数详情
supabase functions get search-models --project-ref YOUR_PROJECT_ID
```

#### 4️⃣ 测试远程函数

```bash
# 获取你的项目 URL
# 格式: https://YOUR_PROJECT_ID.supabase.co

curl -i --location --request POST 'https://YOUR_PROJECT_ID.supabase.co/functions/v1/search-models' \
  --header "Authorization: Bearer YOUR_ANON_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"query":"搜索测试"}'
```

---

## 🔧 常见问题

### Q1: 如何更新已部署的函数？

**本地环境：**
```bash
# 修改代码后重新部署
supabase functions deploy search-models --no-verify-jwt
```

**远程环境：**
```bash
# 重新部署会自动覆盖旧版本
supabase functions deploy search-models
```

### Q2: 如何删除函数？

```bash
# 删除本地函数（不会删除代码文件）
supabase functions delete search-models --local

# 删除远程函数
supabase functions delete search-models --project-ref YOUR_PROJECT_ID
```

### Q3: 函数启动失败怎么办？

```bash
# 查看详细日志
supabase functions logs search-models --tail

# 检查环境变量
supabase secrets list --project-ref YOUR_PROJECT_ID
```

### Q4: 如何热重载函数？

本地开发时，Supabase 支持热重载：
- 修改 `index.ts` 文件后，保存即可自动重新加载
- 无需重新部署

### Q5: 如何调试函数？

```bash
# 1. 查看实时日志
supabase functions logs search-models --tail

# 2. 在代码中添加 console.log
console.log("调试信息:", data);

# 3. 使用 Deno inspector（配置在 config.toml）
# inspector_port = 8083
```

---

## 📝 代码更新工作流

### 推荐的更新流程：

1. **修改代码**
   ```bash
   vim supabase/functions/search-models/index.ts
   ```

2. **本地测试**
   ```bash
   # 如果服务未启动
   supabase start

   # 重新部署到本地
   supabase functions deploy search-models --no-verify-jwt

   # 测试函数
   curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
     -H 'Content-Type: application/json' \
     -d '{"query":"测试"}'

   # 查看日志
   supabase functions logs search-models --tail
   ```

3. **部署到远程**
   ```bash
   # 确认本地测试通过后
   supabase functions deploy search-models
   ```

---

## 🔐 安全注意事项

1. **不要提交敏感信息**
   - 使用 `.env.local` 存储本地环境变量
   - 在 Supabase Dashboard 中管理远程环境变量
   - 确保 `.gitignore` 包含 `.env.local`

2. **JWT 验证**
   - 开发环境可以使用 `--no-verify-jwt` 方便测试
   - 生产环境应启用 JWT 验证保护 API

3. **API 密钥管理**
   - 本地使用 Service Role Key
   - 远程使用适当权限的密钥

---

## 📚 相关命令速查

```bash
# 服务管理
supabase start              # 启动本地服务
supabase stop               # 停止本地服务
supabase status             # 查看服务状态

# 函数管理
supabase functions list                     # 列出所有函数
supabase functions deploy <name>            # 部署函数
supabase functions delete <name>            # 删除函数
supabase functions logs <name> --tail       # 查看实时日志

# 密钥管理
supabase secrets list                       # 列出所有环境变量
supabase secrets set KEY=value              # 设置环境变量

# 项目链接
supabase link                               # 链接远程项目
supabase unlink                             # 取消链接
```

---

## 🆘 获取帮助

- **官方文档**: https://supabase.com/docs/guides/functions
- **CLI 文档**: https://supabase.com/docs/reference/cli
- **项目 Issues**: https://github.com/supabase/supabase/issues
