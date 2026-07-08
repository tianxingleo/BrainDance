# Supabase 备份恢复指南

## 备份信息
- **备份时间**: 2026-01-20 19:30:18
- **原始项目**: BrainDance
- **Supabase 版本**: 本地 Docker (17.6)
- **备份内容**:
  - 数据库：`database/full_backup.dump.gz`（自定义压缩格式）
  - 配置文件：`config/`

## 前置条件
1. 安装 Docker 和 Docker Compose
2. 安装 Supabase CLI（可选）
3. 确保端口 54321-54327 未被占用
4. Git 克隆 BrainDance 项目

## 恢复步骤

### 步骤 1：准备环境
```bash
# 进入项目目录
cd BrainDance

# 如果是新电脑，克隆项目
git clone <your-repo-url>
cd supabase
```

### 步骤 2：恢复配置文件
```bash
# 备份新的配置文件（如果存在）
cp config.toml config.toml.new

# 恢复备份的配置
cp config/seed.sql .
cp -r config/migrations/ .

# 可选：恢复 config.toml（会覆盖自定义设置）
cp config/config.toml .
```

### 步骤 3：启动 Supabase（空白状态）
```bash
# 使用 Supabase CLI
supabase start

# 或使用 Docker
docker compose up -d
```

### 步骤 4：恢复数据库
```bash
# 解压数据库备份
gunzip database/full_backup.dump.gz

# 恢复数据库（使用 pg_restore）
docker exec -i supabase_db_BrainDance pg_restore -U postgres -d postgres -c database/full_backup.dump

# 验证恢复
docker exec -i supabase_db_BrainDance psql -U postgres -d postgres -c "\dt"
```

### 步骤 5：验证恢复
```bash
# 检查数据库连接
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "\dt"

# 检查存储桶元数据
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "SELECT * FROM storage.objects LIMIT 5;"

# 检查 pgvector 扩展
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# 访问 Studio
# 打开浏览器访问 http://localhost:54323
```

## 注意事项

### 1. 密码更改
恢复后建议更改以下密码：
- PostgreSQL postgres 用户密码
- JWT secret
- Storage API 密钥

### 2. API 密钥
如果使用了新的 Supabase 项目，需要更新 `.env` 文件中的 API 密钥：
```bash
# 编辑环境变量
nano ../ai_engine/3dgs/.env

# 更新以下配置
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=sb_new_secret_key_here
```

### 3. 端口冲突
如果端口已被占用，修改 `config.toml` 中的端口设置。

### 4. 扩展问题
如果 pgvector 扩展未自动启用，手动执行：
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### 5. Edge Functions
如果使用了 Edge Functions，需要重新配置环境变量：
```bash
cd functions/search-models
# 编辑 .env.local
nano .env.local
# 添加 DASHSCOPE_API_KEY
```

## 故障排除

### 问题：pg_restore 报错版本不匹配
**解决**：确保使用相同版本的 PostgreSQL 客户端工具。

### 问题：认证用户无法登录
**解决**：检查 auth schema 是否正确恢复：
```sql
SELECT * FROM auth.users LIMIT 1;
```

### 问题：存储桶为空
**解决**：检查 storage.objects 表是否有记录：
```sql
SELECT * FROM storage.objects LIMIT 5;
```
注意：实际文件需要按照 STORAGE_MANUAL.md 手动迁移。

### 问题：向量搜索不工作
**解决**：验证 pgvector 扩展已启用：
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
-- 如果未启用，手动启用
CREATE EXTENSION IF NOT EXISTS vector;
```

## 验证检查清单
- [ ] 数据库表全部存在 (`\dt`)
- [ ] RPC 函数可用 (`SELECT * FROM pg_proc WHERE proname = 'match_model_assets';`)
- [ ] RLS 策略已应用 (`\d model_assets`)
- [ ] 存储桶元数据可查询 (`SELECT * FROM storage.objects`)
- [ ] 认证用户可登录
- [ ] pgvector 扩展已启用
- [ ] API 端点可访问 (http://127.0.0.1:54321)

## 相关文件
- `database/full_backup.dump.gz` - 数据库备份
- `config/` - 配置文件目录
- `STORAGE_MANUAL.md` - 存储文件迁移说明

## 下一步
1. 按照 STORAGE_MANUAL.md 迁移存储文件（如果需要）
2. 更新环境变量配置
3. 测试应用程序连接
4. 验证所有功能正常
