# Supabase 备份恢复指南

## 备份信息
- **备份时间**: 2026-01-20 16:51
- **原始项目**: BrainDance
- **备份内容**:
  - 数据库: `full_backup.sql.gz`（包含存储元数据）
  - 配置文件: `config/`

## 前置条件
1. 安装 Docker 和 Docker Compose
2. 安装 Supabase CLI（可选）
3. 确保端口 54321-54327 未被占用

## 恢复步骤

### 步骤 1：准备环境
```bash
# 进入项目目录
cd BrainDance

# 如果是新电脑，克隆项目
git clone <your-repo-url>
cd supabase

# 拉取备份（如果从远程仓库获取）
git pull origin <branch>
```

### 步骤 2：恢复配置文件
```bash
# 备份新的配置文件（如果存在）
cp config.toml config.toml.new
cp -r migrations migrations.new

# 恢复备份的配置
cp config/seed.sql .
cp -r config/migrations/ migrations/
```

### 步骤 3：启动 Supabase（空白状态）
```bash
supabase start
# 或使用 Docker
docker compose up -d
```

### 步骤 4：恢复数据库
```bash
# 解压数据库备份
gunzip database/full_backup.sql.gz

# 恢复数据库
docker exec -i supabase_db_BrainDance psql -U postgres -d postgres < database/full_backup.sql

# 或使用 pg_restore（如果使用自定义格式）
docker exec -i supabase_db_BrainDance pg_restore -U postgres -d postgres -c database/full_backup.dump
```

### 步骤 5：验证恢复
```bash
# 检查数据库连接
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "\dt"

# 检查存储桶元数据
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "SELECT * FROM storage.objects LIMIT 5;"

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
如果使用了新的 Supabase 项目，需要更新 `.env` 文件中的 API 密钥

### 3. 端口冲突
如果端口已被占用，修改 `config.toml` 中的端口设置

### 4. 扩展问题
如果 pgvector 扩展未自动启用，手动执行：
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## 故障排除

### 问题：pg_dump 备份的文件无法恢复
**解决**: 确保使用相同的 PostgreSQL 版本

### 问题：认证用户无法登录
**解决**: 检查 auth schema 是否正确恢复

### 问题：存储桶为空
**解决**: 检查 `storage.objects` 表是否有记录，文件元数据已恢复

## 验证检查清单
- [ ] 数据库表全部存在
- [ ] RPC 函数可用
- [ ] RLS 策略已应用
- [ ] 存储桶元数据可查询
- [ ] 认证用户可登录
- [ ] API 端点可访问

## 存储文件迁移
⚠️ **重要**: 存储模型文件（.ply、.splat 等）未包含在此备份中！

请参阅 `STORAGE_MANUAL.md` 了解如何手动迁移存储文件。
