# 备份摘要

## 备份信息
- **备份时间**: 2026-01-20 19:30:18
- **项目名称**: BrainDance
- **Supabase 版本**: 本地 Docker (PostgreSQL 17.6)
- **备份位置**: `supabase/backups/20260120-193018/`

## 文件清单

### 数据库备份
- 文件: `database/full_backup.dump.gz`
- 大小: 102K
- 格式: PostgreSQL 自定义压缩格式（pg_dump --format=custom --compress=9）
- 包含:
  - 所有表结构和数据（model_assets, processing_tasks, tasks, rag_docs）
  - 扩展（vector, pg_trgm, pg_net 等）
  - RPC 函数（match_model_assets）
  - RLS 安全策略
  - 存储元数据（storage.objects, storage.buckets）
  - 认证数据（auth.users, auth.sessions）
  - Realtime 配置

### 配置文件
- `config/config.toml` (14KB) - Supabase CLI 配置
- `config/seed.sql` (1KB) - 存储桶初始化脚本
- `config/migrations/` - 所有数据库迁移脚本
- `config/FILE_LIST.md` - 文件清单文档

### 文档
- `README.md` - 恢复指南
- `STORAGE_MANUAL.md` - 存储文件迁移说明
- `BACKUP_SUMMARY.md` - 本文件

## 备份命令
```bash
# 数据库备份
docker exec supabase_db_BrainDance pg_dump -U postgres -d postgres \
  --format=custom --compress=9 --verbose -f /tmp/database_backup.dump

# 复制到主机
docker cp supabase_db_BrainDance:/tmp/database_backup.dump \
  ./supabase/backups/20260120-193018/database/full_backup.dump

# 压缩
gzip ./supabase/backups/20260120-193018/database/full_backup.dump
```

## 注意事项

### 已包含 ✅
- 数据库完整备份（包含所有 schema 和扩展）
- 配置文件和迁移脚本
- 存储元数据（文件名、路径、权限）

### 未包含 ❌
- 存储文件（.ply、.splat、视频等大文件）
  - 详见 STORAGE_MANUAL.md 进行手动迁移
- Docker 镜像（可重新拉取）
- 临时文件和日志
- 敏感凭证明文

## 下一步操作

### 1. 提交到 Git
```bash
cd /home/ltx/projects/BrainDance
git add supabase/backups/20260120-193018/
git commit -m "chore(supabase): 添加 $(date +%Y%m%d) 完整备份"
git push  # 可选
```

### 2. 迁移存储文件（如需要）
按照 STORAGE_MANUAL.md 的说明，手动迁移 braindance-assets bucket 中的实际文件。

### 3. 在新电脑上恢复
按照 README.md 的步骤，在新电脑上恢复 Supabase 环境。

## 验证命令
```bash
# 检查备份文件
ls -lh supabase/backups/20260120-193018/database/
ls -lh supabase/backups/20260120-193018/config/
ls supabase/backups/20260120-193018/*.md

# 验证数据库备份格式
file supabase/backups/20260120-193018/database/full_backup.dump.gz

# 验证配置文件
cat supabase/backups/20260120-193018/config/FILE_LIST.md
```

## 联系信息
如有问题，请检查：
1. README.md - 恢复指南
2. STORAGE_MANUAL.md - 存储迁移说明
3. 项目文档: `/home/ltx/projects/BrainDance/supabase/README.md`
