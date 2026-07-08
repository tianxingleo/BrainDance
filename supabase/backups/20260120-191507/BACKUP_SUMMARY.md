# 备份摘要

## 备份信息
- 备份时间：2026-01-20 19:15:07
- 项目名称：BrainDance
- Supabase 版本：本地 Docker (PostgreSQL 17.6)

## 文件清单
- 数据库：full_backup.dump.gz（压缩格式）
- 配置文件：config/
- 恢复指南：README.md
- 存储迁移说明：STORAGE_MANUAL.md
- 备份摘要：BACKUP_SUMMARY.md

## 文件详情
### 数据库备份
- 文件：database/full_backup.dump.gz
- 大小：约 102KB（压缩后）

### 配置备份
- config.toml - Supabase CLI 配置（14KB）
- seed.sql - 存储桶初始化脚本（1KB）
- migrations/ - 数据库迁移脚本

### 文档
- README.md - 恢复指南（2.8KB）
- STORAGE_MANUAL.md - 存储迁移说明（2.1KB）
- BACKUP_SUMMARY.md - 本摘要文件

## 注意事项
- 存储模型文件（.ply、.splat 等）未包含在 Git 备份中
- 详见 STORAGE_MANUAL.md 进行手动迁移
- .env 文件包含敏感信息，未包含在备份中

## 验证命令
```bash
# 检查备份文件
ls -la supabase/backups/20260120-191507/

# 验证数据库备份
gzip -d < supabase/backups/20260120-191507/database/full_backup.dump.gz | head -20

# 查看恢复指南
cat supabase/backups/20260120-191507/README.md

# 查看存储迁移说明
cat supabase/backups/20260120-191507/STORAGE_MANUAL.md
```
