# 存储文件迁移说明

## 重要提示

⚠️ **模型文件（.ply、.splat 等）体积较大，未包含在 Git 备份中。**

数据库备份已包含存储元数据（文件名、路径、权限），但实际文件需要手动迁移。

## 已备份的内容（数据库中）

以下元数据已包含在 `database/full_backup.dump.gz` 中：
- `storage.buckets` - 存储桶配置（braindance-assets）
- `storage.objects` - 文件元数据（文件名、路径、大小、MIME 类型）
- `storage.migrations` - 存储迁移历史
- `storage.prefixes` - 文件夹结构

## 需要手动迁移的内容

`braindance-assets` bucket 中的实际文件：
- **原始视频**: `raw/` 目录下的视频文件
- **3D 模型**: `output/` 目录下的 .ply、.splat 文件
- **预览配置**: `transforms.json` 等预览配置文件

这些文件通常体积较大（单个模型可能数百 MB），不适合提交到 Git。

## 手动迁移步骤

### 在原电脑上导出

```bash
# 1. 进入项目目录
cd /path/to/BrainDance

# 2. 停止存储服务（确保数据一致性）
docker stop supabase_storage_BrainDance

# 3. 打包存储卷
docker run --rm \
  -v supabase_storage_BrainDance:/source:ro \
  -v $(pwd)/storage_export:/backup \
  alpine:latest \
  sh -c "cd /source && tar czvf /backup/storage_files.tar.gz ."

# 4. 复制到新电脑（U盘、云盘、网络传输等）
# U盘/移动硬盘
cp storage_files.tar.gz /media/usb/
# 或网络传输
scp storage_files.tar.gz user@new-computer:/path/to/
```

### 在新电脑上导入

```bash
# 1. 进入项目目录
cd /path/to/BrainDance

# 2. 启动 Supabase（如果尚未启动）
supabase start

# 3. 停止存储服务
docker stop supabase_storage_BrainDance

# 4. 解压文件
mkdir -p /tmp/storage_restore
tar xzvf storage_files.tar.gz -C /tmp/storage_restore/

# 5. 恢复存储卷
docker run --rm \
  -v supabase_storage_BrainDance:/dest \
  -v /tmp/storage_restore:/source \
  alpine:latest \
  sh -c "cp -r /source/* /dest/"

# 6. 重启存储服务
docker start supabase_storage_BrainDance

# 7. 验证恢复
docker exec supabase_storage_BrainDance ls -la /var/lib/storage/
```

### 使用 Docker Volume 备份（替代方案）

```bash
# 创建临时容器并备份
docker run --rm -v supabase_storage_BrainDance:/data -v $(pwd):/backup alpine tar czf /backup/storage.tar.gz -C /data .

# 恢复
docker run --rm -v supabase_storage_BrainDance:/data -v $(pwd):/backup alpine tar xzf /backup/storage.tar.gz -C /data .
```

## 存储文件大小估算

典型存储文件大小：
- **单个 .ply 模型**: 50MB - 500MB
- **原始视频**: 100MB - 2GB
- **完整项目**: 可能达到 10GB+

建议传输方式：
- **有线传输**: U盘、移动硬盘（适合 >10GB）
- **云存储**: Google Drive、Dropbox、阿里云盘（适合 1-10GB）
- **网络传输**: SCP、SFTP、Rsync（适合 <5GB）

## 验证迁移成功

### 1. 检查文件数量
```bash
# 在新电脑上
docker exec supabase_storage_BrainDance find /var/lib/storage -type f | wc -l
```

### 2. 检查 Studio
1. 访问 http://localhost:54323/studio/storage
2. 点击 braindance-assets bucket
3. 验证文件列表与原电脑一致

### 3. 验证文件可下载
```bash
# 测试下载一个模型文件
curl -o /tmp/test_model.ply \
  "http://127.0.0.1:54321/storage/v1/object/public/braindance-assets/user_id/scene_id/output/point_cloud.ply"
```

### 4. 验证数据库元数据匹配
```sql
-- 检查存储对象数量
SELECT COUNT(*) FROM storage.objects;

-- 检查最近创建的文件
SELECT name, created_at FROM storage.objects ORDER BY created_at DESC LIMIT 10;
```

## 常见问题

### 问题：文件已恢复但无法下载
**解决**：检查文件权限
```bash
docker exec supabase_storage_BrainDance chown -R storage:storage /var/lib/storage/
docker exec supabase_storage_BrainDance chmod -R 755 /var/lib/storage/
```

### 问题：存储服务无法启动
**解决**：检查 Docker 卷
```bash
# 查看卷信息
docker volume ls | grep supabase

# 检查卷挂载
docker inspect supabase_storage_BrainDance
```

### 问题：文件数量不匹配
**解决**：可能是软链接问题，检查实际文件
```bash
# 统计实际文件（不含符号链接）
docker exec supabase_storage_BrainDance find /var/lib/storage -type f | wc -l
```

## 迁移后步骤

1. **验证应用程序**: 测试 AI Worker 是否能正常下载模型
2. **验证搜索功能**: 测试语义搜索是否正常工作
3. **清理临时文件**: 删除 `storage_export/` 和 `storage_files.tar.gz`

## 备份摘要

本次备份包含：
- ✅ 数据库完整备份（包括存储元数据）
- ✅ 配置文件
- ⚠️ 存储文件（需要手动迁移）

如需完整迁移，请按照本指南操作。
