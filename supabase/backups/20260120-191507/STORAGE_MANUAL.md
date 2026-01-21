# 存储文件迁移说明

## 重要提示

⚠️ **模型文件（.ply、.splat 等）体积较大，未包含在 Git 备份中。**

数据库备份已包含存储元数据（文件名、路径、权限），但实际文件需要手动迁移。

## 已备份的内容（数据库中）
- `storage.buckets` - 存储桶配置
- `storage.objects` - 文件元数据（文件名、路径、大小、MIME 类型）
- `storage.migrations` - 存储迁移历史

## 需要手动迁移的内容
- `braindance-assets` bucket 中的实际文件
- 包含：原始视频（raw/）、3D 模型（output/）、预览配置

## 手动迁移步骤

### 在原电脑上导出
```bash
# 1. 停止存储服务
docker stop supabase_storage_BrainDance

# 2. 打包存储卷
docker run --rm \
  -v supabase_storage_BrainDance:/source:ro \
  -v $(pwd)/storage_export:/backup \
  alpine:latest \
  sh -c "cd /source && tar czvf /backup/storage_files.tar.gz ."

# 3. 复制到新电脑（U盘、云盘等）
cp storage_files.tar.gz /path/to/transfer/
```

### 在新电脑上导入
```bash
# 1. 启动 Supabase
supabase start

# 2. 停止存储服务
docker stop supabase_storage_BrainDance

# 3. 解压文件
mkdir -p /tmp/storage_restore
tar xzvf storage_files.tar.gz -C /tmp/storage_restore/

# 4. 恢复存储卷
docker run --rm \
  -v supabase_storage_BrainDance:/dest \
  -v /tmp/storage_restore:/source \
  alpine:latest \
  sh -c "cp -r /source/* /dest/"

# 5. 重启存储服务
docker start supabase_storage_BrainDance
```

## 存储文件大小估算
存储文件可能较大，建议使用以下方式传输：
- 有线传输：U盘、移动硬盘
- 云存储：Google Drive、Dropbox、阿里云盘
- 网络传输：SCP、SFTP

## 验证迁移成功
1. 访问 http://localhost:54323/studio/storage
2. 检查 braindance-assets bucket 中的文件列表
3. 下载并验证文件可以正常打开

## 注意事项
1. 确保在导出前停止存储服务，避免文件被修改
2. 传输过程中保持文件完整性，不要中断
3. 导入后验证所有文件是否完整
4. 如果遇到权限问题，检查文件所有权
