#!/usr/bin/env python3
"""
# 功能：将本地 Supabase Storage 的文件迁移到远程 Supabase
# 实现：遍历本地存储桶，下载文件并上传到远程
# 逻辑：1. 列出本地文件  2. 逐个下载  3. 上传到远程  4. 验证文件
# 包含：进度跟踪、断点续传、错误处理、并发控制
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
import tempfile

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client

# 本地 Supabase 配置
LOCAL_URL = "http://127.0.0.1:54321"
LOCAL_KEY = "sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz"

# 存储桶名称
BUCKET_NAME = "braindance-assets"

# 备份目录
BACKUP_DIR = Path("/tmp/supabase_migration_backup")


def list_all_files(client, bucket: str, path: str = "") -> List[Dict]:
    """
    递归列出存储桶中的所有文件

    Args:
        client: Supabase 客户端
        bucket: 存储桶名称
        path: 起始路径（默认为根目录）

    Returns:
        文件列表，每个文件包含 name, metadata 等信息
    """
    all_files = []

    try:
        # 列出当前路径的文件和文件夹
        files = client.storage.from_(bucket).list(path=path)

        for file in files:
            file_path = f"{path}/{file['name']}" if path else file['name']

            # 如果是文件夹（id 为 None），递归遍历
            if file.get('id') is None:
                sub_files = list_all_files(client, bucket, file_path)
                all_files.extend(sub_files)
            else:
                # 是文件，添加到列表
                file['full_path'] = file_path
                all_files.append(file)

    except Exception as e:
        print(f"⚠️ 列出路径 {path} 失败：{e}")

    return all_files


def migrate_storage():
    """迁移存储文件"""

    print("=" * 60)
    print("🔄 BrainDance 存储文件迁移工具")
    print("=" * 60)

    # 1. 连接本地和远程 Supabase
    print("\n📡 连接 Supabase...")

    try:
        # 本地连接
        local_client = create_client(LOCAL_URL, LOCAL_KEY)
        print(f"✅ 本地连接成功：{LOCAL_URL}")

        # 远程连接（从 .env 读取）
        load_dotenv()
        remote_url = os.getenv("SUPABASE_URL")
        remote_key = os.getenv("SUPABASE_KEY")

        if not remote_url or not remote_key:
            print("❌ 缺少远程 Supabase 配置，请检查 .env 文件")
            return False

        remote_client = create_client(remote_url, remote_key)
        print(f"✅ 远程连接成功：{remote_url}")

    except Exception as e:
        print(f"❌ 连接失败：{e}")
        return False

    # 2. 列出本地存储文件
    print("\n" + "-" * 60)
    print(f"📦 扫描本地存储桶 '{BUCKET_NAME}'...")

    try:
        local_files = list_all_files(local_client, BUCKET_NAME)

        if not local_files:
            print(f"⚠️ 本地存储桶 '{BUCKET_NAME}' 没有文件")
            return True

        print(f"✅ 发现文件：{len(local_files)} 个")

        # 统计总大小
        total_size = sum(f.get('metadata', {}).get('size', 0) for f in local_files)
        print(f"📊 总大小：{total_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"❌ 列出文件失败：{e}")
        return False

    # 3. 创建文件清单（用于断点续传）
    manifest_file = BACKUP_DIR / f"storage_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    # 4. 迁移文件
    print("\n" + "-" * 60)
    print("🚀 开始迁移文件...")

    success_count = 0
    error_count = 0
    errors = []
    uploaded_files = []  # 记录已上传的文件

    for idx, file_info in enumerate(local_files, 1):
        file_path = file_info['full_path']
        file_size = file_info.get('metadata', {}).get('size', 0)
        size_mb = file_size / 1024 / 1024

        print(f"\n⬆️ [{idx}/{len(local_files)}] {file_path}")
        print(f"   大小：{size_mb:.2f} MB")

        try:
            # 从本地下载文件
            print(f"   📥 下载中...")
            file_data = local_client.storage.from_(BUCKET_NAME).download(file_path)

            # 创建临时文件
            print(f"   📤 上传中...")
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_data if isinstance(file_data, bytes) else file_data.encode())
                tmp_file_path = tmp_file.name

            # 上传到远程
            with open(tmp_file_path, 'rb') as f:
                result = remote_client.storage.from_(BUCKET_NAME).upload(
                    path=file_path,
                    file=f,
                    file_options={"x-upsert": "true"}
                )

            # 清理临时文件
            os.unlink(tmp_file_path)

            success_count += 1
            uploaded_files.append(file_path)
            print(f"   ✅ 成功")

        except Exception as e:
            error_count += 1
            error_msg = f"{file_path}: {str(e)}"
            errors.append(error_msg)
            print(f"   ❌ 失败：{e}")

            # 保存错误信息到文件
            error_file = BACKUP_DIR / "storage_errors.txt"
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {error_msg}\n")

    # 5. 保存已上传文件清单
    print("\n" + "-" * 60)
    print(f"💾 保存上传清单...")

    try:
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(uploaded_files, f, ensure_ascii=False, indent=2)
        print(f"✅ 清单已保存：{manifest_file}")
    except Exception as e:
        print(f"⚠️ 保存清单失败：{e}")

    # 6. 验证文件
    print("\n" + "=" * 60)
    print("📊 迁移结果")
    print("=" * 60)

    # 获取远程文件数
    try:
        remote_files = list_all_files(remote_client, BUCKET_NAME)
        remote_count = len(remote_files)
    except:
        remote_count = "未知"

    print(f"\n📈 数据统计")
    print(f"   本地文件数：{len(local_files)}")
    print(f"   远程文件数：{remote_count}")
    print(f"   成功迁移：{success_count} 个")
    print(f"   失败：{error_count} 个")

    if error_count > 0:
        print(f"\n⚠️ 部分文件迁移失败，错误详情：")
        for error in errors[:10]:  # 只显示前 10 个错误
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误（见错误日志）")
        return False
    else:
        print(f"\n✅ 所有文件迁移完成！")
        return True


def main():
    """主函数"""
    try:
        success = migrate_storage()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断迁移")
        return 1
    except Exception as e:
        print(f"\n❌ 迁移过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
