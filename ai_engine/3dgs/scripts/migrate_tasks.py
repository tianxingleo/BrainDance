#!/usr/bin/env python3
"""
# 功能：将本地 Supabase 的 processing_tasks 表迁移到远程 Supabase
# 实现：连接本地和远程数据库，批量迁移任务记录
# 逻辑：1. 连接本地和远程 Supabase  2. 读取本地任务  3. 批量插入远程  4. 验证数据
# 包含：数据备份、批量迁移、进度跟踪、错误处理
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 添加父目录到路径，以便导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client

# 本地 Supabase 配置（被注释的配置）
LOCAL_URL = "http://127.0.0.1:54321"
LOCAL_KEY = "sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz"

# 批量插入大小
BATCH_SIZE = 100

# 备份目录
BACKUP_DIR = Path("/tmp/supabase_migration_backup")


def create_backup(local_client, table_name="processing_tasks"):
    """创建本地数据备份到 JSON 文件"""
    print(f"📦 创建本地数据备份...")

    try:
        # 读取所有数据
        result = local_client.table(table_name).select("*").execute()

        if not result.data:
            print(f"⚠️ 本地 {table_name} 表没有数据")
            return None

        # 保存到 JSON 文件
        backup_file = BACKUP_DIR / f"{table_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(result.data, f, ensure_ascii=False, indent=2)

        print(f"✅ 备份完成：{backup_file}")
        print(f"   备份记录数：{len(result.data)}")
        return result.data

    except Exception as e:
        print(f"❌ 备份失败：{e}")
        return None


def migrate_tasks():
    """迁移 processing_tasks 表"""

    print("=" * 60)
    print("🔄 BrainDance 任务队列迁移工具")
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

    # 2. 创建备份
    print("\n" + "-" * 60)
    backup_data = create_backup(local_client, "processing_tasks")
    if not backup_data:
        print("⚠️ 没有数据需要迁移")
        return True

    # 3. 读取本地数据
    print("\n" + "-" * 60)
    print(f"📊 本地 processing_tasks 表")
    print(f"   总记录数：{len(backup_data)}")

    # 4. 迁移数据（批量插入）
    print("\n" + "-" * 60)
    print("🚀 开始迁移到远程...")

    total = len(backup_data)
    success_count = 0
    error_count = 0
    errors = []

    for i in range(0, total, BATCH_SIZE):
        batch = backup_data[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n📦 批次 {batch_num}/{total_batches} ({len(batch)} 条记录)...")

        try:
            # 使用 upsert 避免 UUID 冲突（基于 id 主键）
            result = remote_client.table("processing_tasks").upsert(
                batch,
                on_conflict="id"  # 如果 id 冲突则更新
            ).execute()

            batch_success = len(result.data) if result.data else len(batch)
            success_count += batch_success

            print(f"   ✅ 成功插入/更新：{batch_success} 条")

        except Exception as e:
            error_count += len(batch)
            error_msg = f"批次 {batch_num} 失败：{str(e)}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")

            # 保存失败的批次到单独文件
            error_file = BACKUP_DIR / f"batch_{batch_num}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
            print(f"   💾 失败批次已保存：{error_file}")

    # 5. 验证数据
    print("\n" + "=" * 60)
    print("📊 迁移结果")
    print("=" * 60)

    # 获取本地和远程记录数
    local_count = len(backup_data)
    remote_result = remote_client.table("processing_tasks").select("count", count="exact").execute()
    remote_count = remote_result.count

    print(f"\n📈 数据统计")
    print(f"   本地记录数：{local_count}")
    print(f"   远程记录数：{remote_count}")
    print(f"   成功迁移：{success_count} 条")
    print(f"   失败：{error_count} 条")

    if local_count == remote_count:
        print(f"\n✅ 数据迁移完成！记录数一致")
        return True
    else:
        print(f"\n⚠️ 记录数不一致，请检查数据")
        if errors:
            print(f"\n❌ 错误列表：")
            for error in errors:
                print(f"   - {error}")
        return False


def main():
    """主函数"""
    try:
        success = migrate_tasks()
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
