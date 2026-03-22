#!/usr/bin/env python3
"""
SHARP 集成测试前置准备脚本

功能：
1. 上传测试图片到 Supabase Storage
2. 在 processing_tasks 表中创建测试任务

使用：
    python scripts/prepare_sharp_test.py
"""
import sys
import os
import time
import uuid
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

load_dotenv()

from supabase import create_client


def upload_test_image(image_path: str, scene_id: str, user_id: str = "test_user") -> bool:
    """上传测试图片到存储桶"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    bucket_name = os.getenv("SUPABASE_BUCKET", "braindance-assets")

    if not supabase_url or not supabase_key:
        print("❌ 缺少 Supabase 配置")
        return False

    try:
        client = create_client(supabase_url, supabase_key)

        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ 图片不存在: {image_path}")
            return False

        storage_path = f"{user_id}/{scene_id}/raw/image.png"

        print(f"📤 上传图片: {image_path.name} → {bucket_name}/{storage_path}")

        with open(image_path, "rb") as f:
            client.storage.from_(bucket_name).upload(
                path=storage_path,
                file=f,
                file_options={"x-upsert": "true"}
            )

        print(f"✅ 图片上传成功")
        return True

    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def create_sharp_test_task(scene_id: str, user_id: str = "test_user") -> dict:
    """在数据库中创建 SHARP 测试任务"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ 缺少 Supabase 配置")
        return None

    try:
        client = create_client(supabase_url, supabase_key)

        task_id = str(uuid.uuid4())

        task_data = {
            "id": task_id,
            "scene_id": scene_id,
            "user_id": user_id,
            "task_type": "single_image_sharp",
            "task_params": {"quality": "high"},
            "status": "pending",
        }

        print(f"📝 创建任务: {task_id}")

        client.table("processing_tasks").insert(task_data).execute()

        print(f"✅ 任务创建成功")
        return task_data

    except Exception as e:
        print(f"❌ 任务创建失败: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("SHARP 集成测试前置准备")
    print("=" * 60)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ 请先配置 .env 文件中的 SUPABASE_URL 和 SUPABASE_KEY")
        return 1

    user_id = "test_user"
    scene_id = f"test_scene_sharp_{int(time.time())}"

    source_image = Path(
        os.getenv(
            "SHARP_TEST_IMAGE_PATH",
            str(Path(__file__).resolve().parents[2] / "demo" / "sharp" / "input.jpg"),
        )
    ).expanduser()

    if not source_image.exists():
        print(f"❌ 测试图片不存在: {source_image}")
        return 1

    if not upload_test_image(str(source_image), scene_id, user_id):
        return 1

    task_data = create_sharp_test_task(scene_id, user_id)
    if not task_data:
        return 1

    print("\n" + "=" * 60)
    print("✅ 前置准备完成")
    print("=" * 60)
    print(f"\n📋 测试信息:")
    print(f"   任务 ID: {task_data['id']}")
    print(f"   场景 ID: {scene_id}")
    print(f"   用户 ID: {user_id}")
    print(f"   任务类型: single_image_sharp")
    print(f"\n🚀 下一步: 启动 Worker 处理任务")
    print(f"   conda activate gs_linux_backup")
    print(f"   python -m src.core.worker")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
