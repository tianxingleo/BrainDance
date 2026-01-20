#!/usr/bin/env python3
"""
Supabase 集成测试：模拟真实的 Worker 任务流程

此测试用于验证 Pipeline 与 Supabase 数据库的集成
需要先在数据库中创建测试任务

SQL 示例:
```sql
INSERT INTO processing_tasks (
    id,
    scene_id,
    user_id,
    task_type,
    task_params,
    status,
    created_at
) VALUES (
    'test_sam3d_001',
    'test_scene_sam3d',
    'test_user',
    'single_image_sam3d',
    '{"quality": "high"}',
    'pending',
    NOW()
);
```
"""
import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from supabase import create_client
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "braindance-assets")


def create_test_task():
    """在数据库中创建测试任务"""
    print("\n📝 创建测试任务...")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ 缺少 Supabase 配置，请检查 .env 文件")
        return None

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    task_data = {
        "id": f"test_sam3d_{int(time.time())}",
        "scene_id": f"test_scene_sam3d_{int(time.time())}",
        "user_id": "test_user",
        "task_type": "single_image_sam3d",
        "task_params": {"quality": "high"},
        "status": "pending",
    }

    try:
        result = supabase.table("processing_tasks").insert(task_data).execute()
        print(f"✅ 任务创建成功: {task_data['id']}")
        return task_data
    except Exception as e:
        print(f"❌ 任务创建失败: {e}")
        return None


def poll_for_completion(task_id, timeout=300):
    """轮询任务状态直到完成"""
    print(f"\n🔄 轮询任务状态: {task_id}")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ 缺少 Supabase 配置")
        return None

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = supabase.table("processing_tasks").select("*").eq("id", task_id).execute()
            if response.data:
                task = response.data[0]
                status = task.get("status")
                logs = task.get("logs", [])

                print(f"状态: {status}", end="")

                if logs:
                    last_log = logs[-1]
                    print(f" | 最新日志: {last_log.get('msg', '')[:50]}...")

                if status == "completed":
                    print("\n✅ 任务完成!")
                    return task
                elif status == "failed":
                    print("\n❌ 任务失败")
                    return task
                elif status == "processing":
                    print(" ⏳ 等待 Worker 处理...")
                else:
                    print(f" ⏳ 等待中 ({status})...")

            time.sleep(5)

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            time.sleep(5)

    print("⏰ 超时")
    return None


def upload_test_image(scene_id, user_id):
    """上传测试图片到存储"""
    print(f"\n📤 上传测试图片...")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ 缺少 Supabase 配置")
        return False

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    test_image = project_root.parent / "demo/SAM3d/test_input.png"
    if not test_image.exists():
        print(f"❌ 测试图片不存在: {test_image}")
        return False

    try:
        storage_path = f"{user_id}/{scene_id}/raw/image.png"
        with open(test_image, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=f,
                file_options={"x-upsert": "true"}
            )
        print(f"✅ 图片已上传: {storage_path}")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def run_integration_test():
    """运行集成测试"""
    print("=" * 60)
    print("Supabase 集成测试")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ 请先配置 .env 文件中的 SUPABASE_URL 和 SUPABASE_KEY")
        return False

    user_id = "test_user"
    task = create_test_task()
    if not task:
        return False

    scene_id = task["scene_id"]

    # 上传测试图片
    if not upload_test_image(scene_id, user_id):
        return False

    # 启动 Worker (在独立进程中或手动启动)
    print("\n" + "=" * 60)
    print("请启动 Worker 来处理此任务:")
    print(f"  cd {project_root}")
    print(f"  conda activate gs_linux_backup")
    print(f"  python -m src.core.worker")
    print("=" * 60)

    # 轮询结果
    result = poll_for_completion(task["id"])

    if result:
        print(f"\n📊 最终结果:")
        print(f"  状态: {result.get('status')}")
        print(f"  日志数: {len(result.get('logs', []))}")
        return result.get("status") == "completed"

    return False


if __name__ == "__main__":
    try:
        success = run_integration_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
