#!/usr/bin/env python3
"""
Supabase 完整测试流程：创建任务 + 启动 Worker

使用说明:
    1. 先上传图片: python scripts/run_sam3d_test.py --upload
    2. 创建任务并运行: python scripts/run_sam3d_test.py --run
    3. 一步完成: python scripts/run_sam3d_test.py --all
"""
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client


def create_task(scene_id: str, user_id: str = "user1") -> str:
    """在数据库中创建任务，返回任务 ID"""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    client = create_client(supabase_url, supabase_key)
    
    task_data = {
        "scene_id": scene_id,
        "user_id": user_id,
        "task_type": "single_image_sam3d",
        "task_params": "{}",
        "status": "pending",
    }
    
    result = client.table("processing_tasks").insert(task_data).execute()
    
    if result.data:
        task_id = result.data[0]["id"]
        print(f"✅ 任务创建成功: {task_id}")
        return task_id
    else:
        print("❌ 任务创建失败")
        return None


def poll_task(task_id: str, timeout: int = 300):
    """轮询任务状态"""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    client = create_client(supabase_url, supabase_key)
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = client.table("processing_tasks").select("*").eq("id", task_id).execute()
            
            if response.data:
                task = response.data[0]
                status = task.get("status")
                logs = task.get("logs", [])
                
                if logs:
                    last_log = logs[-1]
                    print(f"状态: {status} | 最新: {last_log.get('msg', '')[:60]}...")
                
                if status == "completed":
                    print("\n✅ 任务完成!")
                    return True
                elif status == "failed":
                    print(f"\n❌ 任务失败")
                    if logs:
                        error_log = [l for l in logs if "错误" in l.get("msg", "") or "Error" in l.get("msg", "")]
                        if error_log:
                            print(f"错误: {error_log[-1].get('msg')}")
                    return False
                elif status == "processing":
                    print(f"⏳ Worker 正在处理...")
                else:
                    print(f"⏳ 等待中 ({status})...")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            time.sleep(5)
    
    print("⏰ 超时")
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3D 完整测试流程")
    parser.add_argument("--scene_id", "-s", default="sam3d_scene", help="场景 ID")
    parser.add_argument("--user_id", "-u", default="user1", help="用户 ID")
    parser.add_argument("--upload", action="store_true", help="仅上传图片")
    parser.add_argument("--run", action="store_true", help="仅创建任务并监控")
    parser.add_argument("--all", action="store_true", help="上传 + 创建任务 + 启动 Worker")
    
    args = parser.parse_args()
    
    # 1. 上传图片
    if args.upload or args.all:
        print("=" * 50)
        print("步骤 1: 上传图片")
        print("=" * 50)
        os.system(f"python scripts/upload_image.py -s {args.scene_id} -u {args.user_id}")
    
    # 2. 创建任务
    if args.run or args.all:
        print("\n" + "=" * 50)
        print("步骤 2: 创建任务")
        print("=" * 50)
        task_id = create_task(args.scene_id, args.user_id)
        
        if task_id:
            print("\n" + "=" * 50)
            print("步骤 3: 启动 Worker")
            print("=" * 50)
            print("请在新终端中运行:")
            print(f"  cd ai_engine/3dgs")
            print(f"  conda activate gs_linux_backup")
            print(f"  python main.py")
            print("\n或者让我帮你监控任务状态...")
            poll_task(task_id)


if __name__ == "__main__":
    main()
