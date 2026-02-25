import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# 加载环境变量
env_path = os.path.join(os.path.dirname(__file__), "ai_engine", "3dgs", ".env")
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("错误: 找不到 SUPABASE_URL 或 SUPABASE_KEY 环境变量")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

bucket_name = "braindance-assets"
user_id = "test1"
scene_id = "scene_party_001"
file_path = f"{user_id}/{scene_id}/raw/video.mp4"
local_video_path = "ai_engine/3dgs/test_workspace/test.mp4"

def upload_and_create_task():
    print(f"1. 准备上传本地视频 {local_video_path} 到 {bucket_name}/{file_path}...")
    
    if not os.path.exists(local_video_path):
        print(f"错误: 本地视频文件不存在: {local_video_path}")
        return

    # 读取文件并上传
    with open(local_video_path, "rb") as f:
        try:
            # 尝试覆盖上传
            res = supabase.storage.from_(bucket_name).upload(
                file=f,
                path=file_path,
                file_options={"cacheControl": "3600", "upsert": "true"}
            )
            print(f"✅ 视频上传成功: {res}")
        except Exception as e:
            print(f"上传失败 (可能是因为文件已存在但物理文件丢失导致状态异常): {e}")
            print("尝试先删除再上传...")
            try:
                supabase.storage.from_(bucket_name).remove([file_path])
                # 重新读取文件指针
                f.seek(0)
                res = supabase.storage.from_(bucket_name).upload(
                    file=f,
                    path=file_path,
                    file_options={"cacheControl": "3600", "upsert": "true"}
                )
                print(f"✅ 视频重新上传成功: {res}")
            except Exception as e2:
                print(f"❌ 重新上传也失败了: {e2}")
                return

    print("\n2. 准备创建 Task...")
    task_data = {
        "user_id": user_id,
        "scene_id": scene_id,
        "status": "pending",
        "task_type": "video_3dgs",
        "task_params": {}
    }
    
    try:
        response = supabase.table("processing_tasks").insert(task_data).execute()
        print(f"✅ 成功创建任务！")
        print(f"任务详情: {response.data[0]}")
    except Exception as e:
        print(f"❌ 创建任务失败: {e}")

if __name__ == "__main__":
    upload_and_create_task()
