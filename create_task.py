import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# 加载环境变量
load_dotenv("ai_engine/3dgs/.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ 未找到 Supabase 配置！请检查 .env 文件。")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_3dgs_task(user_id: str, scene_id: str, video_path: str = None):
    """
    创建一个视频生成 3DGS 的任务
    """
    print(f"🚀 准备创建 3DGS 任务...")
    print(f"👤 用户 ID: {user_id}")
    print(f"🎬 场景 ID: {scene_id}")
    
    # 1. 如果提供了本地视频路径，先上传到 Supabase Storage
    if video_path and os.path.exists(video_path):
        bucket_name = os.getenv("SUPABASE_BUCKET", "braindance-assets")
        storage_path = f"{user_id}/{scene_id}/raw/video.mp4"
        
        print(f"📤 正在上传视频到存储桶 {bucket_name}: {storage_path} ...")
        with open(video_path, "rb") as f:
            # 注意：如果文件已存在，可能需要先删除或使用 upsert
            supabase.storage.from_(bucket_name).upload(
                file=f,
                path=storage_path,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )
        print("✅ 视频上传成功！")
    elif video_path:
        print(f"⚠️ 警告: 找不到本地视频文件 {video_path}，跳过上传步骤。请确保云端已有该文件。")
    else:
        print("ℹ️ 未提供本地视频路径，假设云端存储中已存在对应的视频文件。")

    # 2. 在 processing_tasks 表中插入一条新记录
    task_data = {
        "user_id": user_id,
        "scene_id": scene_id,
        "status": "pending",
        "task_type": "video_dual_chain",
        "task_params": {
            "slow_pipeline": "video_3dgs",
            "sam3d_vram_threshold_gb": 25,
            "best_frame_sample_count": 8,
        }
    }
    
    print("📝 正在向 processing_tasks 表插入任务数据...")
    response = supabase.table(os.getenv("SUPABASE_TABLE", "processing_tasks")).insert(task_data).execute()
    
    if response.data:
        task_id = response.data[0]['id']
        print(f"🎉 任务创建成功！")
        print(f"🆔 任务 ID: {task_id}")
        print(f"⏳ 状态: pending (等待 Worker 接单)")
    else:
        print("❌ 任务创建失败！")

if __name__ == "__main__":
    # 示例用法：
    # 生成一个随机的场景 ID
    sample_scene_id = "test-scene-001"
    sample_user_id = "test-user-001"
    
    # 如果你有本地视频，可以填入路径，例如：
    # sample_video_path = "./test_video.mp4"
    sample_video_path = None 
    
    create_3dgs_task(
        user_id=sample_user_id, 
        scene_id=sample_scene_id,
        video_path=sample_video_path
    )
