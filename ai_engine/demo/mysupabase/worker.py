import time
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# 配置
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # 必须是 Service Role Key
BUCKET_NAME = "braindance-assets"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def process_task(task):
    """处理单个任务的核心逻辑"""
    task_id = task['id']
    user_id = task['user_id']
    scene_id = task['scene_id']
    
    print(f"\n🚀 [接单] 开始处理任务: User={user_id} | Scene={scene_id}")

    # 1. 更新状态为 "processing"
    supabase.table("processing_tasks").update({"status": "processing"}).eq("id", task_id).execute()

    try:
        # --- 模拟：从 Storage 下载视频 ---
        # 路径约定：user_id/scene_id/raw/input.mp4
        input_path = f"{user_id}/{scene_id}/raw/input.mp4"
        print(f"   ⬇️ 正在下载: {input_path}")
        
        # 真正下载 (这里会报错如果文件不存在，所以要try)
        # video_data = supabase.storage.from_(BUCKET_NAME).download(input_path)
        time.sleep(2) # 假装在下载
        
        # --- 模拟：调用 3DGS AI 引擎 ---
        print(f"   🧠 正在进行 3D Gaussian Splatting 训练 (假装跑了很久)...")
        time.sleep(3) 

        # --- 模拟：上传结果 ---
        output_path = f"{user_id}/{scene_id}/output/model.ply"
        dummy_result = b"Ply model data header..."
        
        print(f"   ⬆️ 正在上传结果: {output_path}")
        supabase.storage.from_(BUCKET_NAME).upload(
            output_path, 
            dummy_result, 
            file_options={"upsert": "true"}
        )

        # 2. 更新状态为 "completed"
        supabase.table("processing_tasks").update({
            "status": "completed", 
            "updated_at": "now()"
        }).eq("id", task_id).execute()
        
        print(f"✅ [完成] 任务 {task_id} 搞定！")

    except Exception as e:
        print(f"❌ [失败] 任务出错: {e}")
        # 更新状态为 failed
        supabase.table("processing_tasks").update({"status": "failed"}).eq("id", task_id).execute()

def main_loop():
    """主循环：不断轮询数据库"""
    print("👀 Worker 启动，正在监听任务队列...")
    
    while True:
        try:
            # 1. 查询所有 status = 'pending' 的任务
            response = supabase.table("processing_tasks").select("*").eq("status", "pending").execute()
            tasks = response.data

            if tasks:
                print(f"📦 发现 {len(tasks)} 个新任务")
                for task in tasks:
                    process_task(task)
            else:
                # print(".", end="", flush=True) # 心跳包
                pass

            # 休息 5 秒再查，避免 CPU 爆炸
            time.sleep(5)
            
        except Exception as e:
            print(f"⚠️ 轮询出错: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()