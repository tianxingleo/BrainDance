import time
import os
from pathlib import Path

# 引入 Supabase 客户端库，用于连接云数据库和存储
from supabase import create_client, Client
# 引入 python-dotenv 库，用于加载本地 .env 文件中的环境变量（保护密钥安全）
from dotenv import load_dotenv

# 引入项目内部配置类和核心管线函数
from src.config import PipelineConfig
from src.core.pipeline import run_pipeline

# [初始化] 加载当前目录下的 .env 文件
# 这一步必须在所有 os.getenv 调用之前执行，否则读不到变量
load_dotenv()

class CloudWorker:
    """
    ☁️ CloudWorker (云端工人)
    
    职责：
    1. 持续监听 Supabase 数据库中的任务表。
    2. 抢单：发现状态为 'pending' 的任务并锁定。
    3. 执行：下载视频 -> 调用 3DGS 核心引擎 -> 生成模型。
    4. 汇报：实时同步日志到数据库，并将最终结果上传回云存储。
    """

    def __init__(self):
        """
        初始化 Worker：连接 Supabase，准备本地缓存目录。
        """
        # --- 1. 读取环境变量配置 ---
        # 使用 os.getenv 读取 .env 文件中的配置，第二个参数是默认值
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "braindance-assets")  # 存储桶名称
        self.TABLE_NAME = os.getenv("SUPABASE_TABLE", "processing_tasks")    # 任务表名称
        
        # --- 2. 防御性检查 ---
        # 如果关键配置缺失，直接报错停止，避免后续出现莫名其妙的连接错误
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("❌ 初始化失败：未找到 Supabase 配置！请检查 .env 文件是否存在且填写正确。")

        # --- 3. 建立连接 ---
        # 创建 Supabase 客户端实例，后续所有数据库/存储操作都通过它进行
        self.supabase: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        
        # --- 4. 准备本地工作区 ---
        # 🟢 [修改后] 找回原来的 "braindance_workspace"，实现路径归一化
        # 这样所有的任务数据和模型都会存放在 /home/ltx/braindance_workspace
        self.CACHE_DIR = Path.home() / "braindance_workspace"
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 🟢 [新增] 定义一个公共模型目录，用于存放 SAM 权重等，实现多任务共享
        self.MODELS_DIR = self.CACHE_DIR / "models"
        self.MODELS_DIR.mkdir(exist_ok=True)
        
        # --- 5. 初始化日志缓冲区 ---
        # [关键设计] 用于解决“读写冲突”问题。
        # 我们不再每次 Select 数据库，而是将当前任务的所有日志保存在这个内存列表里。
        # 每次有新日志，append 进这里，然后把整个列表覆盖上传到云端。
        self.current_task_logs = []

    def _sync_log(self, task_id):
        """
        🔄 [内部方法] 日志同步器
        
        功能：将内存中的日志缓冲区 (self.current_task_logs) 全量推送到 Supabase。
        设计哲学：采用“内存为王，覆盖更新”策略，避免多线程下的数据覆盖问题。
        """
        try:
            # 直接调用 Update 接口，将 logs 字段更新为当前的内存列表
            self.supabase.table(self.TABLE_NAME).update({
                "logs": self.current_task_logs
            }).eq("id", task_id).execute()
        except Exception as e:
            # ⚠️ 注意：日志同步失败属于“非致命错误”。
            # 如果网络抖动导致日志没发出去，不应该中断核心训练任务，所以这里只打印不抛出异常。
            print(f"⚠️ [网络抖动] 日志同步跳过: {e}")

    def start(self):
        """
        🚀 [主入口] 启动监听循环
        这是外部调用的唯一入口，启动后会进入死循环，直到被手动停止。
        """
        print(f"🚀 [CloudWorker] 启动成功! 正在监听任务表: [{self.TABLE_NAME}]")
        try:
            while True:
                # 执行一次“心跳”检测
                self._tick()
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 中断信号，优雅退出
            print("\n🛑 [CloudWorker] 接收到停止信号，服务已关闭。")

    def _tick(self):
        """
        💓 [心跳函数] 单次轮询逻辑
        """
        try:
            # --- 1. 轮询数据库 ---
            # 查询条件：状态(status)必须是 'pending' (待处理)
            # limit(1)：每次只取 1 个任务，避免贪多嚼不烂
            response = self.supabase.table(self.TABLE_NAME)\
                .select("*").eq("status", "pending").limit(1).execute()
            
            # --- 2. 判断是否有任务 ---
            if response.data:
                # 🎯 发现任务！立即处理
                # response.data 是一个列表，我们取第一个元素
                self._process_task(response.data[0])
            else:
                # 💤 没有任务，休眠 3 秒
                # 避免死循环空转导致 CPU 占用率过高，同时也减少数据库压力
                time.sleep(3)
                # 打印一个小点，证明程序还活着 (心跳显示)
                print(".", end="", flush=True)
                
        except Exception as e:
            # 🛡️ 容错处理
            # 比如数据库断连、查询超时等，打印错误并强制休息 5 秒，防止错误刷屏
            print(f"\n⚠️ 轮询错误: {e}")
            time.sleep(5)

    def _process_task(self, task):
        """
        ⚙️ [核心逻辑] 处理单个任务
        """
        # --- 1. 解包任务信息 ---
        task_id = task['id']            # 任务唯一ID
        scene_id = task['scene_id']     # 场景/项目ID (作为文件名)
        # 获取用户ID，如果数据库里没存 user_id 字段，就用默认值 'default_user'
        user_id = task.get('user_id', 'default_user') 
        
        print(f"\n📥 [接收任务] 场景ID: {scene_id} | 任务ID: {task_id}")

        # --- 2. 重置日志缓冲区 ---
        # [重要] 开始新任务前，必须清空上一条任务的残留日志，防止串台
        self.current_task_logs = []

        # --- 3. 定义回调函数 (闭包) ---
        # 这个函数会传给 pipeline.py，让核心引擎在深层代码里也能发日志
        def on_pipeline_log(message):
            # A. 构造标准日志对象 (时间戳 + 消息)
            log_entry = {
                "ts": int(time.time()), # 当前秒级时间戳
                "msg": message
            }
            # B. 写入本地内存 (操作极快，绝对不丢数据)
            self.current_task_logs.append(log_entry)
            
            # C. 触发云端同步
            self._sync_log(task_id)

        try:
            # =================== 阶段 A: 锁定任务 ===================
            # 将状态改为 'processing'，告诉其他 Worker 这个任务我接了，别抢
            # 同时清空 logs 字段，准备开始新纪录
            self.supabase.table(self.TABLE_NAME).update({
                "status": "processing",
                "logs": []
            }).eq("id", task_id).execute()

            # =================== 阶段 B: 下载资源 ===================
            on_pipeline_log("正在从云端下载视频...")
            
            # 构造本地存储路径: ./temp_cache/xxx.mp4
            video_path = self.CACHE_DIR / f"{scene_id}.mp4"
            # 构造云端下载路径: user_id/scene_id/raw/video.mp4
            storage_path = f"{user_id}/{scene_id}/raw/video.mp4"
            
            # 下载文件流并写入本地
            try:
                with open(video_path, 'wb') as f:
                    # 从指定 Bucket 下载
                    res = self.supabase.storage.from_(self.BUCKET_NAME).download(storage_path)
                    f.write(res)
            except Exception as e:
                # 针对下载失败做特殊说明，方便排查是路径不对还是网络问题
                raise RuntimeError(f"视频下载失败 (路径: {storage_path}): {e}")

            # =================== 阶段 C: 执行引擎 ===================
            # 准备输出目录
            task_output_dir = self.CACHE_DIR / scene_id  # 直接用场景名做目录
            
            # 实例化配置对象 (Config)
            cfg = PipelineConfig(
                project_name=scene_id,
                video_path=video_path,
                work_root=task_output_dir, # 设定工作目录
                enable_ai=True,            # 开启 AI 增强
                shared_model_dir=self.MODELS_DIR  # 🟢 传入共享模型目录
            )
            
            # 🔥 调用核心管线! 
            # 传入回调函数 on_pipeline_log，实现实时日志
            # 🟢 [修改点 1] 运行 Pipeline 并接收元数据
            # 这里的 run_pipeline 现在返回两个值: (ply_path, metadata_dict)
            try:
                result = run_pipeline(cfg, log_callback=on_pipeline_log)
                
                # 兼容性处理：防止 pipeline 还没改成返回 tuple 导致报错
                if isinstance(result, tuple):
                    final_ply_path, metadata = result
                else:
                    final_ply_path, metadata = result, {}
            except Exception as e:
                # 即使 Pipeline 报错（比如被 AI 拦截了），我们也尝试捕获它跑出的 metadata
                # 这里暂时简单处理，依赖 result 在报错前是否已经产生（实际报错时 result 不会返回）
                # 生产环境下可以把 metadata 放在异常对象里抛出
                raise e

            # 🟢 [修改点 2] 立即同步 AI 分析结果到数据库
            # 不管训练是否成功，只要有分析结果，都应该存下来
            if metadata:
                update_data = {}
                
                # 1. 同步分数
                if "ai_score" in metadata:
                    update_data["quality_score"] = metadata["ai_score"]
                
                # 2. 同步标签
                if "ai_tags" in metadata:
                    update_data["tags"] = metadata["ai_tags"]
                
                # 3. 同步评价原因 (新!)
                if "ai_reason" in metadata:
                    update_data["quality_reason"] = metadata["ai_reason"]
                
                # 执行更新
                if update_data:
                    self.supabase.table(self.TABLE_NAME)\
                        .update(update_data)\
                        .eq("id", task_id)\
                        .execute()
                    on_pipeline_log(f"✅ AI 评分已同步: {metadata.get('ai_score')}分")

            # 校验结果：如果 pipeline 返回 None 或者文件不存在，说明训练挂了
            if not final_ply_path or not Path(final_ply_path).exists():
                raise RuntimeError("Pipeline 执行结束，但未生成有效的 PLY 文件，请检查训练日志。")

            # =================== 阶段 D: 上传结果 ===================
            on_pipeline_log("训练完成，正在上传结果到云端...")
            
            # 1. 上传 PLY 模型文件
            upload_ply_key = f"{user_id}/{scene_id}/output/point_cloud.ply"
            with open(final_ply_path, 'rb') as f:
                self.supabase.storage.from_(self.BUCKET_NAME).upload(
                    path=upload_ply_key, 
                    file=f, 
                    # x-upsert=true 表示如果文件已存在则覆盖
                    file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
                )

            # 2. 上传 transforms.json (用于网页预览)
            # 假设该文件在 PLY 同级目录或配置指定的目录
            if cfg.transforms_file.exists():
                upload_json_key = f"{user_id}/{scene_id}/output/transforms.json"
                with open(cfg.transforms_file, 'rb') as f:
                    self.supabase.storage.from_(self.BUCKET_NAME).upload(
                        path=upload_json_key,
                        file=f,
                        file_options={"content-type": "application/json", "x-upsert": "true"}
                    )
                on_pipeline_log("上传 transforms.json 成功")

            # =================== 阶段 E: 完结撒花 ===================
            # 更新状态为 'completed'
            self.supabase.table(self.TABLE_NAME).update({
                "status": "completed"
            }).eq("id", task_id).execute()
            
            on_pipeline_log("✅ 任务全部完成")
            print("✅ 任务完成")

        except Exception as e:
            # =================== 异常处理 ===================
            print(f"❌ 任务处理失败: {e}")
            
            # 1. 尝试将错误信息记录到云端日志，让用户知道死在哪一步了
            try:
                self.current_task_logs.append({"ts": int(time.time()), "msg": f"❌ 严重错误: {str(e)}"})
                self._sync_log(task_id)
            except:
                pass # 如果这时候连网都断了，就放弃写日志
            
            # 2. 将任务状态标记为 'failed'，避免死循环重试
            self.supabase.table(self.TABLE_NAME).update({"status": "failed"}).eq("id", task_id).execute()
        
        finally:
            # =================== 🧹 清理工作 (新增逻辑) ===================
            import shutil # 确保引入 shutil

            # 1. 删除源视频文件
            if 'video_path' in locals() and video_path.exists():
                try:
                    os.remove(video_path)
                    print(f"🗑️ 已删除临时视频: {video_path.name}")
                except Exception as e:
                    print(f"⚠️ 删除视频失败: {e}")
            
            # 2. 删除任务输出目录 (包含图片、COLMAP数据、PLY等所有中间产物)
            # ⚠️ 警告：如果你还没有修改 ai_segmentor.py 让模型下载到公共目录，
            # 这里的删除操作会把下载在里面的 AI 模型也删掉！请务必先做“模型搬家”。
            if 'task_output_dir' in locals() and task_output_dir.exists():
                try:
                    shutil.rmtree(task_output_dir)
                    print(f"🗑️ 已清空任务工作区: {task_output_dir.name}")
                except Exception as e:
                    print(f"⚠️ 清理工作区失败: {e}")

            # 3. 重置日志
            self.current_task_logs = []