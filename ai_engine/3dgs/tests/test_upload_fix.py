# test_upload_fix.py
# 功能：专门测试大文件上传及重试机制
# 实现：模拟 pipeline_base.py 中的上传逻辑，使用现有的测试文件

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
import sys
project_root = "/home/jiangbeihu/ltx/projects/BrainDance/ai_engine/3dgs"
sys.path.append(os.path.join(project_root, "src"))

from core.pipeline_base import BasePipeline

class MockPipeline(BasePipeline):
    def run(self):
        return {}

    def log(self, msg, level="INFO"):
        print(f"[{level}] {msg}")

def test_large_upload():
    # 使用报错的文件
    ply_path = "/home/jiangbeihu/ltx/projects/BrainDance/ai_engine/3dgs/src/modules/results/scene_party_001.ply"
    if not os.path.exists(ply_path):
        print(f"❌ 测试文件不存在: {ply_path}")
        return

    # 初始化一个简单的 Pipeline 实例
    pipeline = MockPipeline({"user_id": "debug_user"})
    pipeline.context = {"user_id": "debug_user"}
    
    params = {
        "scene_id": "test_upload_debug_001",
        "user_id": "debug_user"
    }
    
    metadata = {
        "ai_description": "Test upload fix for large files",
        "ai_tags": ["test", "debug"]
    }

    print(f"🚀 开始测试上传大文件 (153MB): {ply_path}")
    start_time = time.time()
    
    # 手动从 .env 加载
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, ".env"))
    except ImportError:
        print("ℹ️ python-dotenv 未安装，尝试手动解析 .env...")
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        os.environ[k] = v
    
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("⚠️ 环境变量 SUPABASE_URL 或 SUPABASE_KEY 未设置，将跳过真实上传")
    
    result = pipeline.upload_and_record(ply_path, metadata, params)
    
    duration = time.time() - start_time
    if result:
        print(f"✅ 上传成功! 耗时: {duration:.2f}s")
        print(f"🔗 远程路径: {result}")
    else:
        print(f"❌ 上传失败! 请检查日志输出。耗时: {duration:.2f}s")

if __name__ == "__main__":
    test_large_upload()
