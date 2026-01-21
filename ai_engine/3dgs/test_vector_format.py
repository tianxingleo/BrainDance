#!/usr/bin/env python3
"""
测试向量格式修复
验证 embedding 是否正确存储为 PostgreSQL 数组格式 (1536 维)
而不是 Python 列表的 JSON 字符串
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from supabase import create_client

# 加载环境变量
load_dotenv()

from src.modules.knowledge_base import KnowledgeBase

def test_vector_format():
    """测试向量格式是否正确"""
    print("=" * 60)
    print("🧪 测试向量格式修复")
    print("=" * 60)

    # 1. 初始化 Supabase 客户端
    sup_url = os.getenv("SUPABASE_URL")
    sup_key = os.getenv("SUPABASE_KEY")

    if not sup_url or not sup_key:
        print("❌ 缺少 Supabase 配置，请检查 .env 文件")
        return False

    print(f"📡 连接 Supabase: {sup_url[:30]}...")

    try:
        supabase = create_client(sup_url, sup_key)
    except Exception as e:
        print(f"❌ Supabase 连接失败: {e}")
        return False

    # 2. 初始化 KnowledgeBase
    kb = KnowledgeBase(supabase)

    # 3. 创建测试数据
    import time
    import uuid
    scene_id = f"test_vector_format_{int(time.time())}"

    task_data = {
        "scene_id": scene_id,
        "user_id": "test_user",
        "id": str(uuid.uuid4())  # 使用有效的 UUID 格式
    }

    metadata = {
        "ai_description": "这是一个测试场景，包含一张木桌子和一把椅子。房间里光线充足，墙上挂着几幅画。",
        "ai_objects": ["木桌子", "椅子", "画"],
        "ai_tags": ["室内", "木制家具", "明亮"],
        "ai_score": 85,
        "ai_reason": "场景清晰，光照充足，物体明确"
    }

    ply_path = "test_user/test_scene/output/point_cloud.ply"

    print(f"\n📝 测试场景: {scene_id}")
    print(f"   描述: {metadata['ai_description'][:50]}...")
    print(f"   物体: {metadata['ai_objects']}")

    # 4. 调用 add_asset (这会生成向量并存储)
    print("\n🧠 正在生成向量并存储到数据库...")
    success = kb.add_asset(task_data, metadata, ply_path)

    if not success:
        print("❌ 添加资产失败")
        return False

    # 5. 验证结果
    print("\n🔍 验证向量格式...")
    time.sleep(1)  # 等待数据库写入完成

    # 查询数据库
    try:
        response = supabase.table("model_assets").select(
            "scene_id, embedding, description"
        ).eq("scene_id", scene_id).execute()

        if not response.data:
            print("❌ 未找到刚插入的记录")
            return False

        record = response.data[0]
        embedding = record.get("embedding")

        print(f"\n📊 向量分析结果:")
        print(f"   Scene ID: {record['scene_id']}")
        print(f"   Embedding 类型: {type(embedding).__name__}")

        if embedding is None:
            print("❌ 向量为空")
            return False

        if isinstance(embedding, list):
            # 已经是 Python 列表 (JSON 数组格式)
            dims = len(embedding)
            print(f"✅ 格式: Python 列表 (JSON 数组)")
            print(f"   维度: {dims}")
            if dims != 1536:
                print(f"⚠️  警告: 期望 1536 维，实际 {dims} 维")
            else:
                print(f"✅ 维度正确: 1536")
            return dims == 1536

        elif isinstance(embedding, str):
            # 是字符串，检查是 JSON 数组格式 (pgvector 正确格式) 还是其他格式
            if embedding.startswith("["):
                # JSON 数组格式 (pgvector 正确格式!)
                inner = embedding.strip("[]")
                dims = len(inner.split(",")) if inner else 0
                print(f"✅ 格式: JSON 数组字符串 (pgvector 正确格式!)")
                print(f"   字符数: {len(embedding)}")
                print(f"   维度: {dims}")
                if dims != 1536:
                    print(f"⚠️  警告: 期望 1536 维，实际 {dims} 维")
                else:
                    print(f"✅ 维度正确: 1536")
                print(f"\n✅ 这是正确的格式！pgvector 通过 REST API 接收 JSON 数组字符串")
                return dims == 1536
            elif embedding.startswith("{"):
                # PostgreSQL 数组格式 (错误格式 - 之前代码的错误)
                inner = embedding.strip("{}")
                dims = len(inner.split(",")) if inner else 0
                print(f"❌ 格式错误: PostgreSQL 数组")
                print(f"   字符数: {len(embedding)}")
                print(f"   维度: {dims}")
                print(f"\n💡 pgvector 不接受 PostgreSQL 数组格式，只接受 JSON 数组")
                return False

        else:
            print(f"❌ 未知格式: {type(embedding)}")
            return False

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False

def main():
    success = test_vector_format()
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试通过! 向量格式正确")
    else:
        print("❌ 测试失败! 向量格式有问题")
    print("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
