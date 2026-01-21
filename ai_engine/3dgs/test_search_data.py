#!/usr/bin/env python3
"""
添加更多测试数据到向量数据库
"""
import os
import sys
import time
import uuid
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from supabase import create_client
from src.modules.knowledge_base import KnowledgeBase

load_dotenv()

def add_test_scene(scene_id, description, objects, tags, ply_path):
    """添加测试场景"""
    sup_url = os.getenv("SUPABASE_URL")
    sup_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(sup_url, sup_key)
    kb = KnowledgeBase(supabase)

    task_data = {
        "scene_id": scene_id,
        "user_id": "test_user",
        "id": str(uuid.uuid4())
    }

    metadata = {
        "ai_description": description,
        "ai_objects": objects,
        "ai_tags": tags,
        "ai_score": 85,
        "ai_reason": "测试数据"
    }

    success = kb.add_asset(task_data, metadata, ply_path)
    print(f"  {'✅' if success else '❌'} {scene_id}: {description[:40]}...")
    return success

def main():
    print("=" * 60)
    print("🧪 添加测试场景数据")
    print("=" * 60)

    scenes = [
        # 场景1: 书房
        ("test_study_room", "明亮的书房，有一张写字台、椅子和书架。墙上挂着风景画。", 
         ["写字台", "椅子", "书架", "风景画"], ["书房", "室内", "明亮"], "test_user/test_study/output/point_cloud.ply"),
        
        # 场景2: 卧室
        ("test_bedroom", "温馨的卧室，有大床、衣柜和床头柜。窗帘遮光良好。",
         ["床", "衣柜", "床头柜", "窗帘"], ["卧室", "室内", "温馨", "暗光"], "test_user/test_bedroom/output/point_cloud.ply"),
        
        # 场景3: 厨房
        ("test_kitchen", "现代化的厨房，有冰箱、灶台和洗碗机。台面整洁。",
         ["冰箱", "灶台", "洗碗机", "台面"], ["厨房", "室内", "现代"], "test_user/test_kitchen/output/point_cloud.ply"),
        
        # 场景4: 客厅
        ("test_living_room", "宽敞的客厅，有沙发、茶几和电视柜。地毯柔软舒适。",
         ["沙发", "茶几", "电视柜", "地毯"], ["客厅", "室内", "宽敞"], "test_user/test_living/output/point_cloud.ply"),
        
        # 场景5: 办公室
        ("test_office", "简约的办公室，有电脑桌、办公椅和文件柜。",
         ["电脑桌", "办公椅", "文件柜"], ["办公室", "室内", "简约"], "test_user/test_office/output/point_cloud.ply"),
    ]

    print(f"\n添加 {len(scenes)} 个测试场景...\n")
    for scene_id, desc, objects, tags, ply_path in scenes:
        add_test_scene(scene_id, desc, objects, tags, ply_path)
        time.sleep(0.5)  # 避免 API 限流

    print("\n" + "=" * 60)
    print("✅ 测试数据添加完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
