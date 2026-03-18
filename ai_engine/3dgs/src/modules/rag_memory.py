# src/modules/rag_memory.py
# 功能：实现RAG记忆模块，将场景信息存入向量数据库用于后续检索
# 实现：使用阿里云DashScope的文本嵌入服务生成向量，存储到Supabase向量表
# 逻辑：1. 将文本转换为向量 2. 将向量和元数据存入数据库 3. 支持场景检索
# 包含：RagMemory类、文本嵌入方法、知识库存储方法
import os
from typing import Optional
try:
    # lazy import OpenAI - may not be installed in all dev environments
    from openai import OpenAI
except Exception:  # pragma: no cover - defensive
    OpenAI = None
try:
    from supabase import Client
except Exception:  # pragma: no cover - defensive
    Client = object

class RagMemory:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        # 使用兼容 OpenAI 协议的 Embedding 服务（按需初始化）
        # 如果 openai SDK 不可用，self.client 会是 None，embed_text 需要被替换/mock
        if OpenAI is not None:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                api_key = os.getenv("DASHSCOPE_API_KEY")
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            except Exception:
                self.client = None
        else:
            self.client = None
        self.model = "text-embedding-v2"  # 阿里云的 embedding 模型

    def embed_text(self, text: str):
        """将文本转换为向量"""
        if not self.client:
            raise RuntimeError("Embedding client not initialized")
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

    def save_to_knowledge_base(self, task_data: dict, description: str, objects: list):
        """保存到 Supabase 向量表"""
        try:
            # 1. 生成向量
            # 将"描述 + 物品列表 + Tags"组合在一起变成向量，增加搜索命中率
            combined_text = f"{description} 包含物体: {', '.join(objects)} 标签: {task_data.get('tags', [])}"
            vector = self.embed_text(combined_text)

            # ⚠️ 关键修复: pgvector 期望 JSON 数组格式 "[...]"，不是 PostgreSQL 数组 "{...}"
            # 直接使用 Python list，supabase Python SDK 会将其序列化为 JSON 数组
            embedding_json = vector  # Python list 会自动序列化为 JSON 数组 [0.1, 0.2, ...]

            # 2. 构造与 model_assets 表一致的行并存入数据库
            tags = task_data.get('tags') or []
            meta_info = task_data.get('meta_info') or {}

            row = {
                "scene_id": task_data['scene_id'],
                "user_id": task_data.get('user_id', 'default'),
                "description": description,
                "objects": objects,
                "tags": tags,
                "embedding": embedding_json,  # 使用 Python list (自动序列化为 JSON 数组)
                # 尝试从 task_data 里读取 ply_path/preview 路径，若无则 None
                "ply_path": task_data.get('ply_path'),
                "meta_info": meta_info,
            }

            # 防御：必须有 embedding 才写入
            if not row.get('embedding'):
                print(f"⚠️ [记忆模块] 缺少 embedding，跳过写入: scene_id={task_data.get('scene_id')}")
                return False

            # 使用 model_assets（仓库 migration 中存在的表）进行 upsert
            self.supabase.table("model_assets").upsert(row, on_conflict="scene_id").execute()
            print(f"🧠 [记忆模块] 已将场景 '{task_data['scene_id']}' 写入 model_assets (upsert)")
            return True
            
        except Exception as e:
            print(f"⚠️ [记忆模块] 保存失败: {e}")
            return False
