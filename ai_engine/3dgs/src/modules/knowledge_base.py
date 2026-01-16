import os
import json
from openai import OpenAI
from supabase import Client

class KnowledgeBase:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
        # 使用兼容 OpenAI 接口的 Embedding 服务 (这里以阿里云为例，或者直接用 OpenAI)
        # 建议使用 text-embedding-v3-small (OpenAI) 或 text-embedding-v2 (Aliyun)
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v2" # 确保维度是 1536

    def _get_embedding(self, text: str):
        """调用模型生成向量"""
        text = text.replace("\n", " ")
        try:
            resp = self.client.embeddings.create(input=[text], model=self.model)
            return resp.data[0].embedding
        except Exception as e:
            print(f"⚠️ Embedding 生成失败: {e}")
            return None

    def add_asset(self, task_data: dict, metadata: dict, ply_path: str):
        """
        将生成好的模型存入知识库
        task_data: 原始任务信息 (user_id, scene_id, id)
        metadata: Pipeline 返回的 AI 分析数据 (description, objects, score...)
        ply_path: 模型在云端的存储路径
        """
        scene_id = task_data['scene_id']
        description = metadata.get('ai_description', '')
        objects = metadata.get('ai_objects', [])
        tags = metadata.get('ai_tags', [])
        
        # 1. 构造带有权重的文本 (Subject-First Strategy)
        # 核心逻辑：显式标注并重复主体物体，以增强其在向量空间中的权重
        main_subject = objects[0] if objects else "未知物体"
        weighted_text = (
            f"核心物体: {main_subject}。 {main_subject}。 "  # 重复一次主体，增加向量权重
            f"详细描述: {description}。 "
            f"包含物品: {', '.join(objects)}。 "
            f"环境标签: {', '.join(tags)}。"
        )
        
        print(f"🧠 [RAG] 正在向量化: {weighted_text[:30]}...")
        
        # 2. 生成向量
        vector = self._get_embedding(weighted_text)
        if not vector:
            return False

        # 3. 构造数据库记录
        record = {
            "scene_id": scene_id,
            "user_id": task_data.get('user_id'),
            "source_task_id": task_data.get('id'),
            
            # 语义数据
            "description": description,
            "objects": objects,
            "tags": tags,
            "embedding": vector,
            
            # 资产路径 (用于未来复用)
            "ply_path": ply_path,
            
            # 技术参数 (存入 JSONB)
            "meta_info": {
                "quality_score": metadata.get('ai_score', 0),
                "quality_reason": metadata.get('ai_reason', ''),
                "engine_version": "nerfstudio-splatfacto"
            }
        }

        # 4. 插入 Supabase (使用 upsert 避免重复)
        try:
            self.supabase.table("model_assets").upsert(
                record, on_conflict="scene_id"
            ).execute()
            print(f"📚 [RAG] 资产 '{scene_id}' 已更新/入库！")
            return True
        except Exception as e:
            print(f"❌ [RAG] 入库失败: {e}")
            return False

    def search_similar_assets(self, query_text: str, limit=5):
        """(未来功能) 语义搜索相似模型"""
        query_vector = self._get_embedding(query_text)
        if not query_vector: return []
        
        # 调用 Supabase 的向量匹配函数 (需要先在 SQL 定义 RPC，见下文)
        params = {
            "query_embedding": query_vector, 
            "match_threshold": 0.7, 
            "match_count": limit
        }
        res = self.supabase.rpc("match_model_assets", params).execute()
        return res.data