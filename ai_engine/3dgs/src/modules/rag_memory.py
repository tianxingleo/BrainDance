# src/modules/rag_memory.py
# 功能：实现RAG记忆模块，将场景信息存入向量数据库用于后续检索
# 实现：使用阿里云DashScope的文本嵌入服务生成向量，存储到Supabase向量表
# 逻辑：1. 将文本转换为向量 2. 将向量和元数据存入数据库 3. 支持场景检索
# 包含：RagMemory类、文本嵌入方法、知识库存储方法
import os
from openai import OpenAI
from supabase import Client

class RagMemory:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
        # 使用兼容 OpenAI 协议的 Embedding 服务
        # 如果用阿里云 DashScope，Base URL 是 https://dashscope.aliyuncs.com/compatible-mode/v1
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"), # 复用之前的 Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v2" # 阿里云的 embedding 模型

    def embed_text(self, text: str):
        """将文本转换为向量"""
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
            # 将“描述 + 物品列表 + Tags”组合在一起变成向量，增加搜索命中率
            combined_text = f"{description} 包含物体: {', '.join(objects)} 标签: {task_data.get('tags', [])}"
            vector = self.embed_text(combined_text)

            # 2. 存入数据库
            data = {
                "scene_id": task_data['scene_id'],
                "user_id": task_data.get('user_id', 'default'),
                "description": description,
                "objects": objects,
                "embedding": vector
            }
            
            self.supabase.table("model_knowledge_base").insert(data).execute()
            print(f"🧠 [记忆模块] 已将场景 '{task_data['scene_id']}' 存入向量库")
            return True
            
        except Exception as e:
            print(f"⚠️ [记忆模块] 保存失败: {e}")
            return False