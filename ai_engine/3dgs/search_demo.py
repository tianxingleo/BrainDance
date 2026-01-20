import os
import time
import json
import datetime
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

class ModelSearcher:
    def __init__(self):
        # --- 配置初始化 ---
        self.sup_url = os.getenv("SUPABASE_URL")
        self.sup_key = os.getenv("SUPABASE_KEY")
        self.api_key = os.getenv("DASHSCOPE_API_KEY") # 阿里云 Qwen Key
        
        if not self.sup_url or not self.sup_key:
            raise ValueError("❌ 缺少 Supabase 配置，请检查 .env")

        # 初始化 Supabase
        self.supabase: Client = create_client(self.sup_url, self.sup_key)
        
        # 初始化 Embedding 客户端 (兼容 OpenAI 接口)
        self.ai_client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # ⚠️ 必须和存入时用的模型一致！
        self.embed_model = "text-embedding-v2" 

    def _parse_query_intent(self, user_query: str):
        """让 LLM 提取搜索词和时间范围"""
        today = datetime.date.today().isoformat()
        
        system_prompt = f"""
        你是搜索助手。当前日期是: {today}。
        用户会输入一句搜索请求，你需要提取：
        1. search_text: 真正用于搜索物体的描述（去掉时间词）。
        2. start_time: ISO8601 格式的开始时间 (UTC)，如果没有则为 null。
        3. end_time: ISO8601 格式的结束时间 (UTC)，如果没有则为 null。

        例子1: "找一下上周拍的红色杯子"
        输出: {{"search_text": "红色杯子", "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-07T23:59:59Z"}}
        
        例子2: "搜索之前的猫" (无具体时间)
        输出: {{"search_text": "猫", "start_time": null, "end_time": null}}

        只返回 JSON。
        """
        
        resp = self.ai_client.chat.completions.create(
            model="qwen-plus", # 建议用稍微聪明点的模型处理逻辑
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"} # 强制 JSON
        )
        return json.loads(resp.choices[0].message.content)

    def get_embedding(self, text: str):
        """将自然语言转换为 1536 维向量"""
        try:
            resp = self.ai_client.embeddings.create(
                input=[text],
                model=self.embed_model
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding 生成失败: {e}")
            return None

    def search(self, query_text: str, top_k=5, threshold=0.01):
        """执行语义搜索"""
        print(f"\n🧠 分析意图: '{query_text}' ...")
        start_time = time.time()

        # 1. 解析意图 (LLM)
        intent = self._parse_query_intent(query_text)
        real_query = intent.get('search_text', query_text)
        t_start = intent.get('start_time')
        t_end = intent.get('end_time')
        
        print(f"    -> 🔍 语义搜索: '{real_query}'")
        if t_start: print(f"    -> 📅 时间范围: {t_start} 至 {t_end}")

        # 2. 生成向量 (只用 real_query，去掉了时间词干扰)
        query_vector = self.get_embedding(real_query)
        if not query_vector:
            return

        # 3. 调用 RPC (带时间参数)
        try:
            rpc_params = {
                "query_embedding": query_vector,
                "match_threshold": threshold, # 相似度门槛 (0-1)
                "match_count": top_k,
                "filter_start": t_start, # 传给 SQL
                "filter_end": t_end      # 传给 SQL
            }
            response = self.supabase.rpc("match_model_assets", rpc_params).execute()
            results = response.data
            
        except Exception as e:
            print(f"❌ 数据库查询失败: {e}")
            return

        duration = time.time() - start_time
        print(f"✅ 耗时 {duration:.2f}s，找到 {len(results)} 个结果:\n")

        # Step 3: 格式化输出结果
        if not results:
            print("🤷‍♂️ 未找到相关模型。试着描述得更具体一点？")
            return

        for idx, item in enumerate(results):
            # 获取公开下载链接 (假设 Bucket 是公开的，或者是私有的需要签名)
            # 这里演示获取临时签名链接 (有效期 60秒)
            # 假设 ply_path 存在 item['ply_path'] 里，或者我们通过 scene_id 拼出来
            
            # 注意：RPC返回的字段取决于你的 SQL function select 了哪些字段
            # 假设你之前 SQL 里 select 了 id, scene_id, description, similarity
            
            scene_id = item.get('scene_id', 'Unknown')
            desc = item.get('description', '无描述')
            score = item.get('similarity', 0)
            
            # 尝试构造个下载链接 (仅作演示)
            try:
                # 假设路径规则是 user_id/scene_id/output/point_cloud.ply
                # 如果你的 RPC 没返回 user_id，这里可能拼不准，需要优化 RPC 返回更多字段
                # 这里暂时演示逻辑
                print(f"[{idx+1}] 🏆 相似度: {score:.2%}")
                print(f"    🎬 场景: {scene_id}")
                print(f"    📝 描述: {desc}")
                print("-" * 40)
            except:
                pass

if __name__ == "__main__":
    searcher = ModelSearcher()
    
    # 交互式循环
    while True:
        user_input = input("\n请输入搜索内容 (输入 q 退出): ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if user_input.strip():
            searcher.search(user_input, top_k=3)