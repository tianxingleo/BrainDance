import os
import json
# 1. 导入国产模型组件
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# === 配置部分 ===
# 阿里云 DashScope API Key (去 aliyun.com 开通)
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxx" 

DATA_DIR = "./data"
DB_DIR = "./chroma_db_qwen"

# === 核心函数 1: 加载并绑定数据 ===
def load_and_bind_data(directory):
    documents = []
    
    # 1. 先扫描所有的“场景文件” (scene_summary.json) 建立索引
    # 作用：建立 scene_id -> 场景描述 的映射，方便后面给图片绑定信息
    scene_map = {} 
    
    for root, _, files in os.walk(directory):
        if "scene_summary.json" in files:
            with open(os.path.join(root, "scene_summary.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 假设文件夹名字就是场景ID，例如 ./data/scene_01/
                scene_id = os.path.basename(root) 
                scene_map[scene_id] = data.get("summary", "")
                
                # == 存入父级信息 (3D模型) ==
                # 这样用户搜“白色触控笔模型”时能找到它
                doc = Document(
                    page_content=data["summary"], # 搜索内容
                    metadata={
                        "type": "model_summary",    # 类型：模型总览
                        "scene_id": scene_id,       # 绑定ID <--- 核心
                        "file_path": os.path.join(root, "model.ply"), # 假定模型路径
                        "shooting_mode": data["shooting_strategy"]["mode"]
                    }
                )
                documents.append(doc)

    # 2. 再扫描所有的“图片帧” (frame_xxxx.json)
    for root, _, files in os.walk(directory):
        scene_id = os.path.basename(root) # 获取当前所在的场景ID
        
        for file in files:
            if file.startswith("frame_") and file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # == 存入子级信息 (图片) ==
                # 这里实现了“绑定”：metadata 里既有自己的 filename，也有父亲的 scene_id
                meta = {
                    "type": "image_frame",         # 类型：图片帧
                    "scene_id": scene_id,          # 绑定ID <--- 核心：指向属于哪个模型
                    "filename": data["filename"],
                    "file_path": os.path.join(root, data["filename"]), # 图片真实路径
                    "quality": data.get("quality_score", 0),
                    # 把父级的背景知识也放进去，方便后续检索展示
                    "parent_context": scene_map.get(scene_id, "未知场景") 
                }
                
                doc = Document(
                    page_content=data["description"], # 搜索内容
                    metadata=meta
                )
                documents.append(doc)

    return documents

# === 核心函数 2: 国产化向量库构建 ===
def build_index(documents):
    # 使用阿里 DashScope 的 embedding 模型 (text-embedding-v1 或 v2)
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vector_db

# === 核心函数 3: 检索与定位 ===
def search_engine(query):
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 检索 Top 3
    results = db.similarity_search(query, k=3)
    
    print(f"\n用户提问: {query}")
    print("-" * 30)
    
    context_list = []
    
    for doc in results:
        meta = doc.metadata
        content = doc.page_content
        
        # == 定位逻辑 ==
        if meta["type"] == "image_frame":
            print(f"🔴 [找到图片] 属于模型: {meta['scene_id']}")
            print(f"   图片路径: {meta['file_path']}")
            print(f"   画面描述: {content[:30]}...")
            context_list.append(f"图片(属于{meta['scene_id']}): {content}")
            
        elif meta["type"] == "model_summary":
            print(f"🔵 [找到模型] ID: {meta['scene_id']}")
            print(f"   模型路径: {meta['file_path']}")
            print(f"   模型描述: {content[:30]}...")
            context_list.append(f"模型整体信息: {content}")
            
    return context_list

# === 核心函数 4: 国产 LLM 回答 ===
def chat_with_qwen(query, context_list):
    # 使用 Qwen-Turbo 或 Qwen-Max
    llm = ChatTongyi(model_name="qwen-turbo") 
    
    context_str = "\n".join(context_list)
    prompt = f"""
    基于以下检索到的多模态数据（包含模型信息和图片细节）回答用户问题：
    {context_str}
    
    用户问题: {query}
    """
    
    response = llm.invoke(prompt)
    print("\n🤖 Qwen 回答:", response.content)

# === 运行示例 ===
if __name__ == "__main__":
    # 1. 只有数据变动时运行
    # docs = load_and_bind_data(DATA_DIR)
    # build_index(docs)
    
    # 2. 检索测试
    # 场景：我想找某个模型的俯视图
    search_results = search_engine("有没有正上方俯拍的视角？")
    chat_with_qwen("有没有正上方俯拍的视角？", search_results)