import os  # Python 标准库，用于操作系统交互，比如读取文件路径、环境变量
import json  # Python 标准库，用于解析 JSON 格式的数据
# 从 LangChain 导入 OpenAI 的接口类
# OpenAIEmbeddings: 负责把文字转换成向量（一串数字）
# ChatOpenAI: 负责调用 GPT-3.5/GPT-4 进行对话
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 从 LangChain 社区包导入 Chroma 向量数据库的接口
# Chroma 是一个轻量级的向量数据库，可以运行在本地
from langchain_community.vectorstores import Chroma
# 导入 Document 对象。这是 LangChain 中最基础的数据单元
# 它包含两个属性：page_content (主要文本内容) 和 metadata (元数据/附加信息)
from langchain.schema import Document

# === 配置部分 ===
# 设置 OpenAI 的 API Key。LangChain 会自动读取这个环境变量来鉴权。
# 也可以在实例化类时通过 api_key 参数传递，但环境变量更安全。
os.environ["OPENAI_API_KEY"] = "sk-..." 

DATA_DIR = "./data"  # 定义数据源目录，假设你的 json 和图片都放在这
DB_DIR = "./chroma_db_local" # 定义向量数据库在硬盘上的存储路径，避免每次运行都要重新生成

# === 1. 数据加载与清洗 (Load & Process) ===
def load_data_from_local(directory):
    """
    遍历本地文件夹，读取 JSON，并构建带有本地路径 Metadata 的文档对象。
    """
    documents = [] # 用于存放处理好的 Document 对象列表
    print(f"正在扫描目录: {directory} ...")
    
    # os.walk 会递归遍历文件夹，root是当前路径，files是该路径下的文件列表
    for root, _, files in os.walk(directory):
        for file in files:
            # 只处理 .json 后缀的文件
            if file.endswith(".json"):
                full_path = os.path.join(root, file) # 拼接成完整的文件绝对路径
                
                # 读取 JSON 内容到内存变量 data 中
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    # 如果文件损坏或格式不对，打印错误并跳过
                    print(f"读取失败 {file}: {e}")
                    continue

                # --- 策略 A: 处理单帧描述 (frame_xxxxx.json) ---
                # 判断逻辑：如果 json 里有 'description' 和 'filename' 字段，说明它是单帧数据的元数据
                if "description" in data and "filename" in data:
                    # 获取该 json 对应的图片文件名
                    image_filename = data["filename"]
                    # 拼凑出图片的本地路径（假设图片和 json 在同一目录下）
                    image_local_path = os.path.join(root, image_filename)
                    
                    # 构造 Metadata (元数据)
                    # Metadata 非常重要！它不会被向量化，但会被原样存下来。
                    # 当 RAG 检索到这段文字时，我们需要通过 metadata 找回原始的图片在哪里。
                    meta = {
                        "source": "frame",          # 标记数据来源类型
                        "file_path": image_local_path, # <--- 核心：存下本地图片的路径，方便最后展示
                        "quality": data.get("quality_score", 0), # 存一下质量分，以后也许能用来过滤
                        "filename": image_filename
                    }
                    
                    # 创建 LangChain 的 Document 对象
                    # page_content: 这部分文字会被送去计算向量，用于语义搜索 ("找俯拍的图片")
                    # metadata: 绑定的附属信息
                    doc = Document(page_content=data["description"], metadata=meta)
                    documents.append(doc) # 加入列表

                # --- 策略 B: 处理场景总览 (scene_summary.json) ---
                # 判断逻辑：如果 json 里有 'summary' 字段
                elif "summary" in data:
                    meta = {
                        "source": "scene",
                        "file_path": full_path, # 场景总结没有单一图片，这里存 json 文件的路径
                        # 使用 .get 防止字段不存在报错，默认值为 "unknown"
                        "mode": data.get("shooting_strategy", {}).get("mode", "unknown")
                    }
                    # 这里把 summary 字段作为主要搜索内容
                    doc = Document(page_content=data["summary"], metadata=meta)
                    documents.append(doc)
                    
    print(f"构建了 {len(documents)} 个文档对象。")
    return documents

# === 2. 向量化与存储 (Index) ===
def build_or_update_index(documents):
    """
    将文档向量化并存入本地 ChromaDB。
    """
    # 初始化 Embeddings 模型。
    # 这会调用 OpenAI 的接口 (text-embedding-3-small 等)，把文字变成高维数组。
    # 注意：这一步会消耗 token。
    embedding_model = OpenAIEmbeddings() 
    
    # 核心步骤：从文档构建向量库
    # 1. 把 documents 里的 page_content 发送给 OpenAI 变成向量。
    # 2. 把 向量 + page_content + metadata 一起存入 Chroma 数据库。
    # 3. persist_directory 参数表示把数据保存到硬盘文件夹，否则程序结束数据就丢了。
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print("向量数据库构建完成。")
    return vector_db

# === 3. 检索 (Retrieve) ===
def retrieve_relevant_info(query, vector_db, k=2):
    """
    根据问题搜索最相关的 JSON 片段。
    """
    # similarity_search: 计算用户 query 的向量与数据库中所有向量的相似度 (通常是余弦相似度)
    # k=2: 只返回最相似的前 2 个文档
    results = vector_db.similarity_search(query, k=k)
    return results

# === 4. 生成回答 (Generate) ===
def generate_response(query, retrieved_docs):
    """
    让 LLM 根据检索到的信息回答问题。
    """
    # 初始化 LLM (大语言模型)
    # temperature=0.7: 让回答稍微灵活一点，0 为最严谨
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # 构建上下文 Context
    # 我们要把检索到的碎片信息拼成一段话，喂给 GPT
    context_text = ""
    referenced_files = [] # 用于收集被引用的文件路径
    
    for i, doc in enumerate(retrieved_docs):
        # 拼接文本内容
        context_text += f"信息片段 {i+1}: {doc.page_content}\n"
        # 提取 metadata 中的文件路径
        if "file_path" in doc.metadata:
            referenced_files.append(doc.metadata["file_path"])
    
    # 构造 Prompt (提示词)
    # 使用 f-string 将 context_text 和用户 query 嵌入模板
    prompt = f"""
    你是一个智能助手。请根据以下检索到的背景信息回答用户问题。
    
    [背景信息]
    {context_text}
    
    [用户问题]
    {query}
    
    请在回答中指明你参考了哪些具体的画面特征。
    """
    
    # invoke: 发送提示词给 GPT-4 并等待返回
    response = llm.invoke(prompt)
    
    # response.content 是 AI 回复的纯文本
    # referenced_files 是为了我们在前端或终端里知道 AI 看了哪些图
    return response.content, referenced_files

# === 主流程入口 ===
if __name__ == "__main__":
    # 1. 建库阶段 (注释掉是因为只有第一次运行或者有新数据时才跑)
    # 如果你是第一次运行，取消下面两行的注释：
    # docs = load_data_from_local(DATA_DIR) # 读取文件
    # db = build_or_update_index(docs)      # 生成向量并存盘
    
    # 2. 加载阶段
    # 初始化 embedding 模型 (检索时也需要把用户的问题变成向量，所以这里也需要它)
    embedding_model = OpenAIEmbeddings()
    # 直接从硬盘加载之前存好的数据库，不需要重新消耗 token 计算向量
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    
    # 3. 交互阶段
    user_query = "有没有那种正上方俯拍视角的图片，而且要清晰一点的？"
    
    # 第一步：去数据库找资料
    relevant_docs = retrieve_relevant_info(user_query, db)
    
    # 第二步：把资料和问题给 GPT 写回答
    answer, files = generate_response(user_query, relevant_docs)
    
    # 打印结果
    print("\n=== AI 回答 ===")
    print(answer)
    print("\n=== 关联的本地文件 ===")
    for f in files:
        print(f"File: {f}")