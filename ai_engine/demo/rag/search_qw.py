import os
import json
import sys

# === 核心依赖 ===
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# === 配置区域 ===
# 建议配置环境变量，或者在这里填入
# os.environ["DASHSCOPE_API_KEY"] = "sk-你的阿里云APIKey"

# 数据库和数据源路径
DB_PERSIST_DIR = "./chroma_db_v4"  # 向量库存储位置
DATA_SOURCE_DIR = "./data"         # 你的数据根目录

# === 全局单例：Embedding 模型 (使用 v4) ===
# 这样不仅入库用它，搜索也用它，保证向量空间一致
def get_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        # v4 支持特定参数优化，这里使用默认配置即可满足需求
    )

# ==========================================
# 功能模块 1: 数据入库 (Ingest)
# ==========================================
def ingest_data(root_directory):
    """
    遍历目录 -> 解析 JSON -> 绑定模型与图片 -> 存入向量库
    """
    print(f"\n🚀 开始扫描目录: {root_directory}")
    
    documents = []
    embedding_model = get_embeddings()
    
    # 1. 遍历根目录下的每一个子文件夹 (假设每个子文件夹是一个模型/场景)
    # 结构假设: ./data/scene_01/frame_001.json
    for scene_folder_name in os.listdir(root_directory):
        scene_path = os.path.join(root_directory, scene_folder_name)
        
        if not os.path.isdir(scene_path):
            continue
            
        print(f"   📂 处理模型文件夹: {scene_folder_name}")
        
        # --- 步骤 A: 获取该模型的“全局背景” (Scene Summary) ---
        scene_summary_text = ""
        scene_info = {}
        summary_path = os.path.join(scene_path, "scene_summary.json")
        
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                    # 提取关键信息，作为所有子图片的“背景知识”
                    scene_summary_text = scene_data.get("summary", "")
                    scene_info = scene_data # 存下来备用
            except Exception as e:
                print(f"      ⚠️ 读取场景总结失败: {e}")
        
        # --- 步骤 B: 遍历该文件夹下的所有图片 JSON ---
        for filename in os.listdir(scene_path):
            if filename.startswith("frame_") and filename.endswith(".json"):
                json_path = os.path.join(scene_path, filename)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                    
                    # === 核心逻辑：内容融合 (Context Fusion) ===
                    # 我们不仅存图片的描述，还把场景的描述加进去。
                    # 这样搜 "桌子上的笔" (场景信息) 或 "俯拍" (图片信息) 都能搜到这张图。
                    combined_text = f"""
                    [场景背景]: {scene_summary_text}
                    [画面细节]: {frame_data.get('description', '')}
                    [拍摄理由]: {frame_data.get('reason_for_selection', '')}
                    """
                    
                    # === 构建元数据 (Metadata) ===
                    # 这些是搜索结果返回给你的“定位器”
                    img_filename = frame_data.get("filename", "")
                    meta = {
                        "scene_id": scene_folder_name,          # 属于哪个模型
                        "type": "image_frame",
                        "filename": img_filename,               # 图片文件名
                        "file_path": os.path.join(scene_path, img_filename), # 图片绝对路径
                        "quality": frame_data.get("quality_score", 0),
                        "parent_summary": scene_summary_text[:50] + "..." # 方便预览
                    }
                    
                    doc = Document(page_content=combined_text, metadata=meta)
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"      ❌ 解析 {filename} 失败: {e}")

    # 2. 批量写入数据库
    if documents:
        print(f"\n📦 正在将 {len(documents)} 条数据写入向量库 (ChromaDB)...")
        # from_documents 会自动初始化库并保存
        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=DB_PERSIST_DIR
        )
        print("✅ 入库完成！")
    else:
        print("⚠️ 未找到有效数据，跳过入库。")

# ==========================================
# 功能模块 2: 语义搜索 (Search)
# ==========================================
def search_assets(query_text, top_k=3):
    """
    输入文本 -> 搜索向量库 -> 返回具体的模型ID和图片路径
    """
    print(f"\n🔍 正在使用 text-embedding-v4 搜索: '{query_text}'")
    
    # 1. 加载数据库
    if not os.path.exists(DB_PERSIST_DIR):
        print("❌ 错误：数据库不存在。请先执行功能 [1] 进行数据入库。")
        return

    embedding_model = get_embeddings()
    db = Chroma(persist_directory=DB_PERSIST_DIR, embedding_function=embedding_model)
    
    # 2. 执行搜索 (使用 similarity_search_with_score 可以看到匹配分数)
    # score 越低越相似 (Chroma 默认是 L2 距离)
    results = db.similarity_search_with_score(query_text, k=top_k)
    
    # 3. 格式化输出
    print("-" * 50)
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        print(f"🏆 排名 {i+1} (匹配分: {score:.4f})")
        print(f"   📂 所属模型 (Scene): {meta.get('scene_id')}")
        print(f"   🖼️ 图片文件 (File) : {meta.get('filename')}")
        print(f"   📍 完整路径: {meta.get('file_path')}")
        print(f"   📝 内容摘要: {doc.page_content.strip().replace(chr(10), ' ')[:100]}...")
        print("-" * 50)

# ==========================================
# 主程序入口 (交互菜单)
# ==========================================
def main():
    while True:
        print("\n" + "="*30)
        print("   🤖 BrainDance 资产检索引擎")
        print("="*30)
        print("1. 📥 [入库] 扫描文件夹并建立索引")
        print("2. 🔍 [搜索] 查找模型或图片")
        print("q. 🚪 退出")
        
        choice = input("\n请选择功能 (1/2/q): ").strip().lower()
        
        if choice == '1':
            # 检查目录是否存在
            if not os.path.exists(DATA_SOURCE_DIR):
                print(f"❌ 目录 {DATA_SOURCE_DIR} 不存在，请检查配置。")
            else:
                ingest_data(DATA_SOURCE_DIR)
                
        elif choice == '2':
            query = input("请输入搜索内容 (例如: '俯拍视角的白色笔'): ")
            if query:
                search_assets(query)
                
        elif choice == 'q':
            print("再见！")
            break
        else:
            print("输入无效，请重新选择。")

if __name__ == "__main__":
    # 检查 API Key
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量。")
        key = input("请输入你的阿里云 DashScope API Key: ")
        os.environ["DASHSCOPE_API_KEY"] = key.strip()
        
    main()