import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ================= 配置区域 =================
# 3. 使用 os.getenv 读取变量
SUPABASE_URL = os.getenv("SUPABASE_URL")
# 注意：一定要用 SERVICE_KEY，不要用 ANNON_KEY
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") 

BUCKET_NAME = "braindance-assets"

# 模拟的测试数据
TEST_USER_ID = "user_test_001"
TEST_SCENE_ID = "scene_demo_01"

# 初始化客户端
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 核心功能函数 =================

def test_upload_file():
    """测试：上传一个文本文件（模拟上传视频）"""
    print("\n--- 1. 开始测试上传 ---")
    
    # 构造我们约定好的路径结构
    file_path = f"{TEST_USER_ID}/{TEST_SCENE_ID}/raw/test_log.txt"
    
    # 模拟文件内容（你可以换成 open('video.mp4', 'rb')）
    file_content = b"Hello Supabase! This is a test file from Python."
    
    try:
        # upsert=True 表示如果文件存在就覆盖，方便反复测试
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": "text/plain", "upsert": "true"}
        )
        print(f"✅ 上传成功！")
        print(f"   存储路径: {file_path}")
        # 如果是老版本SDK，response可能包含数据；新版本通常无报错即成功
        return file_path
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return None

def test_list_files():
    """测试：查看文件是否真的存在"""
    print("\n--- 2. 开始测试列表查询 ---")
    
    # 我们查询 user_test_001/scene_demo_01/raw/ 文件夹下的内容
    folder_path = f"{TEST_USER_ID}/{TEST_SCENE_ID}/raw"
    
    try:
        files = supabase.storage.from_(BUCKET_NAME).list(folder_path)
        if files:
            print(f"✅ 在 '{folder_path}' 下找到了 {len(files)} 个文件:")
            for f in files:
                print(f"   - {f['name']} (大小: {f['metadata']['size']} bytes)")
        else:
            print("⚠️ 文件夹是空的 (或者路径不对)")
    except Exception as e:
        print(f"❌ 查询失败: {e}")

def test_get_signed_url(cloud_path):
    """测试：获取下载链接 (因为桶是私有的，必须用签名URL)"""
    print("\n--- 3. 开始测试生成下载链接 ---")
    
    try:
        # 生成一个有效期 60 秒的链接
        response = supabase.storage.from_(BUCKET_NAME).create_signed_url(cloud_path, 60)
        
        # 兼容不同版本的 SDK 返回格式
        signed_url = response.get('signedURL') if isinstance(response, dict) else response
        
        print(f"✅ 签名链接生成成功 (有效期60秒):")
        print(f"   {signed_url}")
        print("   👉 你可以将这个链接粘贴到浏览器里，看看能不能下载。")
    except Exception as e:
        print(f"❌ 生成链接失败: {e}")

def test_download_bytes(cloud_path):
    """测试：直接下载文件流到内存 (适合 Python 处理脚本)"""
    print("\n--- 4. 开始测试直接下载文件流 ---")
    
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(cloud_path)
        print(f"✅ 下载成功！文件大小: {len(data)} bytes")
        print(f"   文件内容: {data.decode('utf-8')}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 上传
    uploaded_path = test_upload_file()
    
    if uploaded_path:
        # 2. 确认文件存在
        test_list_files()
        
        # 3. 获取给前端展示用的 URL
        test_get_signed_url(uploaded_path)
        
        # 4. Worker 下载数据进行处理
        test_download_bytes(uploaded_path)
    
    print("\n=== 测试结束 ===")