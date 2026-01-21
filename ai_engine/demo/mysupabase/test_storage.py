# 引入操作系统模块，用于读取环境变量（如 SUPABASE_URL）
import os

# 引入 Supabase 的客户端构建器和类型提示
from supabase import create_client, Client

# 引入 dotenv 库，用于读取本地的 .env 文件
# 这是为了安全：不要把密钥直接写在代码里，防止上传到 GitHub 被盗用
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件内容到系统环境变量中
# 只有执行了这一步，os.getenv() 才能读到文件里的值
load_dotenv()

# ================= 配置区域 =================

# 3. 读取 Supabase 的项目地址
# 对应 .env 文件里的 SUPABASE_URL
SUPABASE_URL = os.getenv("SUPABASE_URL")

# 读取 Supabase 的密钥
# ⚠️ 关键点：后端脚本(Worker)必须使用 SERVICE_KEY (Service Role Key)
# 区别：
# - ANON_KEY: 给前端用的，权限受限，必须遵守 RLS (行级安全策略)
# - SERVICE_KEY: 给后端用的，拥有“上帝权限”，可以绕过 RLS 读写所有数据
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") 

# 定义我们要操作的存储桶名称
# 必须先在 Supabase 网页控制台 -> Storage -> New Bucket 创建好这个名字
BUCKET_NAME = "braindance-assets"

# 模拟的测试数据 (在真实业务中，这些通常来自数据库的任务表)
TEST_USER_ID = "user_test_001"   # 模拟用户 ID
TEST_SCENE_ID = "scene_demo_01"  # 模拟场景/项目 ID

# 初始化 Supabase 客户端实例
# 这一步建立连接，后续所有操作都通过这个 `supabase` 变量调用
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 核心功能函数 =================

def test_upload_file():
    """
    测试：上传文件
    模拟场景：前端 Flutter 把视频传上来，或者是 Worker 把处理好的结果传回去
    """
    print("\n--- 1. 开始测试上传 ---")
    
    # 构造文件存储路径 (Key)
    # Supabase 存储没有真实的文件夹，是通过 "/" 分隔符来模拟文件夹结构的
    # 结果类似：user_test_001/scene_demo_01/raw/test_log.txt
    file_path = f"{TEST_USER_ID}/{TEST_SCENE_ID}/raw/test_log.txt"
    
    # 模拟文件内容的二进制数据 (bytes)
    # 在实际场景中，这里通常是: with open("video.mp4", "rb") as f: file_content = f.read()
    file_content = b"Hello Supabase! This is a test file from Python."
    
    try:
        # 1. supabase.storage.from_(BUCKET_NAME) -> 选中某个桶
        #    注意：Python 中 `from` 是关键字，所以 SDK 里方法名叫 `from_` (带下划线)
        # 2. .upload() -> 执行上传
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,   # 存到云端的哪个位置
            file=file_content, # 要存的内容 (二进制流)
            file_options={
                "content-type": "text/plain", # 告诉浏览器这是文本 (如果是视频用 video/mp4)
                "upsert": "true"              # ⚠️ 重要：如果文件已存在，直接覆盖 (Update + Insert)
            }
        )
        print(f"✅ 上传成功！")
        print(f"   存储路径: {file_path}")
        
        # 返回上传成功的路径，供后面的函数测试用
        return file_path
        
    except Exception as e:
        # 如果网络不通、桶不存在、或者权限不足，会捕获异常
        print(f"❌ 上传失败: {e}")
        return None

def test_list_files():
    """
    测试：列出文件
    模拟场景：检查用户是否真的上传了视频，或者查看某个目录下有哪些文件
    """
    print("\n--- 2. 开始测试列表查询 ---")
    
    # 指定要查询的“文件夹”路径
    # 注意：不要包含具体文件名，只要目录部分
    folder_path = f"{TEST_USER_ID}/{TEST_SCENE_ID}/raw"
    
    try:
        # .list() -> 获取指定目录下的文件元数据列表
        files = supabase.storage.from_(BUCKET_NAME).list(folder_path)
        
        if files:
            print(f"✅ 在 '{folder_path}' 下找到了 {len(files)} 个文件:")
            for f in files:
                # f 是一个字典，包含 name, id, metadata, created_at 等信息
                print(f"   - {f['name']} (大小: {f['metadata']['size']} bytes)")
        else:
            print("⚠️ 文件夹是空的 (或者路径不对)")
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")

def test_get_signed_url(cloud_path):
    """
    测试：获取临时下载链接 (Signed URL)
    模拟场景：你的桶是 Private (私有) 的，不能直接访问。
    你需要生成一个带“签名令牌”的临时链接给前端展示图片，或者给外部工具下载。
    """
    print("\n--- 3. 开始测试生成下载链接 ---")
    
    try:
        # .create_signed_url(路径, 有效期秒数)
        # 这里设置 60 秒后链接失效，保证安全
        response = supabase.storage.from_(BUCKET_NAME).create_signed_url(cloud_path, 60)
        
        # 处理 SDK 返回值的兼容性问题
        # 新版 SDK 直接返回字符串 URL，旧版可能返回包含 signedURL 键的字典
        signed_url = response.get('signedURL') if isinstance(response, dict) else response
        
        print(f"✅ 签名链接生成成功 (有效期60秒):")
        print(f"   {signed_url}")
        print("   👉 这个链接可以在任何浏览器直接打开，不受权限限制(直到过期)。")
        
    except Exception as e:
        print(f"❌ 生成链接失败: {e}")

def test_download_bytes(cloud_path):
    """
    测试：下载文件流
    模拟场景：Python Worker 需要把视频下载到内存或存到本地硬盘，然后喂给 3DGS 算法进行处理。
    """
    print("\n--- 4. 开始测试直接下载文件流 ---")
    
    try:
        # .download() -> 直接返回文件的二进制数据 (bytes)
        # 注意：对于大文件 (如 1GB 视频)，建议使用流式下载 (stream)，这里演示的是小文件一次性下载
        data = supabase.storage.from_(BUCKET_NAME).download(cloud_path)
        
        print(f"✅ 下载成功！文件大小: {len(data)} bytes")
        
        # 因为我们上传的是文本，所以可以 decode 成字符串打印出来
        # 如果是视频或图片，这里不能 decode，而是应该 open('local.mp4', 'wb').write(data)
        print(f"   文件内容: {data.decode('utf-8')}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")

# ================= 主程序入口 =================
# 当你运行 `python test_storage.py` 时，会执行下面的代码
if __name__ == "__main__":
    
    # 1. 执行上传，并获取上传后的路径
    uploaded_path = test_upload_file()
    
    # 只有上传成功了，才继续后面的测试
    if uploaded_path:
        
        # 2. 确认文件是否真的存在于列表中
        test_list_files()
        
        # 3. 获取给前端用的 URL (这是给用户看的)
        test_get_signed_url(uploaded_path)
        
        # 4. 获取给 AI 算法用的二进制数据 (这是给 Worker 用的)
        test_download_bytes(uploaded_path)
    
    print("\n=== 测试结束 ===")