import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. 加载 .env 文件 (默认加载当前目录下的 .env)
load_dotenv()

# 2. 从环境变量中读取配置
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# 检查是否成功读取 (可选，为了调试)
if not url or not key:
    raise ValueError("未找到 SUPABASE_URL 或 SUPABASE_KEY，请检查 .env 文件")

# 3. 创建客户端
supabase: Client = create_client(url, key)

# 测试一下
print("客户端初始化成功，URL:", url)