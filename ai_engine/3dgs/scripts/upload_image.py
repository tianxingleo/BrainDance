#!/usr/bin/env python3
"""
Supabase 图片上传工具

使用方法:
    python scripts/upload_image.py --scene_id sam3d_scene --user_id user1
    python scripts/upload_image.py --file /path/to/image.png --scene_id sam3d_scene
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

def upload_image(
    file_path: str,
    scene_id: str,
    user_id: str = "user1",
    bucket_name: str = "braindance-assets"
):
    """上传图片到 Supabase Storage"""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ 缺少 Supabase 配置，请检查 .env 文件")
        print("需要: SUPABASE_URL, SUPABASE_KEY")
        return False
    
    try:
        client = create_client(supabase_url, supabase_key)
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 上传路径
        storage_path = f"{user_id}/{scene_id}/raw/image.png"
        
        print(f"📤 上传中: {file_path.name} → {bucket_name}/{storage_path}")
        
        with open(file_path, "rb") as f:
            result = client.storage.from_(bucket_name).upload(
                path=storage_path,
                file=f,
                file_options={"x-upsert": "true"}
            )
        
        print(f"✅ 上传成功!")
        
        # 获取公开URL
        public_url = client.storage.from_(bucket_name).get_public_url(storage_path)
        print(f"🔗 公开链接: {public_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="上传图片到 Supabase Storage")
    parser.add_argument("--file", "-f", help="图片文件路径")
    parser.add_argument("--scene_id", "-s", required=True, help="场景 ID")
    parser.add_argument("--user_id", "-u", default="user1", help="用户 ID")
    parser.add_argument("--bucket", "-b", default="braindance-assets", help="存储桶名称")
    
    args = parser.parse_args()
    
    # 如果没有指定文件，使用默认测试图片
    if not args.file:
        test_image = Path(__file__).parent.parent / "test_data/images/test_image.png"
        if test_image.exists():
            args.file = str(test_image)
        else:
            print("❌ 未指定文件，且默认测试图片不存在")
            return
    
    success = upload_image(
        file_path=args.file,
        scene_id=args.scene_id,
        user_id=args.user_id,
        bucket_name=args.bucket
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
