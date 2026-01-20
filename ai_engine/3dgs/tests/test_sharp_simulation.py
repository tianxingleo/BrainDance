#!/usr/bin/env python3
"""
Sharp单图生成3DGS模拟测试程序

功能：
1. 从Supabase消息队列读取Sharp相关任务
2. 将任务状态修改为pending
3. 执行Sharp Pipeline进行模拟测试
4. 验证处理结果和状态变更

使用：
    cd ai_engine/3dgs
    conda activate gs_linux_backup
    python tests/test_sharp_simulation.py --help
"""
import sys
import os
import time
import argparse
import subprocess
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


class SharpSimulationTest:
    """Sharp模拟测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "braindance-assets")
        self.table_name = os.getenv("SUPABASE_TABLE", "processing_tasks")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("❌ 缺少Supabase配置，请检查.env文件")
        
        # 保存并清除代理环境变量（Supabase + SOCKS 代理存在兼容性问题）
        self._saved_env = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                self._saved_env[var] = os.environ[var]
                del os.environ[var]
        
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
        except ImportError as e:
            # 恢复环境变量并抛出错误
            for var, value in self._saved_env.items():
                os.environ[var] = value
            raise ValueError(f"Supabase客户端初始化失败: {e}\n解决方案: pip install httpx[socks]")
        except Exception as e:
            # 恢复环境变量并抛出错误
            for var, value in self._saved_env.items():
                os.environ[var] = value
            raise
        
        self.test_results = []
        
    def __del__(self):
        """析构函数：确保代理环境变量被恢复"""
        for var, value in self._saved_env.items():
            os.environ[var] = value
        
    def find_sharp_tasks(self, limit: int = 10) -> List[Dict]:
        """
        查询Sharp相关的任务列表
        
        Args:
            limit: 返回数量限制
            
        Returns:
            任务列表
        """
        print(f"\n🔍 查询Sharp相关任务...")
        
        try:
            response = self.client.table(self.table_name) \
                .select("*") \
                .eq("task_type", "single_image_sharp") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
                
            tasks = response.data
            print(f"✅ 找到 {len(tasks)} 个Sharp任务")
            
            for i, task in enumerate(tasks, 1):
                status = task.get("status", "unknown")
                created_at = task.get("created_at", "")
                print(f"  {i}. [{status}] {task['scene_id']} - {created_at}")
                
            return tasks
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []
            
    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """根据任务ID获取任务详情"""
        print(f"\n🔍 查询任务: {task_id}")
        
        try:
            response = self.client.table(self.table_name) \
                .select("*") \
                .eq("id", task_id) \
                .execute()
                
            if response.data:
                task = response.data[0]
                print(f"✅ 找到任务: {task['scene_id']}")
                print(f"   状态: {task.get('status')}")
                print(f"   类型: {task.get('task_type')}")
                return task
            else:
                print("❌ 任务不存在")
                return None
                
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return None
            
    def update_task_status(self, task_id: str, status: str = "pending",
                          clear_logs: bool = True) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            clear_logs: 是否清空日志
            
        Returns:
            是否成功
        """
        print(f"\n📝 更新任务状态: {task_id} → {status}")
        
        try:
            update_data = {"status": status}
            
            if clear_logs:
                update_data["logs"] = []
                
            response = self.client.table(self.table_name) \
                .update(update_data) \
                .eq("id", task_id) \
                .execute()
                
            if response.data:
                print(f"✅ 状态更新成功")
                return True
            else:
                print("❌ 状态更新失败")
                return False
                
        except Exception as e:
            print(f"❌ 更新失败: {e}")
            return False
            
    def create_test_task(self, scene_id: str = None, user_id: str = "test_user") -> Dict:
        """
        创建新的测试任务
        
        Args:
            scene_id: 场景ID（可选，自动生成）
            user_id: 用户ID
            
        Returns:
            创建的任务信息
        """
        print(f"\n📝 创建测试任务...")
        
        if not scene_id:
            scene_id = f"test_sharp_simulation_{int(time.time())}"
            
        task_id = str(uuid.uuid4())  # 使用有效的UUID格式
        
        task_data = {
            "id": task_id,
            "scene_id": scene_id,
            "user_id": user_id,
            "task_type": "single_image_sharp",
            "task_params": {
                "quality": "high",
                "simulation_test": True
            },
            "status": "pending",
        }
        
        try:
            response = self.client.table(self.table_name).insert(task_data).execute()
            print(f"✅ 测试任务创建成功: {task_id}")
            print(f"   场景ID: {scene_id}")
            print(f"   用户ID: {user_id}")
            return task_data
            
        except Exception as e:
            print(f"❌ 任务创建失败: {e}")
            raise
            
    def upload_test_image(self, image_path: str, scene_id: str,
                          user_id: str = "test_user") -> bool:
        """
        上传测试图片到Storage
        
        Args:
            image_path: 图片路径
            scene_id: 场景ID
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        print(f"\n📤 上传测试图片...")
        
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ 图片不存在: {image_path}")
            return False
            
        storage_path = f"{user_id}/{scene_id}/raw/image.png"
        
        try:
            with open(image_path, "rb") as f:
                self.client.storage.from_(self.bucket_name).upload(
                    path=storage_path,
                    file=f,
                    file_options={"x-upsert": "true"}
                )
                
            print(f"✅ 图片上传成功: {storage_path}")
            return True
            
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return False
            
    def run_pipeline_locally(self, task_id: str, scene_id: str) -> bool:
        """
        本地运行Pipeline（直接调用，不经过Worker轮询）
        
        Args:
            task_id: 任务ID
            scene_id: 场景ID
            
        Returns:
            是否成功
        """
        print(f"\n🚀 启动Pipeline执行...")
        print(f"   任务ID: {task_id}")
        print(f"   场景ID: {scene_id}")
        
        # 准备工作目录
        work_dir = project_root / "temp_workspace" / scene_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集日志
        logs = []
        
        def log_callback(message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = {"ts": int(time.time()), "msg": message}
            logs.append(log_entry)
            print(f"  [{timestamp}] {message}")
            
        # 创建上下文（模拟Worker）
        context = {
            "task_id": task_id,
            "scene_id": scene_id,
            "work_root": str(work_dir),
            "log_callback": log_callback,
        }
        
        # 获取Pipeline
        print("\n🔧 加载Pipeline...")
        try:
            from src.core.factory import PipelineFactory
            pipeline = PipelineFactory.get_pipeline("single_image_sharp", context)
            print(f"✅ Pipeline加载成功: {pipeline.__class__.__name__}")
        except Exception as e:
            print(f"❌ Pipeline加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # 执行Pipeline
        print("\n🔥 执行Sharp Pipeline...")
        try:
            # 查找输入图片
            input_image = Path.home() / "braindance_workspace" / f"{scene_id}.png"
            
            if not input_image.exists():
                # 尝试从Storage下载
                print(f"📥 从Storage下载图片...")
                try:
                    storage_path = f"test_user/{scene_id}/raw/image.png"
                    response = self.client.storage.from_(self.bucket_name).download(storage_path)
                    with open(input_image, "wb") as f:
                        f.write(response)
                    print(f"✅ 下载成功: {input_image}")
                except Exception as download_error:
                    print(f"❌ 下载失败: {download_error}")
                    print("请先使用 --upload-image 参数上传测试图片")
                    return False
                    
            # 执行
            ply_path, metadata = pipeline.run(str(input_image), {})
            
            print(f"\n✅ Pipeline执行成功!")
            print(f"📂 输出文件: {ply_path}")
            print(f"📊 元数据: {metadata}")
            
            # 记录结果
            self.test_results.append({
                "task_id": task_id,
                "scene_id": scene_id,
                "ply_path": ply_path,
                "metadata": metadata,
                "logs_count": len(logs),
                "success": True
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            self.test_results.append({
                "task_id": task_id,
                "scene_id": scene_id,
                "success": False,
                "error": str(e)
            })
            
            return False
            
    def poll_task_status(self, task_id: str, timeout: int = 600,
                         check_interval: int = 5) -> Optional[Dict]:
        """
        轮询任务状态直到完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
            
        Returns:
            最终任务状态
        """
        print(f"\n🔄 轮询任务状态: {task_id}")
        print(f"   超时: {timeout}秒")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.client.table(self.table_name) \
                    .select("*") \
                    .eq("id", task_id) \
                    .execute()
                    
                if response.data:
                    task = response.data[0]
                    status = task.get("status", "unknown")
                    logs = task.get("logs", [])
                    
                    elapsed = int(time.time() - start_time)
                    print(f"   [{elapsed}s] 状态: {status}", end="")
                    
                    if logs:
                        last_log = logs[-1]
                        print(f" | 最新: {last_log.get('msg', '')[:40]}...", end="")
                        
                    print()
                    
                    if status == "completed":
                        print("\n✅ 任务完成!")
                        return task
                    elif status == "failed":
                        print("\n❌ 任务失败")
                        return task
                        
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"❌ 查询失败: {e}")
                time.sleep(check_interval)
                
        print("⏰ 轮询超时")
        return None
        
    def run_full_simulation(self, task_id: str = None, create_new: bool = False,
                           image_path: str = None) -> bool:
        """
        运行完整的模拟测试流程
        
        Args:
            task_id: 任务ID（可选）
            create_new: 是否创建新任务
            image_path: 测试图片路径
            
        Returns:
            是否成功
        """
        print("=" * 70)
        print("Sharp单图生成3DGS模拟测试")
        print("=" * 70)
        
        try:
            # 1. 获取或创建任务
            if create_new:
                scene_id = f"sharp_sim_{int(time.time())}"
                task = self.create_test_task(scene_id)
                task_id = task["id"]
                print(f"\n📌 使用新任务: {task_id}")
            elif task_id:
                task = self.get_task_by_id(task_id)
                if task:
                    scene_id = task["scene_id"]
                else:
                    return False
            else:
                # 查找最新的Sharp任务
                tasks = self.find_sharp_tasks(1)
                if tasks:
                    task = tasks[0]
                    task_id = task["id"]
                    scene_id = task["scene_id"]
                    print(f"\n📌 使用最新任务: {task_id}")
                else:
                    print("❌ 没有找到Sharp任务，请使用 --create-test 创建新任务")
                    return False
                    
            # 2. 如果需要，上传测试图片
            if image_path:
                if not self.upload_test_image(image_path, scene_id):
                    return False
                    
            # 3. 将状态改为pending
            if not self.update_task_status(task_id, "pending"):
                return False
                
            # 4. 启动Pipeline执行
            if not self.run_pipeline_locally(task_id, scene_id):
                # 更新任务状态为失败
                self.update_task_status(task_id, "failed")
                return False
                
            # 5. 更新任务状态为完成
            if not self.update_task_status(task_id, "completed"):
                print("⚠️ 状态更新失败，但Pipeline执行成功")
                
            # 6. 显示结果摘要
            print("\n" + "=" * 70)
            print("测试结果摘要")
            print("=" * 70)
            
            for result in self.test_results:
                status = "✅ 成功" if result["success"] else "❌ 失败"
                print(f"\n任务: {result['task_id']}")
                print(f"  状态: {status}")
                print(f"  日志数: {result.get('logs_count', 0)}")
                
                if result["success"]:
                    print(f"  输出: {result.get('ply_path', 'N/A')}")
                else:
                    print(f"  错误: {result.get('error', 'Unknown')}")
                    
            return any(r["success"] for r in self.test_results)
            
        except Exception as e:
            print(f"\n❌ 测试过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Sharp单图生成3DGS模拟测试程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 创建新测试任务并运行
  python tests/test_sharp_simulation.py --create-test --upload-image /path/to/image.png
  
  # 使用指定任务ID运行
  python tests/test_sharp_simulation.py --task-id <task_id>
  
  # 查看最新的Sharp任务
  python tests/test_sharp_simulation.py --list
  
  # 轮询任务状态
  python tests/test_sharp_simulation.py --poll <task_id>
        """
    )
    
    parser.add_argument("--create-test", action="store_true",
                       help="创建新的测试任务")
    parser.add_argument("--task-id", type=str,
                       help="指定任务ID")
    parser.add_argument("--latest", action="store_true",
                       help="使用最新的Sharp任务")
    parser.add_argument("--list", action="store_true",
                       help="列出所有Sharp任务")
    parser.add_argument("--upload-image", type=str,
                       help="指定要上传的测试图片路径")
    parser.add_argument("--poll", type=str, metavar="TASK_ID",
                       help="轮询指定任务的状态")
    parser.add_argument("--timeout", type=int, default=600,
                       help="轮询超时时间（秒），默认600秒")
    
    args = parser.parse_args()
    
    # 初始化测试
    try:
        test = SharpSimulationTest()
    except ValueError as e:
        print(e)
        return 1
        
    # 执行操作
    if args.list:
        # 列出任务
        test.find_sharp_tasks()
        return 0
        
    elif args.poll:
        # 轮询任务状态
        result = test.poll_task_status(args.poll, args.timeout)
        if result and result.get("status") == "completed":
            return 0
        else:
            return 1
            
    elif args.create_test or args.task_id or args.latest:
        # 运行完整模拟测试
        success = test.run_full_simulation(
            task_id=args.task_id,
            create_new=args.create_test,
            image_path=args.upload_image
        )
        return 0 if success else 1
        
    else:
        # 默认显示帮助
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit(main())
