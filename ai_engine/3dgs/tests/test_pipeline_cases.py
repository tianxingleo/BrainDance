"""
测试用例集：SingleImageSAM3DPipeline

三种测试模式：
1. Unit Test: 直接调用 pipeline，无需任何外部依赖
2. Mock Worker Test: 模拟 Worker 上下文，验证日志回调
3. Integration Test: 集成 Supabase 测试 (需要真实凭证)
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.core.factory import PipelineFactory
from src.config import PipelineConfig
from src.pipelines.single_image_sam3d import SingleImageSAM3DPipeline


class TestConfig:
    @staticmethod
    def get_test_image():
        """获取测试图片路径"""
        test_image = project_root.parent / "demo/SAM3d/test_input.png"
        if not test_image.exists():
            raise FileNotFoundError(f"测试图片不存在: {test_image}")
        return str(test_image)

    @staticmethod
    def get_test_context(work_dir: str = None):
        """获取模拟的 Worker 上下文"""
        if work_dir is None:
            work_dir = str(project_root / "output/test_unit")
        return {
            "task_id": "test_task_001",
            "scene_id": "test_scene_001",
            "work_root": work_dir,
            "log_callback": lambda msg: print(f"[LOG] {msg}"),
        }


class TestSAM3DPipeline:
    """SAM3D Pipeline 单元测试"""

    def test_01_pipeline_creation(self):
        """测试 01: Pipeline 实例化"""
        print("\n" + "=" * 50)
        print("测试 01: Pipeline 实例化")
        print("=" * 50)

        context = TestConfig.get_test_context()
        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        assert pipeline is not None
        assert isinstance(pipeline, SingleImageSAM3DPipeline)
        assert pipeline.task_id == "test_task_001"
        assert pipeline.scene_id == "test_scene_001"
        print("✅ Pipeline 实例化成功")
        return True

    def test_02_pipeline_run_without_mask(self):
        """测试 02: Pipeline 执行 (自动 Mask)"""
        print("\n" + "=" * 50)
        print("测试 02: Pipeline 执行 (自动生成 Mask)")
        print("=" * 50)

        context = TestConfig.get_test_context(str(project_root / "output/test_auto_mask"))
        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        input_image = TestConfig.get_test_image()
        config = PipelineConfig()

        params = {
            "repo_path": str(config.sam3d_repo_path),
            "model_dir": str(config.sam3d_checkpoint_dir),
        }

        ply_path, metadata = pipeline.run(input_image, params)

        assert ply_path is not None
        assert Path(ply_path).exists()
        assert metadata["engine"] == "sam3d"
        assert "original_image" in metadata

        print(f"✅ 生成模型: {ply_path}")
        print(f"✅ 元数据: {metadata}")
        return True

    def test_03_pipeline_run_with_mask(self):
        """测试 03: Pipeline 执行 (指定 Mask)"""
        print("\n" + "=" * 50)
        print("测试 03: Pipeline 执行 (指定自定义 Mask)")
        print("=" * 50)

        context = TestConfig.get_test_context(str(project_root / "output/test_custom_mask"))
        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        input_image = TestConfig.get_test_image()
        config = PipelineConfig()

        # 使用现有的自动生成的 mask
        auto_mask = str(project_root / "output/test_auto_mask/temp_test_input_mask.png")

        params = {
            "repo_path": str(config.sam3d_repo_path),
            "model_dir": str(config.sam3d_checkpoint_dir),
            "mask_path": auto_mask,
        }

        ply_path, metadata = pipeline.run(input_image, params)

        assert ply_path is not None
        assert Path(ply_path).exists()

        print(f"✅ 使用自定义 Mask 生成: {ply_path}")
        return True

    def test_04_pipeline_with_log_callback(self):
        """测试 04: 验证日志回调功能"""
        print("\n" + "=" * 50)
        print("测试 04: 日志回调功能")
        print("=" * 50)

        logs = []

        def mock_log_callback(message):
            logs.append(message)
            print(f"[回调日志] {message}")

        context = TestConfig.get_test_context(str(project_root / "output/test_log"))
        context["log_callback"] = mock_log_callback

        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        input_image = TestConfig.get_test_image()
        config = PipelineConfig()

        params = {
            "repo_path": str(config.sam3d_repo_path),
            "model_dir": str(config.sam3d_checkpoint_dir),
        }

        ply_path, metadata = pipeline.run(input_image, params)

        # 验证日志回调被调用
        assert len(logs) > 0
        assert any("SAM3D" in log for log in logs)
        assert any("启动" in log or "生成" in log or "完成" in log for log in logs)

        print(f"✅ 捕获日志 {len(logs)} 条")
        print(f"✅ 日志回调工作正常")
        return True

    def test_05_pipeline_error_handling(self):
        """测试 05: 错误处理"""
        print("\n" + "=" * 50)
        print("测试 05: 错误处理")
        print("=" * 50)

        context = TestConfig.get_test_context(str(project_root / "output/test_error"))
        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        # 使用不存在的图片
        params = {
            "repo_path": str(PipelineConfig().sam3d_repo_path),
            "model_dir": str(PipelineConfig().sam3d_checkpoint_dir),
        }

        try:
            pipeline.run("/nonexistent/image.png", params)
            print("❌ 应该抛出异常")
            return False
        except Exception as e:
            print(f"✅ 正确捕获异常: {type(e).__name__}")
            return True


class TestSAM3DMockWorker:
    """模拟 Worker 测试"""

    def test_mock_worker_execution(self):
        """模拟 Worker 执行完整流程"""
        print("\n" + "=" * 50)
        print("模拟 Worker 执行流程")
        print("=" * 50)

        logs = []

        def on_pipeline_log(message):
            log_entry = {
                "ts": 1234567890,
                "msg": message
            }
            logs.append(log_entry)
            print(f"[Worker Log] {message}")

        context = {
            "task_id": "mock_task_001",
            "scene_id": "mock_scene_001",
            "work_root": str(project_root / "output/test_mock_worker"),
            "log_callback": on_pipeline_log,
        }

        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

        config = PipelineConfig()
        input_image = TestConfig.get_test_image()

        params = {
            "repo_path": str(config.sam3d_repo_path),
            "model_dir": str(config.sam3d_checkpoint_dir),
        }

        ply_path, metadata = pipeline.run(input_image, params)

        print(f"✅ Worker 模拟执行成功")
        print(f"✅ 生成日志 {len(logs)} 条")
        print(f"✅ 最终输出: {ply_path}")

        return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀" * 20)
    print("SAM3D Pipeline 测试套件")
    print("🚀" * 20)

    tester = TestSAM3DPipeline()
    worker_tester = TestSAM3DMockWorker()

    results = []

    try:
        results.append(("Pipeline 实例化", tester.test_01_pipeline_creation()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("Pipeline 实例化", False))

    try:
        results.append(("自动 Mask 生成", tester.test_02_pipeline_run_without_mask()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("自动 Mask 生成", False))

    try:
        results.append(("自定义 Mask 生成", tester.test_03_pipeline_run_with_mask()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("自定义 Mask 生成", False))

    try:
        results.append(("日志回调功能", tester.test_04_pipeline_with_log_callback()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("日志回调功能", False))

    try:
        results.append(("错误处理", tester.test_05_pipeline_error_handling()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("错误处理", False))

    try:
        results.append(("模拟 Worker 执行", worker_tester.test_mock_worker_execution()))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results.append(("模拟 Worker 执行", False))

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} | {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"总计: {passed} 通过, {failed} 失败")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
