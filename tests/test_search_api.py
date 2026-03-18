#!/usr/bin/env python3
"""
文件: tests/test_search_api.py
功能: 测试 search-models Edge Function
作者: BrainDance Team
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# 尝试加载 python-dotenv（可选）
try:
    from dotenv import load_dotenv
    # 自动查找并加载 .env 文件
    env_paths = [
        Path(__file__).parent / '.env.local',  # tests/.env.local
        Path(__file__).parent.parent / 'ai_engine' / '3dgs' / '.env',  # ai_engine/3dgs/.env
        Path(__file__).parent / '.env',  # tests/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ 已加载环境配置: {env_path.relative_to(Path(__file__).parent.parent)}")
            break
except ImportError:
    pass  # python-dotenv 未安装，跳过


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    passed: bool
    response_time: float
    details: str


class SearchAPITester:
    """搜索 API 测试类"""

    def __init__(self, base_url: str = None):
        # 优先级: 参数 > 环境变量 > 默认值
        if base_url is None:
            # 尝试从多个环境变量构建 URL
            search_api_url = os.getenv('SEARCH_API_URL')
            supabase_url = os.getenv('SUPABASE_URL')

            if search_api_url:
                base_url = search_api_url
            elif supabase_url:
                # 从 SUPABASE_URL 构建 Edge Function URL
                base_url = f"{supabase_url}/functions/v1/search-models"
            else:
                # 默认本地开发 URL
                base_url = 'http://127.0.0.1:54321/functions/v1/search-models'

        self.base_url = base_url
        self.api_key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY')
        print(f"Using API Key: {self.api_key}")
        self.results: List[TestResult] = []

    def send_request(self, query: str, threshold: float = 0.5) -> Dict[str, Any]:
        """发送搜索请求"""
        payload = {
            "query": query,
            "threshold": threshold
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.time()
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response_time = time.time() - start_time

            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text}")

            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "data": response.json() if response.text else None,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {
                "status_code": 0,
                "response_time": time.time() - start_time,
                "data": {"error": str(e)},
                "success": False
            }

    def test_basic_search(self) -> TestResult:
        """测试 1: 基础搜索"""
        print("\n[测试 1] 基础搜索测试")
        result = self.send_request("宿舍")

        if result["success"] and result["data"].get("success"):
            details = f"找到 {len(result['data'].get('results', []))} 个结果"
            print(f"✓ 通过 - {details}")
            return TestResult("基础搜索", True, result["response_time"], details)
        else:
            details = f"失败: {result['data'].get('error', '未知错误')}"
            print(f"✗ 失败 - {details}")
            return TestResult("基础搜索", False, result["response_time"], details)

    def test_time_range_search(self) -> TestResult:
        """测试 2: 时间范围搜索"""
        print("\n[测试 2] 时间范围搜索测试")
        result = self.send_request("上周拍的宿舍")

        if result["success"] and result["data"].get("success"):
            intent = result["data"].get("intent", {})
            has_time_filter = intent.get("filter_start") and intent.get("filter_end")

            if has_time_filter:
                details = f"时间范围: {intent['filter_start']} 至 {intent['filter_end']}"
                print(f"✓ 通过 - {details}")
                return TestResult("时间范围搜索", True, result["response_time"], details)
            else:
                details = "未提取时间范围"
                print(f"⚠ 警告 - {details}")
                return TestResult("时间范围搜索", True, result["response_time"], details)
        else:
            details = f"失败: {result['data'].get('error', '未知错误')}"
            print(f"✗ 失败 - {details}")
            return TestResult("时间范围搜索", False, result["response_time"], details)

    def test_high_threshold(self) -> TestResult:
        """测试 3: 高阈值搜索"""
        print("\n[测试 3] 高阈值搜索测试")
        result = self.send_request("宿舍", threshold=0.9)

        if result["success"] and result["data"].get("success"):
            result_count = len(result["data"].get("results", []))
            details = f"找到 {result_count} 个结果 (阈值 0.9)"
            print(f"✓ 通过 - {details}")
            return TestResult("高阈值搜索", True, result["response_time"], details)
        else:
            details = f"失败: {result['data'].get('error', '未知错误')}"
            print(f"✗ 失败 - {details}")
            return TestResult("高阈值搜索", False, result["response_time"], details)

    def test_empty_query(self) -> TestResult:
        """测试 4: 空查询 (应失败)"""
        print("\n[测试 4] 空查询测试 (预期失败)")
        result = self.send_request("")

        if not result["success"] or (result["data"].get("success") == False):
            details = "正确拒绝空查询"
            print(f"✓ 通过 - {details}")
            return TestResult("空查询", True, result["response_time"], details)
        else:
            details = "应该拒绝空查询但没有"
            print(f"✗ 失败 - {details}")
            return TestResult("空查询", False, result["response_time"], details)

    def test_long_query(self) -> TestResult:
        """测试 5: 超长查询 (应失败)"""
        print("\n[测试 5] 超长查询测试 (预期失败)")
        long_query = "a" * 501
        result = self.send_request(long_query)

        if not result["success"] or (result["data"].get("success") == False):
            details = "正确拒绝超长查询"
            print(f"✓ 通过 - {details}")
            return TestResult("超长查询", True, result["response_time"], details)
        else:
            details = "应该拒绝超长查询但没有"
            print(f"✗ 失败 - {details}")
            return TestResult("超长查询", False, result["response_time"], details)

    def test_response_format(self) -> TestResult:
        """测试 6: 响应格式验证"""
        print("\n[测试 6] 响应格式验证测试")
        result = self.send_request("测试查询")

        if result["success"] and result["data"]:
            data = result["data"]
            required_fields = ["success", "intent", "threshold", "results"]
            required_intent_fields = ["original_query", "parsed_search_text"]

            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)

            if "intent" in data:
                for field in required_intent_fields:
                    if field not in data["intent"]:
                        missing_fields.append(f"intent.{field}")

            if not missing_fields:
                details = "响应格式正确"
                print(f"✓ 通过 - {details}")
                return TestResult("响应格式", True, result["response_time"], details)
            else:
                details = f"缺少字段: {', '.join(missing_fields)}"
                print(f"✗ 失败 - {details}")
                return TestResult("响应格式", False, result["response_time"], details)
        else:
            details = "无法获取响应"
            print(f"✗ 失败 - {details}")
            return TestResult("响应格式", False, result["response_time"], details)

    def run_all_tests(self) -> None:
        """运行所有测试"""
        print("=" * 60)
        print("BrainDance 搜索 API 测试套件")
        print("=" * 60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API 端点: {self.base_url}")

        # 运行所有测试
        self.results.append(self.test_basic_search())
        self.results.append(self.test_time_range_search())
        self.results.append(self.test_high_threshold())
        self.results.append(self.test_empty_query())
        self.results.append(self.test_long_query())
        self.results.append(self.test_response_format())

        # 生成测试报告
        self.generate_report()

    def generate_report(self) -> None:
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("测试报告")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        avg_response_time = sum(r.response_time for r in self.results) / total_tests

        print(f"\n总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✓")
        print(f"失败: {failed_tests} ✗")
        print(f"通过率: {(passed_tests/total_tests*100):.1f}%")
        print(f"平均响应时间: {avg_response_time:.2f}s")

        print("\n详细结果:")
        print("-" * 60)
        for i, result in enumerate(self.results, 1):
            status = "✓ 通过" if result.passed else "✗ 失败"
            print(f"{i}. {result.test_name:20s} {status:10s} {result.response_time:.2f}s")
            print(f"   详情: {result.details}")

        print("\n" + "=" * 60)

        if failed_tests == 0:
            print("🎉 所有测试通过!")
        else:
            print(f"⚠️  {failed_tests} 个测试失败,请检查")


def main():
    """主函数"""
    import sys

    # 允许从命令行指定 API URL
    api_url = sys.argv[1] if len(sys.argv) > 1 else None

    tester = SearchAPITester(api_url) if api_url else SearchAPITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
