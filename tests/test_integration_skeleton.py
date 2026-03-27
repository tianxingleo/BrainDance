from __future__ import annotations

import subprocess
import unittest
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "tests" / "scripts"
HTTP_DIR = PROJECT_ROOT / "tests" / "http"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"


def run_command(args: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if dry_run:
        env["BRAINDANCE_IT_DRY_RUN"] = "1"
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


class IntegrationSkeletonTests(unittest.TestCase):
    def test_pubspec_declares_integration_test_dependency(self) -> None:
        pubspec = (APP_DIR / "pubspec.yaml").read_text(encoding="utf-8")
        self.assertIn("integration_test:", pubspec)
        self.assertIn("sdk: flutter", pubspec)

    def test_expected_integration_test_files_exist(self) -> None:
        expected_files = [
            APP_DIR / "integration_test" / "auth_flow_test.dart",
            APP_DIR / "integration_test" / "task_submission_test.dart",
            APP_DIR / "integration_test" / "recall_flow_test.dart",
            APP_DIR / "integration_test" / "realtime_flow_test.dart",
            APP_DIR / "integration_test" / "community_flow_test.dart",
            APP_DIR / "integration_test" / "edge_function_flow_test.dart",
            APP_DIR / "integration_test" / "local_ai_catalog_test.dart",
            APP_DIR / "integration_test" / "support" / "test_bootstrap.dart",
            APP_DIR / "integration_test" / "support" / "test_env.dart",
            APP_DIR / "integration_test" / "support" / "supabase_assertions.dart",
        ]
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in expected_files if not path.exists()]
        self.assertEqual(missing, [])

    def test_expected_fixture_and_http_assets_exist(self) -> None:
        expected_files = [
            PROJECT_ROOT / "app" / ".env.test.example",
            FIXTURES_DIR / "supabase_seed_minimal.sql",
            FIXTURES_DIR / "supabase_seed_realtime.sql",
            FIXTURES_DIR / "supabase_seed_agent.sql",
            FIXTURES_DIR / "cleanup_integration.sql",
            HTTP_DIR / "search_models_smoke.sh",
            HTTP_DIR / "confirm_text_image_smoke.sh",
            HTTP_DIR / "agent_recall_stream_smoke.sh",
            OUTPUT_DIR / ".gitkeep",
        ]
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in expected_files if not path.exists()]
        self.assertEqual(missing, [])

    def test_run_flutter_integration_tests_maps_group_in_dry_run(self) -> None:
        script = SCRIPTS_DIR / "run_flutter_integration_tests.sh"
        result = run_command(
            [str(script), "--group", "auth", "--env", "admin"],
            dry_run=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("group=auth", result.stdout)
        self.assertIn("env=admin", result.stdout)
        self.assertIn("integration_test/auth_flow_test.dart", result.stdout)
        self.assertIn("dry-run enabled", result.stdout)

    def test_run_flutter_integration_tests_rejects_unknown_group(self) -> None:
        script = SCRIPTS_DIR / "run_flutter_integration_tests.sh"
        result = run_command(
            [str(script), "--group", "unknown"],
            dry_run=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported group", result.stderr)

    def test_mutate_processing_task_status_requires_args(self) -> None:
        script = SCRIPTS_DIR / "mutate_processing_task_status.sh"
        result = run_command([str(script)], dry_run=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr)

    def test_run_edge_function_smoke_tests_requires_target(self) -> None:
        script = SCRIPTS_DIR / "run_edge_function_smoke_tests.sh"
        result = run_command([str(script)], dry_run=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr)

    def test_run_edge_function_smoke_tests_dispatches_search_models(self) -> None:
        script = SCRIPTS_DIR / "run_edge_function_smoke_tests.sh"
        result = run_command([str(script), "search-models"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[edge-smoke] target=search-models", result.stdout)
        self.assertIn("[http-search-models] dry-run wrote", result.stdout)

    def test_run_edge_function_smoke_tests_dispatches_confirm_text_image(self) -> None:
        script = SCRIPTS_DIR / "run_edge_function_smoke_tests.sh"
        result = run_command([str(script), "confirm-text-image"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[edge-smoke] target=confirm-text-image", result.stdout)
        self.assertIn("[http-confirm-text-image] dry-run wrote", result.stdout)

    def test_run_edge_function_smoke_tests_dispatches_agent_recall(self) -> None:
        script = SCRIPTS_DIR / "run_edge_function_smoke_tests.sh"
        result = run_command([str(script), "agent-recall"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[edge-smoke] target=agent-recall", result.stdout)
        self.assertIn("[http-agent-recall] dry-run wrote", result.stdout)

    def test_run_edge_function_smoke_tests_rejects_unknown_target(self) -> None:
        script = SCRIPTS_DIR / "run_edge_function_smoke_tests.sh"
        result = run_command([str(script), "unknown-target"], dry_run=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported target", result.stderr)

    def test_seed_script_rejects_unknown_profile(self) -> None:
        script = SCRIPTS_DIR / "seed_supabase_test_data.sh"
        result = run_command([str(script), "--profile", "invalid"], dry_run=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported profile", result.stderr)

    def test_seed_script_reports_fixture_in_dry_run(self) -> None:
        script = SCRIPTS_DIR / "seed_supabase_test_data.sh"
        result = run_command([str(script), "--profile", "agent"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("supabase_seed_agent.sql", result.stdout)
        self.assertIn("dry-run enabled", result.stdout)

    def test_cleanup_script_reports_fixture_in_dry_run(self) -> None:
        script = SCRIPTS_DIR / "cleanup_supabase_test_data.sh"
        result = run_command([str(script)], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("cleanup_integration.sql", result.stdout)
        self.assertIn("dry-run enabled", result.stdout)

    def test_http_search_models_smoke_script_writes_output_file_in_dry_run(self) -> None:
        script = HTTP_DIR / "search_models_smoke.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
          output_file = Path(temp_dir) / "search_models_response.json"
          result = run_command([str(script), str(output_file)], dry_run=True)
          self.assertEqual(result.returncode, 0, msg=result.stderr)
          self.assertTrue(output_file.exists())
          payload = output_file.read_text(encoding="utf-8")
          self.assertIn('"target":"search-models"', payload)

    def test_http_confirm_text_image_smoke_script_writes_output_file_in_dry_run(self) -> None:
        script = HTTP_DIR / "confirm_text_image_smoke.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
          output_file = Path(temp_dir) / "confirm_text_image_response.json"
          result = run_command([str(script), str(output_file)], dry_run=True)
          self.assertEqual(result.returncode, 0, msg=result.stderr)
          self.assertTrue(output_file.exists())
          payload = output_file.read_text(encoding="utf-8")
          self.assertIn('"target":"confirm-text-image"', payload)

    def test_http_agent_recall_stream_smoke_script_writes_output_file_in_dry_run(self) -> None:
        script = HTTP_DIR / "agent_recall_stream_smoke.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
          output_file = Path(temp_dir) / "agent_recall_stream.jsonl"
          result = run_command([str(script), str(output_file)], dry_run=True)
          self.assertEqual(result.returncode, 0, msg=result.stderr)
          self.assertTrue(output_file.exists())
          payload = output_file.read_text(encoding="utf-8")
          self.assertIn('"target":"agent-recall"', payload)

    def test_run_full_integration_suite_dry_run_orchestrates_steps(self) -> None:
        script = SCRIPTS_DIR / "run_full_integration_suite.sh"
        result = run_command([str(script), "--mode", "local"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[suite] dry-run enabled", result.stdout)
        self.assertIn("[bootstrap] dry-run enabled", result.stdout)
        self.assertIn("[seed] profile=minimal", result.stdout)
        self.assertIn("[flutter-it] group=auth", result.stdout)
        self.assertIn("[edge-smoke] target=search-models", result.stdout)
        self.assertTrue((OUTPUT_DIR / "flutter").exists())
        self.assertTrue((OUTPUT_DIR / "edge").exists())
        self.assertTrue((OUTPUT_DIR / "sql").exists())
        self.assertTrue((OUTPUT_DIR / "storage").exists())


if __name__ == "__main__":
    unittest.main()
