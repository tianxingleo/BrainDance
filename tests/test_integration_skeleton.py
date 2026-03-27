from __future__ import annotations

import subprocess
import unittest
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "tests" / "scripts"


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

    def test_run_full_integration_suite_dry_run_orchestrates_steps(self) -> None:
        script = SCRIPTS_DIR / "run_full_integration_suite.sh"
        result = run_command([str(script), "--mode", "local"], dry_run=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[suite] dry-run enabled", result.stdout)
        self.assertIn("[bootstrap] dry-run enabled", result.stdout)
        self.assertIn("[seed] profile=minimal", result.stdout)
        self.assertIn("[flutter-it] group=auth", result.stdout)
        self.assertIn("[edge-smoke] target=search-models", result.stdout)


if __name__ == "__main__":
    unittest.main()
