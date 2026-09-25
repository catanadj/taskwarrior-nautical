from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "nautical"
EXPECTED_VERSION = "7.6.0"


class NauticalCliContractTests(unittest.TestCase):
    """Keep global launcher options usable before and after installation."""

    def run_launcher(self, launcher: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(launcher), *arguments],
            cwd=launcher.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_describes_global_options_and_commands(self) -> None:
        result = self.run_launcher(LAUNCHER, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: nautical", result.stdout)
        self.assertIn("--help", result.stdout)
        self.assertIn("--version", result.stdout)
        self.assertIn("install", result.stdout)
        self.assertIn("navigator", result.stdout)

    def test_version_reports_release_version_without_loading_runtime(self) -> None:
        result = self.run_launcher(LAUNCHER, "--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"Nautical {EXPECTED_VERSION}")
        self.assertEqual(result.stderr, "")

    def test_installed_launcher_supports_global_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / "nautical"
            shutil.copy2(LAUNCHER, installed)
            installed.chmod(0o755)

            version = subprocess.run(
                [str(installed), "--version"],
                cwd=installed.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(version.returncode, 0, version.stderr)
            self.assertEqual(version.stdout.strip(), f"Nautical {EXPECTED_VERSION}")

            help_result = subprocess.run(
                [str(installed), "--help"],
                cwd=installed.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--version", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
