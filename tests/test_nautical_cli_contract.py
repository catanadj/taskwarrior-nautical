from __future__ import annotations

import contextlib
import io
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "nautical"
EXPECTED_VERSION = "7.6.1"
EXPECTED_COMMAND_DESCRIPTIONS = {
    "install": "Install hooks, runtime files, and configuration.",
    "runtime-clean": "Remove obsolete managed runtime files safely.",
    "doctor": "Inspect installation, configuration, and runtime health.",
    "queue-status": "Show pending lifecycle and manual-review queue status.",
    "queue-review": "Interactively review queued manual-review items.",
    "review": "Alias for queue-review with the same interactive review flow.",
    "backup": "Create a verified backup of Nautical data.",
    "restore": "Restore Nautical data from a verified backup.",
    "reconcile": "Plan or apply chain and lifecycle reconciliation.",
    "query": "Query tasks through the public Nautical query API.",
    "navigator": "Open the interactive task navigator.",
}


class NauticalCliContractTests(unittest.TestCase):
    """Keep global launcher options usable before and after installation."""

    def test_operator_tools_have_no_obsolete_dev_tool_wrappers(self) -> None:
        for name in ("nautical_doctor.py", "nautical_reconcile.py"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "nautical_core" / "tools" / name).is_file())
                self.assertFalse((ROOT / "dev_tools" / name).exists())

    def run_launcher(self, launcher: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(launcher), *arguments],
            cwd=launcher.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_nautical_dispatches_supported_subcommands(self) -> None:
        entrypoint = runpy.run_path(str(LAUNCHER), run_name="_nautical_dispatch_contract")
        previous_argv = list(sys.argv)
        previous_run_path = entrypoint["runpy"].run_path
        calls: list[tuple[str, str | None, list[str]]] = []
        targets = {
            "install": str(ROOT / "nautical_core" / "tools" / "nautical_install.py"),
            "doctor": str(ROOT / "nautical_core" / "tools" / "nautical_doctor.py"),
            "queue-status": str(ROOT / "nautical_core" / "tools" / "nautical_queue_status.py"),
            "reconcile": str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"),
            "navigator": str(ROOT / "nautical_navigator.py"),
        }

        def fake_run_path(target: str, run_name: str | None = None) -> dict[str, object]:
            calls.append((target, run_name, list(sys.argv)))
            return {}

        try:
            entrypoint["runpy"].run_path = fake_run_path
            for command in targets:
                sys.argv = ["nautical", command, "--json"]
                self.assertEqual(entrypoint["main"](), 0, command)
            sys.argv = ["nautical", "unknown"]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(entrypoint["main"](), 2)
        finally:
            entrypoint["runpy"].run_path = previous_run_path
            sys.argv = previous_argv

        self.assertEqual(len(calls), len(targets))
        for (target, run_name, argv), (command, expected_target) in zip(calls, targets.items()):
            with self.subTest(command=command):
                self.assertEqual(target, expected_target)
                self.assertEqual(run_name, "__main__")
                self.assertEqual(argv[0], expected_target)
                if command == "install":
                    self.assertEqual(argv[1:3], ["--source", str(ROOT)])

        previous_install_target = entrypoint["COMMANDS"]["install"]
        previous_source = os.environ.get("NAUTICAL_SOURCE")
        try:
            entrypoint["COMMANDS"]["install"] = Path("/tmp/nautical-missing-install.py")
            os.environ["NAUTICAL_SOURCE"] = str(ROOT)
            entrypoint["runpy"].run_path = fake_run_path
            calls.clear()
            sys.argv = ["nautical", "install"]
            self.assertEqual(entrypoint["main"](), 0)
            self.assertTrue(calls)
            self.assertEqual(calls[0][0], targets["install"])
        finally:
            entrypoint["COMMANDS"]["install"] = previous_install_target
            if previous_source is None:
                os.environ.pop("NAUTICAL_SOURCE", None)
            else:
                os.environ["NAUTICAL_SOURCE"] = previous_source
            entrypoint["runpy"].run_path = previous_run_path
            sys.argv = previous_argv

    def test_help_describes_global_options_and_commands(self) -> None:
        result = self.run_launcher(LAUNCHER, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: nautical", result.stdout)
        self.assertIn("--help", result.stdout)
        self.assertIn("--version", result.stdout)
        for command, description in EXPECTED_COMMAND_DESCRIPTIONS.items():
            with self.subTest(command=command):
                self.assertIn(f"{command:<15}", result.stdout)
                self.assertIn(description, result.stdout)

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
