"""Command-line and exit-status contracts for the golden test runner."""

import contextlib
import importlib
import io
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class GoldenRunnerContractTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "dev_tools/nautical_golden_tests.py", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def test_repeated_filters_verbose_and_shuffle_are_deterministic(self):
        args = (
            "--only",
            "test_random_salt_namespaces_draws",
            "--only",
            "test_modifier_boundary_paths_agree_and_advance_strictly",
            "--verbose",
            "--shuffle-seed",
            "20260925",
        )

        first = self._run(*args)
        second = self._run(*args)
        registered_order = self._run(
            "--only",
            "test_random_salt_namespaces_draws",
            "--only",
            "test_modifier_boundary_paths_agree_and_advance_strictly",
            "--verbose",
        )

        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(
            registered_order.returncode,
            0,
            registered_order.stdout + registered_order.stderr,
        )
        first_names = re.findall(r"^✓ (test_[^:]+):", first.stdout, re.MULTILINE)
        second_names = re.findall(r"^✓ (test_[^:]+):", second.stdout, re.MULTILINE)
        ordered_names = re.findall(
            r"^✓ (test_[^:]+):", registered_order.stdout, re.MULTILINE
        )
        expected = {
            "test_random_salt_namespaces_draws",
            "test_modifier_boundary_paths_agree_and_advance_strictly",
        }
        self.assertEqual(set(first_names), expected)
        self.assertEqual(first_names, second_names)
        self.assertEqual(
            ordered_names,
            ["test_modifier_boundary_paths_agree_and_advance_strictly", "test_random_salt_namespaces_draws"],
        )
        self.assertIn("Total tests run: 2", first.stdout)
        self.assertIn("Passed: 2", first.stdout)
        self.assertIn("Failed: 0", first.stdout)

    def test_failed_registered_case_returns_nonzero_and_summary(self):
        runner = importlib.import_module("dev_tools.nautical_golden_tests")

        def failing_case():
            raise AssertionError("intentional runner contract failure")

        output = io.StringIO()
        with (
            patch.object(runner, "TESTS", [failing_case]),
            patch.object(sys, "argv", ["nautical_golden_tests.py", "--verbose"]),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as raised:
                runner.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("intentional runner contract failure", output.getvalue())
        self.assertIn("Total tests run: 1", output.getvalue())
        self.assertIn("Failed: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
