"""Explicit isolation checks for extracted golden-test collections."""

import os
import subprocess
import sys
import tempfile
import unittest


class GoldenFixtureIsolationTests(unittest.TestCase):
    def test_term_fixture_restores_existing_and_missing_values(self):
        from dev_tools.golden_tests.support import test_term

        original = os.environ.get("TERM")
        try:
            os.environ["TERM"] = "before-test"
            with test_term("xterm"):
                self.assertEqual(os.environ.get("TERM"), "xterm")
            self.assertEqual(os.environ.get("TERM"), "before-test")

            os.environ.pop("TERM", None)
            with test_term("xterm"):
                self.assertEqual(os.environ.get("TERM"), "xterm")
            self.assertNotIn("TERM", os.environ)
        finally:
            if original is None:
                os.environ.pop("TERM", None)
            else:
                os.environ["TERM"] = original

    def test_domain_collections_are_immutable_tuples(self):
        from dev_tools.golden_tests import (
            installer,
            configuration,
            lifecycle,
            modify,
            navigator,
            operator,
            reconcile,
            scheduling,
            timeline,
        )

        for module in (configuration, installer, lifecycle, modify, navigator, operator, reconcile, scheduling, timeline):
            self.assertIsInstance(module.TESTS, tuple)
            self.assertTrue(all(callable(test) for test in module.TESTS))

    def test_timeline_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.timeline'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.timeline', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_operator_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.operator'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.operator', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_navigator_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.navigator'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.navigator', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_installer_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.installer'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.installer', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_configuration_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.configuration'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.configuration', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scheduling_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.scheduling'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.scheduling', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_modify_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.modify'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.modify', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_storage_domain_import_does_not_load_sibling_domains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module('dev_tools.golden_tests.storage'); "
                "loaded = sorted(name for name in sys.modules "
                "if name.startswith('dev_tools.golden_tests.') "
                "and name not in {'dev_tools.golden_tests.storage', "
                "'dev_tools.golden_tests.support'}); "
                "print(loaded); raise SystemExit(bool(loaded))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_representative_domains_pass_in_fresh_processes(self):
        commands = (
            "reconcile_tool_computes_year_ordinal_anchor",
            "lifecycle_application_happy_path_real_stack",
        )
        for selector in commands:
            result = subprocess.run(
                [
                    sys.executable,
                    "dev_tools/nautical_golden_tests.py",
                    "--only",
                    selector,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Failed: 0", result.stdout)

    def test_navigator_core_override_does_not_poison_following_hook_test(self):
        with tempfile.TemporaryDirectory(prefix="nautical-golden-taskdata-") as taskdata:
            env = dict(os.environ)
            env.pop("TASKRC", None)
            env["TASKDATA"] = taskdata
            env["NAUTICAL_CORE_PATH"] = os.getcwd()
            result = subprocess.run(
                [
                    sys.executable,
                    "dev_tools/nautical_golden_tests.py",
                    "--shuffle-seed",
                    "1",
                    "--only",
                    "navigator_uses_anchor_and_anchor_file_sources",
                    "--only",
                    "reconcile_expiration_cp_advances_from_recurrence_target",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Passed: 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
