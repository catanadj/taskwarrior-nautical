"""Explicit isolation checks for extracted golden-test collections."""

import os
import subprocess
import sys
import tempfile
import unittest


class GoldenFixtureIsolationTests(unittest.TestCase):
    def test_domain_collections_are_immutable_tuples(self):
        from dev_tools.golden_tests import hooks, lifecycle, reconcile

        for module in (hooks, lifecycle, reconcile):
            self.assertIsInstance(module.TESTS, tuple)
            self.assertTrue(all(callable(test) for test in module.TESTS))

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
