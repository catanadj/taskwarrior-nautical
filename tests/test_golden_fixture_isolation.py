"""Explicit isolation checks for extracted golden-test collections."""

import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
