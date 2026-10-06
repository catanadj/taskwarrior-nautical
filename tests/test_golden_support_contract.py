from __future__ import annotations

import inspect
import unittest

from dev_tools.golden_tests import support


class GoldenSupportContractTests(unittest.TestCase):
    def test_parse_due_only_swallows_expected_parse_errors(self) -> None:
        source = inspect.getsource(support.parse_due)
        self.assertNotIn("except Exception", source)
        self.assertIsNone(support.parse_due("not-a-date"))
        self.assertEqual(support.parse_due("2026-10-06").year, 2026)
