from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from dev_tools.golden_tests import support


class GoldenSupportContractTests(unittest.TestCase):
    def test_parse_due_only_swallows_expected_parse_errors(self) -> None:
        source = inspect.getsource(support.parse_due)
        self.assertNotIn("except Exception", source)
        self.assertIsNone(support.parse_due("not-a-date"))
        self.assertEqual(support.parse_due("2026-10-06").year, 2026)

    def test_preview_and_natural_helpers_do_not_hide_production_failures(self) -> None:
        import nautical_core

        with patch.object(
            nautical_core,
            "build_and_cache_hints",
            side_effect=RuntimeError("preview invariant failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "preview invariant failed"):
                support.build_preview("w:mon")

        with patch.object(
            nautical_core,
            "describe_anchor_expr",
            side_effect=RuntimeError("natural invariant failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "natural invariant failed"):
                support.must_natural("w:mon")
