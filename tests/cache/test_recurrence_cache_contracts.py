from __future__ import annotations

import unittest

import nautical_core as core


class RecurrenceCacheContractTests(unittest.TestCase):
    def test_cached_parse_returns_independent_dnf_instances(self) -> None:
        expression = "w:mon@t=09:00 + m:1"

        first = core.parse_anchor_expr_to_dnf_cached(expression)
        second = core.parse_anchor_expr_to_dnf_cached(expression)

        self.assertIsNot(first, second)
        first[0][0]["spec"] = "tue"
        self.assertEqual(second[0][0]["spec"], "mon")
        self.assertEqual(
            core.parse_anchor_expr_to_dnf_cached(expression)[0][0]["spec"],
            "mon",
        )


if __name__ == "__main__":
    unittest.main()
