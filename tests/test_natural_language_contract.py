"""Direct behavioral contracts for recurrence natural-language descriptions."""

from __future__ import annotations

import unittest

import nautical_core as core
import nautical_core.natural_language as natural_language
from nautical_core import natural_language_api


class NaturalLanguageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = natural_language_api.for_core(module=core)

    def direct_description(self, expression: str) -> str:
        return natural_language.describe_anchor_expr(
            expression,
            parse_anchor_expr_to_dnf_cached=core.parse_anchor_expr_to_dnf_cached,
            describe_anchor_expr_from_dnf=self.binding._describe_anchor_expr_from_dnf,
        )

    def direct_dnf_description(self, expression: str, mode: str) -> str:
        dnf = core.validate_anchor_expr_strict(expression)
        return natural_language.describe_anchor_dnf(
            dnf,
            {"anchor_mode": mode},
            try_bucket_rand_monthly=self.binding._try_bucket_rand_monthly,
            parse_dt_any=core.parse_dt_any,
            describe_anchor_term=self.binding.describe_anchor_term,
        )

    def test_supported_expressions_have_stable_direct_and_public_text(self) -> None:
        cases = {
            "w:mon": "Mondays",
            "m:1": "the 1st day of each month",
            "w:mon|w:fri": "either Mondays or Fridays",
            "w/2:mon": "every 2 weeks: Mondays",
            "y:12-25": "Dec 25 each year",
            "m/2:31": "every 2 months among months that have day 31",
            "m:-1@prev-fri": "the previous Friday before the last day of each month",
            "y:12-31@next-thu": "Dec 31 each year, then the next Thursday",
            "y:12-31@prev-thu": "Dec 31 each year, then the previous Thursday",
            "w:mon..fri@t=06..17/3h": "Mondays through Fridays every 3h within 06:00–17:00",
            "w:mon@t=09:00,17:30": "Mondays at 09:00, 17:30",
            "w:mon@t=rand(06..18/3)": (
                "Mondays 3 deterministic random times, one per bucket, within 06:00–18:00"
            ),
            "(w/2:2rand@t=18:00 | w/3:thu@t=12:00)": (
                "either 2 random days every 2 weeks at 18:00 or Thursdays every 3 weeks at 12:00"
            ),
            "malformed": "",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                direct = self.direct_description(expression)
                public = core.describe_anchor_expr(expression)
                self.assertEqual(direct, expected)
                self.assertEqual(public, expected)
                self.assertEqual(direct, public)

    def test_mode_tails_are_bound_to_direct_dnf_formatter(self) -> None:
        expression = "w:mon"
        expected = {
            "skip": "Mondays; skip missed anchors",
            "flex": "Mondays; skip past anchors; respect future anchors",
            "all": "Mondays; backfill all missed anchors",
        }
        for mode, text in expected.items():
            with self.subTest(mode=mode):
                direct = self.direct_dnf_description(expression, mode)
                public = core.describe_anchor_dnf(
                    core.validate_anchor_expr_strict(expression), {"anchor_mode": mode}
                )
                self.assertEqual(direct, text)
                self.assertEqual(direct, public)

    def test_shared_tail_compression_is_preserved_in_direct_formatter(self) -> None:
        cases = {
            "(w:mon)+(m:1|m:2|m:3)": (
                "Mondays that fall on either the 1st, the 2nd, or the 3rd day of each month"
            ),
            "(w:mon..wed)+(m:1..10)+(y:01-01|y:10-01)": (
                "Mondays through Wednesdays that fall on days 1–10 of each month and within "
                "either Jan 1 or Oct 1 each year"
            ),
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.direct_description(expression), expected)
                self.assertEqual(core.describe_anchor_expr(expression), expected)
                dnf = core.validate_anchor_expr_strict(expression)
                self.assertEqual(
                    natural_language.describe_anchor_dnf(
                        dnf,
                        {"anchor_mode": "skip"},
                        try_bucket_rand_monthly=self.binding._try_bucket_rand_monthly,
                        parse_dt_any=core.parse_dt_any,
                        describe_anchor_term=self.binding.describe_anchor_term,
                    ),
                    expected + "; skip missed anchors",
                )

    def test_direct_formatter_is_order_stable(self) -> None:
        expressions = ["w:mon", "m:1", "y:12-25", "w/2:mon", "malformed"]
        normal = [self.direct_description(expression) for expression in expressions]
        shuffled = [self.direct_description(expression) for expression in reversed(expressions)]
        self.assertEqual(shuffled, list(reversed(normal)))


if __name__ == "__main__":
    unittest.main()
