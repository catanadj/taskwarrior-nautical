from __future__ import annotations

from datetime import date
import unittest

import nautical_core as core
from nautical_core.parsing.parser_models import ParseError


class SteppedDateRangeTests(unittest.TestCase):
    def test_monthly_range_accepts_day_step_and_expands_from_range_start(self) -> None:
        dnf = core.validate_anchor_expr_strict("m:1..31/3d")

        self.assertEqual(dnf[0][0]["spec"], "1..31/3d")
        self.assertEqual(
            core.expand_monthly_cached("1..31/3d", 2026, 2),
            [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
        )
        self.assertEqual(core.expand_monthly_cached("30..31/3d", 2026, 2), [])

    def test_yearly_range_accepts_day_step_and_crosses_month_boundaries(self) -> None:
        dnf = core.validate_anchor_expr_strict("y:06-01..08-15/2d")

        self.assertEqual(dnf[0][0]["spec"], "06-01..08-15/2d")
        dates = core.expand_yearly_cached("06-01..08-15/2d", 2026)

        self.assertEqual(dates[0], date(2026, 6, 1))
        self.assertEqual(dates[1], date(2026, 6, 3))
        self.assertEqual(dates[-1], date(2026, 8, 14))
        self.assertEqual(len(dates), 38)

    def test_descriptions_explain_the_shared_step(self) -> None:
        self.assertIn("3-day steps", core.describe_anchor_expr("m:1..31/3d"))
        self.assertIn("2-day steps", core.describe_anchor_expr("y:06-01..08-15/2d"))

    def test_yearly_step_skips_february_29_when_the_year_is_not_leap(self) -> None:
        non_leap = core.expand_yearly_cached("02-29..03-03/2d", 2025)
        leap = core.expand_yearly_cached("02-29..03-03/2d", 2024)

        self.assertEqual(non_leap, [date(2025, 3, 1), date(2025, 3, 3)])
        self.assertEqual(leap, [date(2024, 2, 29), date(2024, 3, 2)])

    def test_step_suffix_requires_positive_calendar_days(self) -> None:
        for expression in (
            "m:1..31/0d",
            "y:06-01..08-15/2h",
            "m:15/2d",
            "y:d100..d110/2d",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ParseError):
                    core.validate_anchor_expr_strict(expression)


if __name__ == "__main__":
    unittest.main()
