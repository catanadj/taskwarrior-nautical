"""Direct unittest coverage migrated from the golden registry."""

from __future__ import annotations

import unittest

import nautical_core as core
from nautical_core import position_selection, season_support


class NaturalLanguageMigrationTests(unittest.TestCase):
    def test_seasonal_selection_uses_configured_boundary_advice(self) -> None:
        descriptions = (
            ("(w:mon)@in-spring=first", "the first Monday of each spring"),
            (
                "(w:mon)@in-spring=first,last",
                "the first and last Mondays of each spring",
            ),
            (
                "(m:1)@in-summer=first",
                "the first matching date from the 1st day of each month during each summer",
            ),
        )
        for expression, expected in descriptions:
            with self.subTest(expression=expression):
                self.assertEqual(core.describe_anchor_expr(expression), expected)

        dnf = core.validate_anchor_expr_strict("(w:mon)@in-spring=first,15th")
        advice = position_selection.selection_advice_for_dnf(dnf)
        self.assertTrue(any("15th can never contribute" in item for item in advice))
        boundary = season_support.season_boundary_description("spring")
        mode = season_support.active_mode()
        self.assertIn(f"@in-spring uses {mode} {boundary} boundaries.", advice)

    def test_common_recurrence_descriptions_retain_user_facing_phrasing(self) -> None:
        cases = (
            ("w:mon", "Mondays"),
            ("w:mon,tue,fri", "either Mondays, Tuesdays, or Fridays"),
            ("w/2:mon", "every 2 weeks: Mondays"),
            ("m:15", "the 15th day of each month"),
            ("m:-1", "the last day of each month"),
            ("m:2nd-mon", "the 2nd Monday of each month"),
            ("m:last-fri", "the last Friday of each month"),
            ("m:5bd", "the 5th business day of each month"),
            ("y:12-25", "Dec 25 each year"),
            ("y:01-01..01-31", "Jan each year"),
            ("m:15@t=09:00", "the 15th day of each month at 09:00"),
            ("w:mon@t=09:00,fri@t=15:00", "either Mondays at 09:00 or Fridays at 15:00"),
            ("moon:full + y:jul", "full-moon phase window begins in Jul each year"),
            ("w:fri@moon=full", "Fridays on full-moon phase dates"),
            ("w:mon,wed,fri + y:apr", "Mondays, Wednesdays, or Fridays in Apr each year"),
            ("y:rand + y:apr", "one random day each year in Apr"),
            ("y:rand", "one random day each year"),
            ("y:rand + y:apr,jul,oct", "one random day each year in Apr, Jul, or Oct"),
            ("y:rand + y:04-20..05-15", "one random day each year in Apr 20–May 15"),
            (
                "m:-1@nbd",
                "the last day of each month if business day; otherwise the next business day",
            ),
            ("y:04-25@+2d", "Apr 25 each year, 2 days later"),
            ("y:04-25@-2d", "Apr 25 each year, 2 days earlier"),
            ("y:04-24@+1bd", "Apr 24 each year, 1 business day later"),
            (
                "m:-1@pbd@-2bd",
                "the last day of each month if business day; otherwise the previous business day, 2 business days earlier",
            ),
        )
        for expression, expected_phrase in cases:
            with self.subTest(expression=expression):
                actual = core.describe_anchor_expr(expression)
                self.assertTrue(actual)
                self.assertIn(expected_phrase.casefold(), actual.casefold())

    def test_complex_recurrence_descriptions_keep_combined_constraints(self) -> None:
        cases = (
            (
                "m/2:31",
                "every 2 months among months that have day 31",
            ),
            (
                "m/2:2nd-mon",
                "every 2 months among months that have the 2nd Monday",
            ),
            (
                "m:-1@nbd@t=09:00",
                "the last day of each month if business day; otherwise the next business day at 09:00",
            ),
            (
                "m:1@nw",
                "the 1st day of each month if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)",
            ),
            (
                "y:04-25@-2d@t=12:00",
                "Apr 25 each year, 2 days earlier at 12:00",
            ),
            (
                "w/3:rand",
                "every 3 weeks: one random day every 3 weeks",
            ),
            (
                "w:mon + m:1 + y:01-01..03-31",
                "Mondays that fall on the 1st day of each month and within Jan–Mar each year",
            ),
        )
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assertEqual(core.describe_anchor_expr(expression), expected)

    def test_yearly_random_or_terms_remain_distinct(self) -> None:
        actual = core.describe_anchor_expr("y:rand | y:apr,jul,oct")
        self.assertTrue(actual.startswith("either one random day each year or "), actual)

    def test_year_ordinals_and_invalid_shorthand_have_direct_guidance(self) -> None:
        valid = {
            "y:d1": "the 1st day of each year",
            "y:d-1": "the last day of each year",
            "y:d100..d110": "days 100–110 of each year",
            "y:d-7..d-1": "the final 7 days of each year",
            "y:w20": "ISO week 20 each ISO year",
            "y:w-1": "the final ISO week of each ISO year",
            "y:w10..w13": "ISO weeks 10–13 each ISO year",
            "y:w-4..w-1": "the final 4 ISO weeks of each ISO year",
            "y:w20 + w:mon": "Mondays in ISO week 20 each ISO year",
            "y/2:w1": "every 2 ISO years: ISO week 1",
        }
        for expression, expected in valid.items():
            with self.subTest(expression=expression):
                self.assertEqual(core.describe_anchor_expr(expression), expected)

        invalid = (
            ("y:d100..110", "Use 'd100..d110'"),
            ("y:w10..13", "Use 'w10..w13'"),
            ("w:w01", "Use 'y:w1' instead of 'w:w01'"),
            ("y:day100", "dN year-day"),
        )
        for expression, guidance in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(guidance, str(raised.exception))

    def test_business_day_rolls_and_counted_random_are_described_and_bounded(self) -> None:
        descriptions = {
            "m:-1@nbd": "the last day of each month if business day; otherwise the next business day",
            "m:-1@pbd": "the last day of each month if business day; otherwise the previous business day",
            "m:15@nw": "the 15th day of each month if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)",
            "w:2rand": "2 random days each week",
            "m:3rand": "3 random days each month",
            "y:2rand": "2 random days each year",
            "y:2rand + y:apr,jul,oct": "2 random days each year in Apr, Jul, or Oct",
        }
        for expression, expected in descriptions.items():
            with self.subTest(expression=expression):
                self.assertEqual(core.describe_anchor_expr(expression), expected)

        invalid_limits = (
            ("w:8rand", "cannot exceed 7"),
            ("w:6rand@bd", "cannot exceed 5"),
            ("m:32rand", "cannot exceed 31"),
            ("y:367rand", "cannot exceed 366"),
        )
        for expression, expected in invalid_limits:
            with self.subTest(expression=expression):
                with self.assertRaises((core.ParseError, core.YearTokenFormatError)) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(expected, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
