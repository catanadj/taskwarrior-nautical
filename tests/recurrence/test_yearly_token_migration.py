"""Direct parser contracts migrated from the golden registry."""

from __future__ import annotations

import unittest
import json
from datetime import date

import nautical_core as core
import nautical_core.anchor_omit as anchor_omit


class YearlyTokenMigrationTests(unittest.TestCase):
    def test_parser_validation_accepts_the_golden_expression_matrix(self) -> None:
        expressions = (
            "w:mon",
            "w:mon,tue",
            "m:1",
            "m:1,15,31",
            "m:1..15",
            "m:2nd-mon",
            "m:last-fri",
            "m:5bd",
            "y:01-01",
            "y:01-01..12-31",
            "y:q1",
            "y:q1..q2",
            "w:mon@t=09:00",
            "m:15@t=09:00@+1d",
            "m:15@t=09:00@-1d",
            "m:-1@pbd@-2bd",
            "y:04-24@+1bd",
            "(w:mon + m:1) | (w:fri + m:15)",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))

    def test_parser_validation_rejects_the_golden_invalid_matrix(self) -> None:
        cases = (
            ("w:mon-fri", "Invalid weekly range"),
            ("m:1:15", "Invalid monthly range '1:15'. Use '..'"),
            ("y:01-01:12-31", "Yearly ranges must use '..'"),
            ("w:invalid", "Unknown weekday"),
            ("m:32", "Day-of-month '32' out of range. Use 1..31 or -1..-31"),
            ("m:6th-mon", "nth-weekday must be between 1 and 5 (or 'last'). Did you mean 'last-mon'? Offending token: '6th-mon'"),
            ("y:13-01", "Yearly token '13-01' doesn’t match ANCHOR_YEAR_FMT=MD. month '13' is invalid. Did you mean MM-DD? e.g., '04-20'"),
            ("y:01-32", "Yearly token '01-32' doesn’t match ANCHOR_YEAR_FMT=MD. day '32' is invalid. Did you mean MM-DD? e.g., '04-20'"),
            ("w:mon + w:sun", "Weekly anchors joined with '+' never coincide (e.g., Saturday AND Monday). Use '|' instead"),
        )
        for expression, expected_message in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(expected_message.casefold(), str(raised.exception).casefold())

    def test_yearly_token_format_preserves_golden_error_surfaces(self) -> None:
        cases = (
            ("y:05:15", "Yearly token '05:15' uses ':' between numbers. Use '-' and order per ANCHOR_YEAR_FMT=MD. Example: '06-01'."),
            ("y:01-01:12-31", "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."),
            ("y:13-01", "Yearly token '13-01' doesn’t match ANCHOR_YEAR_FMT=MD. month '13' is invalid. Did you mean MM-DD? e.g., '04-20'."),
            ("y:01-32", "Yearly token '01-32' doesn’t match ANCHOR_YEAR_FMT=MD. day '32' is invalid. Did you mean MM-DD? e.g., '04-20'."),
            ("y:04-20..03-10", "Yearly token '04-20..03-10' doesn’t match ANCHOR_YEAR_FMT=MD. end precedes start. Did you mean MM-DD? e.g., '04-20'."),
            ("y:rand-13", "Invalid month in yearly token 'rand-13'. Expected 01..12."),
        )
        for expression, expected_prefix in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertTrue(str(raised.exception).startswith(expected_prefix))
        with self.assertRaisesRegex(core.ParseError, "Unknown yearly month alias"):
            core.validate_anchor_expr_strict("y:foo")
        self.assertTrue(core.validate_anchor_expr_strict("y:q1..q2"))

    def test_yearly_token_format_owner_helper_preserves_golden_contract(self) -> None:
        for spec in ("01-01..12-31", "rand-07", "q1..q2"):
            with self.subTest(spec=spec):
                core._validate_yearly_token_format(spec)
        with self.assertRaisesRegex(core.YearTokenFormatError, "month '13' is invalid"):
            core._validate_yearly_token_format("13-01")

    def test_dnf_yearly_validation_preserves_typed_format_error(self) -> None:
        core._validate_year_tokens_in_dnf([[{"typ": "y", "spec": "01-01"}]])
        with self.assertRaises(core.YearTokenFormatError) as ctx:
            core._validate_year_tokens_in_dnf([[{"typ": "y", "spec": "05:15"}]])
        self.assertIn("uses ':' between numbers", str(ctx.exception))

    def test_canonical_quarter_and_date_tokens_are_accepted(self) -> None:
        for token in ("q1", "q2s", "q1..q2", "q1s..q2s", "01-01", "01-01..31-12"):
            with self.subTest(token=token):
                core._validate_yearly_spec_token(token)

    def test_malformed_or_cross_year_ranges_are_rejected_with_guidance(self) -> None:
        cases = (
            ("3..4", "incomplete"),
            ("13", "Invalid month"),
            ("q3..q1", "end quarter precedes start quarter"),
            ("20-04..10-03", "cross-year ranges"),
        )
        for token, expected_message in cases:
            with self.subTest(token=token):
                with self.assertRaises(core.ParseError) as raised:
                    core._validate_yearly_spec_token(token)
                self.assertIn(expected_message.casefold(), str(raised.exception).casefold())

    def test_year_day_ordinals_validate_strictly(self) -> None:
        for expression in ("y:d1", "y:d-1", "y:d1,d100,d-1", "y:d100..d110", "y:d-7..d-1"):
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))
        for token in ("d1", "d-1", "d100..d110", "d-7..d-1"):
            with self.subTest(token=token):
                core._validate_yearly_spec_token(token)

        invalid = (
            ("y:d0", "Year-day '0' out of range"),
            ("y:d367", "Year-day '367' out of range"),
            ("y:d-367", "Year-day '-367' out of range"),
            ("y:d01", "must not be zero-padded"),
            ("y:d100..110", "Use 'd100..d110'"),
            ("y:d-1..d1", "end precedes start"),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(message, str(raised.exception))

        self.assertEqual(core._parse_y_token("d100"), ("year_day", 100))
        self.assertEqual(core._parse_y_token("d-1"), ("year_day", -1))
        self.assertIsNone(core._parse_y_token("d0"))

    def test_year_day_ordinals_expand_and_schedule_across_leap_years(self) -> None:
        self.assertEqual(
            core.expand_yearly_for_year_strict("d1,d60,d-1", 2023),
            [date(2023, 1, 1), date(2023, 3, 1), date(2023, 12, 31)],
        )
        self.assertEqual(
            core.expand_yearly_for_year_strict("d1,d60,d-1", 2024),
            [date(2024, 1, 1), date(2024, 2, 29), date(2024, 12, 31)],
        )
        self.assertEqual(core.expand_yearly_for_year_strict("d366", 2023), [])
        self.assertEqual(core.expand_yearly_for_year_strict("d1..d366", 2023)[-1], date(2023, 12, 31))
        self.assertEqual(
            core.expand_yearly_for_year_strict("d366,d-366", 2024),
            [date(2024, 1, 1), date(2024, 12, 31)],
        )
        self.assertEqual(
            core.expand_yearly_for_year_strict("d100..d102,d-2..d-1", 2024),
            [
                date(2024, 4, 9), date(2024, 4, 10), date(2024, 4, 11),
                date(2024, 12, 30), date(2024, 12, 31),
            ],
        )

        dnf = core.validate_anchor_expr_strict("y:d60")
        next_date, _meta = core.next_after_expr(
            dnf, date(2023, 3, 1), default_seed=date(2023, 1, 1)
        )
        self.assertEqual(next_date, date(2024, 2, 29))
        self.assertTrue(core.factor_matches_on(dnf[0][0], date(2024, 2, 29), date(2024, 1, 1)))
        self.assertFalse(core.factor_matches_on(dnf[0][0], date(2024, 3, 1), date(2024, 1, 1)))

    def test_iso_week_ordinals_validate_strictly(self) -> None:
        for expression in ("y:w1", "y:w-1", "y:w1,w20,w-1", "y:w10..w13", "y:w-4..w-1"):
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))
        for token in ("w1", "w-1", "w10..w13", "w-4..w-1"):
            with self.subTest(token=token):
                core._validate_yearly_spec_token(token)

        invalid = (
            ("y:w0", "ISO week '0' out of range"),
            ("y:w54", "ISO week '54' out of range"),
            ("y:w-54", "ISO week '-54' out of range"),
            ("y:w01", "must not be zero-padded"),
            ("y:w10..13", "Use 'w10..w13'"),
            ("y:w-1..w1", "end precedes start"),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(message, str(raised.exception))

        self.assertEqual(core._parse_y_token("w20"), ("iso_week", 20))
        self.assertEqual(core._parse_y_token("w-1"), ("iso_week", -1))
        self.assertIsNone(core._parse_y_token("w0"))

    def test_iso_week_ordinals_expand_across_gregorian_year_boundaries(self) -> None:
        self.assertEqual(
            core.expand_yearly_for_year_strict("w53", 2020),
            [date(2020, 12, 28), date(2020, 12, 29), date(2020, 12, 30), date(2020, 12, 31)],
        )
        self.assertEqual(
            core.expand_yearly_for_year_strict("w53", 2021),
            [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
        )
        self.assertEqual(core.expand_yearly_for_year_strict("w1", 2018)[-1], date(2018, 12, 31))
        self.assertEqual(
            core.expand_yearly_for_year_strict("w-1", 2021),
            [
                date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3),
                date(2021, 12, 27), date(2021, 12, 28), date(2021, 12, 29),
                date(2021, 12, 30), date(2021, 12, 31),
            ],
        )
        self.assertEqual(len(core.expand_yearly_for_year_strict("w20", 2024)), 7)
        self.assertEqual(len(core.expand_yearly_for_year_strict("w1..w53", 2021)), 365)

        dnf = core.validate_anchor_expr_strict("y:w53")
        next_date, _meta = core.next_after_expr(dnf, date(2020, 12, 31), default_seed=date(2020, 1, 1))
        self.assertEqual(next_date, date(2021, 1, 1))
        self.assertTrue(core.factor_matches_on(dnf[0][0], date(2021, 1, 2), date(2020, 1, 1)))

    def test_year_ordinals_compose_with_weekdays_or_branches_and_modifiers(self) -> None:
        seed = date(2018, 1, 1)
        cases = (
            ("y:w20 + w:mon", date(2024, 1, 1), date(2024, 5, 13)),
            ("y:w1 + w:mon", date(2018, 12, 29), date(2018, 12, 31)),
            ("y:w1 + w:mon", date(2018, 12, 31), date(2019, 12, 30)),
            ("y:w-1 + w:fri", date(2020, 12, 20), date(2021, 1, 1)),
            ("y:w20@+1d + w:tue", date(2024, 5, 1), date(2024, 5, 14)),
        )
        for expression, after_date, expected in cases:
            with self.subTest(expression=expression, after=after_date):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _meta = core.next_after_expr(dnf, after_date, default_seed=seed)
                self.assertEqual(actual, expected)

        union = core.validate_anchor_expr_strict("y:w20 + w:mon | y:d100")
        first, _meta = core.next_after_expr(union, date(2024, 4, 1), default_seed=seed)
        second, _meta = core.next_after_expr(union, first, default_seed=seed)
        self.assertEqual(first, date(2024, 4, 9))
        self.assertEqual(second, date(2024, 5, 13))

    def test_iso_week_interval_uses_iso_year_buckets(self) -> None:
        seed = date(2018, 1, 1)
        dnf = core.validate_anchor_expr_strict("y/2:w1")
        cases = (
            (date(2019, 12, 29), date(2019, 12, 30)),
            (date(2019, 12, 31), date(2020, 1, 1)),
            (date(2020, 1, 5), date(2022, 1, 3)),
        )
        for after_date, expected in cases:
            with self.subTest(after=after_date):
                actual, _meta = core.next_after_expr(dnf, after_date, default_seed=seed)
                self.assertEqual(actual, expected)
        self.assertTrue(core._interval_allowed_for_atom("y", 2, seed, date(2019, 12, 30), "w1"))
        self.assertTrue(core._interval_allowed_for_atom("y", 2, seed, date(2020, 1, 3), "w1"))

    def test_year_ordinals_constrain_random_and_omitted_candidates(self) -> None:
        seed = date(2024, 1, 1)
        exact = core.validate_anchor_expr_strict("y:rand + y:d60")
        exact_next, _meta = core.next_after_expr(exact, seed, default_seed=seed, seed_base="ordinal-rand")
        self.assertEqual(exact_next, date(2024, 2, 29))

        for expression in ("m:rand + y:w20", "m:2rand + y:w20", "y:rand + y:w20"):
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                first, _meta = core.next_after_expr(dnf, seed, default_seed=seed, seed_base="week-rand")
                self.assertIsNotNone(first)
                self.assertEqual(first.isocalendar().week, 20)
                if "2rand" in expression:
                    second, _meta = core.next_after_expr(dnf, first, default_seed=seed, seed_base="week-rand")
                    self.assertNotEqual(second, first)
                    self.assertEqual(second.isocalendar().week, 20)

        anchor_dnf = core.validate_anchor_expr_strict("y:w20 + w:mon")
        omit_dnf = anchor_omit.validate_omit_expr_strict(
            "y:d134", validate_anchor_expr_cached=core.validate_anchor_expr_strict
        )
        next_unomitted, _meta = anchor_omit.next_after_expr_with_omit(
            anchor_dnf, seed, default_seed=seed, seed_base="ordinal-omit",
            omit_dnf=omit_dnf, core=core,
        )
        self.assertEqual(next_unomitted, date(2025, 5, 12))

    def test_year_ordinal_positional_and_json_round_trips_preserve_dates(self) -> None:
        expression = "(y:w20 + w:mon)@in-year=first@+1d | y:d100"
        seed = date(2024, 1, 1)
        dnf = core.validate_anchor_expr_strict(expression)
        first, _meta = core.next_after_expr(dnf, date(2024, 4, 1), default_seed=seed)
        second, _meta = core.next_after_expr(dnf, first, default_seed=seed)
        self.assertEqual(first, date(2024, 4, 9))
        self.assertEqual(second, date(2024, 5, 14))

        canonical = core.acf_to_original_format(core.build_acf(expression))
        reparsed = core.validate_anchor_expr_strict(canonical)
        canonical_dates = []
        cursor = date(2024, 4, 1)
        for _ in range(4):
            cursor, _meta = core.next_after_expr(reparsed, cursor, default_seed=seed)
            canonical_dates.append(cursor)
        original_dates = []
        cursor = date(2024, 4, 1)
        for _ in range(4):
            cursor, _meta = core.next_after_expr(dnf, cursor, default_seed=seed)
            original_dates.append(cursor)
        self.assertEqual(canonical_dates, original_dates)

        json_roundtrip = json.loads(json.dumps(dnf, ensure_ascii=False))
        self.assertTrue(core.validate_anchor_expr_strict(json_roundtrip))

    def test_documented_year_ordinal_examples_parse(self) -> None:
        examples = (
            "y:d100", "y:d-1", "y:w20 + w:mon", "y:w-1 + w:fri",
            "y:d1,d100,d-1", "y:w10..w13", "y/2:w1",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))


if __name__ == "__main__":
    unittest.main()
