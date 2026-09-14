"""Direct contracts for positional recurrence selection."""

from datetime import date, timedelta
import unittest

import nautical_core as core
from nautical_core import position_selection


class PositionSelectionContracts(unittest.TestCase):
    def test_public_monthly_selector_validation(self) -> None:
        node = core.validate_anchor_expr_strict(
            "(w:tue | w:thu)@in-month=first,3rd,last"
        )[0][0]
        self.assertEqual(node.get("kind"), "select")
        self.assertEqual(node.get("scope"), "month")
        self.assertEqual(node.get("positions"), (1, 3, -1))

        invalid = (
            ("w:tue@in-month=last", "parenthesized candidate group"),
            ("(w:tue | w:thu)@in-decade=last", "Invalid positional selector"),
            ("(w:rand | w:thu)@in-month=last", "cannot contain random selectors"),
            ("(w:tue@bd | w:thu)@in-month=last", "cannot contain modifiers"),
            ("(w:tue | w:thu)@in-month=last@bd", "candidate filter"),
            ("(w:tue | w:thu)@t=09:00@in-month=last", "must appear before"),
            ("((w:tue | w:thu)@in-month=last)@in-month=last", "Nested positional"),
            (
                "(w:tue)@in-month=last + (w:thu)@in-month=last",
                "only one positional selection",
            ),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(message, str(raised.exception))

    def test_public_monthly_scheduler_selects_and_skips_empty_buckets(self) -> None:
        seed = date(2026, 1, 1)
        dnf = core.validate_anchor_expr_strict(
            "(w:tue | w:thu)@in-month=first,last"
        )
        cases = (
            (date(2026, 7, 1), date(2026, 7, 2)),
            (date(2026, 7, 2), date(2026, 7, 30)),
            (date(2026, 7, 30), date(2026, 8, 4)),
        )
        for after_date, expected in cases:
            with self.subTest(after_date=after_date):
                actual, _metadata = core.next_after_expr(
                    dnf, after_date, default_seed=seed, seed_base="chain-a"
                )
                self.assertEqual(actual, expected)

        constrained = core.validate_anchor_expr_strict(
            "(m:20..-1 + w:tue,thu)@in-month=first"
        )
        actual, _metadata = core.next_after_expr(
            constrained, date(2026, 7, 1), default_seed=seed, seed_base="chain-a"
        )
        self.assertEqual(actual, date(2026, 7, 21))

        sparse = core.validate_anchor_expr_strict("(w:mon)@in-month=5th")
        actual, _metadata = core.next_after_expr(
            sparse, date(2026, 7, 1), default_seed=seed, seed_base="chain-a"
        )
        self.assertEqual(actual, date(2026, 8, 31))

    def test_post_selection_modifiers_parse_and_schedule_without_changing_bucket(self) -> None:
        expression = (
            "(w:tue | w:thu)@in-month=last@next-mon@+1d@-2bd@t=09:00,17:30"
        )
        dnf = core.validate_anchor_expr_strict(expression)
        modifiers = dnf[0][0].get("mods") or {}
        self.assertEqual((modifiers.get("roll"), modifiers.get("wd")), ("next-wd", 0))
        self.assertEqual(modifiers.get("day_offset"), 1)
        self.assertEqual(modifiers.get("business_day_offset"), -2)
        self.assertEqual(modifiers.get("t"), [(9, 0), (17, 30)])

        seed = date(2026, 1, 1)
        cases = (
            ("(w:tue | w:thu)@in-month=last@+2d", date(2026, 7, 1), date(2026, 7, 2)),
            ("(w:tue | w:thu)@in-month=last@+2d", date(2026, 7, 2), date(2026, 8, 1)),
            ("(w:tue | w:thu)@in-month=last@-2d", date(2026, 7, 1), date(2026, 7, 28)),
            ("(m:-1)@in-month=last@pbd", date(2026, 1, 1), date(2026, 1, 30)),
            ("(m:-1)@in-month=last@nbd", date(2026, 1, 1), date(2026, 2, 2)),
            ("(w:thu)@in-month=last@next-mon", date(2026, 7, 1), date(2026, 8, 3)),
            ("(w:thu)@in-month=last@+1bd", date(2026, 7, 1), date(2026, 7, 31)),
        )
        for anchor, after_date, expected in cases:
            with self.subTest(anchor=anchor, after_date=after_date):
                parsed = core.validate_anchor_expr_strict(anchor)
                actual, _metadata = core.next_after_expr(
                    parsed, after_date, default_seed=seed
                )
                self.assertEqual(actual, expected)
                self.assertTrue(core.factor_matches_on(parsed[0][0], expected, seed))

        collision = core.validate_anchor_expr_strict(
            "(m:-2,-1)@in-month=first,last@pbd"
        )
        first, _metadata = core.next_after_expr(
            collision, date(2026, 1, 1), default_seed=seed
        )
        second, _metadata = core.next_after_expr(
            collision, first, default_seed=seed
        )
        self.assertEqual((first, second), (date(2026, 1, 30), date(2026, 2, 27)))

        class HolidayCalendar:
            name = "selector-holidays"
            fingerprint = "selector-holidays-v1"

            def is_business_day(self, value):
                return value.weekday() < 5 and value != date(2026, 7, 31)

        custom = core.validate_anchor_expr_strict("(w:thu)@in-month=last@+1bd")
        custom_next, _metadata = core.next_after_expr(
            custom,
            date(2026, 7, 1),
            default_seed=seed,
            business_calendar=HolidayCalendar(),
        )
        self.assertEqual(custom_next, date(2026, 8, 3))

    def test_public_period_scope_validation(self) -> None:
        valid = (
            ("(w:mon | w:wed | w:fri)@in-week=first,last", "week", (1, -1)),
            ("(w:mon)@in-quarter=10th,2nd-last", "quarter", (10, -2)),
            ("(w:mon)@in-year=100th,last", "year", (100, -1)),
        )
        for expression, scope, positions in valid:
            with self.subTest(expression=expression):
                node = core.validate_anchor_expr_strict(expression)[0][0]
                self.assertEqual(node.get("scope"), scope)
                self.assertEqual(node.get("positions"), positions)

        invalid = (
            ("w:mon@in-year=last", "parenthesized candidate group"),
            ("(w:mon)@in-week=8th", "week limit of 7"),
            ("(w:mon)@in-quarter=93rd", "quarter limit of 92"),
            ("(w:mon)@in-year=367th", "year limit of 366"),
            ("(w:mon)@in-year=last@bd", "candidate filter"),
            ("(w:rand)@in-quarter=last", "cannot contain random selectors"),
            ("((w:mon)@in-week=last)@in-year=last", "Nested positional"),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(message, str(raised.exception))

    def test_public_period_scope_scheduler_and_shift_preserve_source_bucket(self) -> None:
        seed = date(2026, 1, 1)
        cases = (
            ("(w:mon | w:wed | w:fri)@in-week=last", date(2026, 7, 14), date(2026, 7, 17)),
            ("(w:mon)@in-quarter=last", date(2026, 4, 1), date(2026, 6, 29)),
            ("(w:mon)@in-year=10th", date(2026, 1, 1), date(2026, 3, 9)),
            ("(y:02-29)@in-year=first", date(2026, 1, 1), date(2028, 2, 29)),
            ("(w:fri)@in-week=last@+3d", date(2026, 7, 12), date(2026, 7, 13)),
            ("(w:mon)@in-quarter=last@+3d", date(2026, 6, 30), date(2026, 7, 2)),
            ("(w:mon)@in-year=last@+7d", date(2027, 1, 1), date(2027, 1, 4)),
            ("(w:mon)@in-year=first@-7d", date(2025, 12, 1), date(2025, 12, 29)),
        )
        for expression, after_date, expected in cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _metadata = core.next_after_expr(
                    dnf, after_date, default_seed=seed
                )
                self.assertEqual(actual, expected)
                self.assertTrue(core.factor_matches_on(dnf[0][0], expected, seed))

        class QuarterHolidayCalendar:
            name = "quarter-holidays"
            fingerprint = "quarter-holidays-v1"

            def is_business_day(self, value):
                return value.weekday() < 5 and value != date(2026, 6, 30)

        custom = core.validate_anchor_expr_strict("(w:mon)@in-quarter=last@+1bd")
        actual, _metadata = core.next_after_expr(
            custom,
            date(2026, 4, 1),
            default_seed=seed,
            business_calendar=QuarterHolidayCalendar(),
        )
        self.assertEqual(actual, date(2026, 7, 1))

    def test_position_selection_documented_examples_and_actionable_errors(self) -> None:
        examples = (
            "(w:mon | w:wed | w:fri)@in-week=last",
            "(w:tue | w:thu)@in-month=first,last",
            "(w:mon)@in-quarter=last@+1bd",
            "(w:mon)@in-year=10th@t=09:00",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                core.validate_anchor_expr_strict(expression)

        invalid = (
            ("w:mon@in-year=last", "parenthesized candidate group"),
            (
                "(w:mon@t=09:00)@in-year=last",
                "place supported modifiers after the selector",
            ),
            (
                "(w:mon)@in-decade=last",
                "@in-week, @in-month, @in-quarter, or @in-year",
            ),
        )
        for expression, guidance in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(guidance, str(raised.exception))

    def test_positions_normalize_aliases_and_preserve_first_seen_order(self) -> None:
        self.assertEqual(
            position_selection.parse_positions(
                "first, 2nd, 11th, 21st, last, 2nd-last", "month"
            ),
            (1, 2, 11, 21, -1, -2),
        )
        self.assertEqual(
            position_selection.parse_positions(
                "first,1st,1,last,1st-last", "month"
            ),
            (1, -1),
        )
        self.assertEqual(position_selection.parse_positions("92nd", "quarter"), (92,))
        self.assertEqual(
            position_selection.parse_positions("100th,366th", "year"), (100, 366)
        )

    def test_positions_reject_malformed_aliases_and_scope_overflow(self) -> None:
        invalid = (
            ("", "month", "cannot be empty"),
            ("first,,last", "month", "empty item"),
            ("0", "month", "zero is invalid"),
            ("11st", "month", "Use '11th'"),
            ("22th", "month", "Use '22nd'"),
            ("22th-last", "month", "Use '22nd-last'"),
            ("second", "month", "Invalid position"),
            ("8th", "week", "week limit of 7"),
            ("32nd", "month", "month limit of 31"),
            ("93rd", "quarter", "quarter limit of 92"),
            ("367th", "year", "year limit of 366"),
        )
        for value, scope, message in invalid:
            with self.subTest(value=value, scope=scope):
                with self.assertRaises(ValueError) as raised:
                    position_selection.parse_positions(value, scope)
                self.assertIn(message, str(raised.exception))

        with self.assertRaisesRegex(ValueError, "Unknown selection scope"):
            position_selection.parse_positions("first", "decade")

    def test_candidate_capacity_bounds_cover_common_deterministic_selectors(self) -> None:
        cases = (
            ("(w:mon)@in-week=first", 1),
            ("(w:mon)@in-month=5th", 5),
            ("(w:mon | w:fri)@in-month=10th", 10),
            ("(m:15)@in-year=12th", 12),
            ("(y:d1..d10)@in-year=10th", 10),
            ("(y:w-1)@in-year=14th", 14),
            ("(y:w-1 + w:fri)@in-year=2nd", 2),
        )
        for expression, expected in cases:
            with self.subTest(expression=expression):
                node = core.validate_anchor_expr_strict(expression)[0][0]
                self.assertEqual(
                    position_selection.candidate_capacity_upper_bound(node), expected
                )

    def test_structural_capacity_bounds_never_underestimate_candidates(self) -> None:
        candidates = (
            "w:fri..mon", "w:mon | w:fri", "m:1,-1", "m:1..7", "m:last-fri",
            "y:d1..d10", "y:w-1", "y:w52..w53", "y:01-01..01-31", "y:apr",
            "y:q1", "y:q1..q2", "y:w-1 + w:fri",
        )
        probes = {
            "week": [date(2023, 12, 25) + timedelta(days=7 * offset) for offset in range(106)],
            "month": [date(year, month, 1) for year in range(2024, 2027) for month in range(1, 13)],
            "quarter": [date(year, month, 1) for year in range(2024, 2027) for month in (1, 4, 7, 10)],
            "year": [date(year, 1, 1) for year in range(2024, 2029)],
        }
        seed = date(2024, 1, 1)
        for expression in candidates:
            for scope, scope_probes in probes.items():
                node = core.validate_anchor_expr_strict(
                    f"({expression})@in-{scope}=first"
                )[0][0]
                upper = position_selection.candidate_capacity_upper_bound(node)
                for probe in scope_probes:
                    start, end = position_selection.period_bounds(scope, probe)
                    current = start
                    count = 0
                    while current <= end:
                        if any(
                            all(
                                core.atom_matches_on(
                                    atom,
                                    current,
                                    seed,
                                    seed_base="capacity-soundness",
                                )
                                for atom in term
                            )
                            for term in node["expr"]
                        ):
                            count += 1
                        current += timedelta(days=1)
                    self.assertLessEqual(
                        count,
                        upper,
                        f"{expression} in {scope} at {start}: {count} exceeds {upper}",
                    )

    def test_validation_rejects_only_fully_impossible_positions(self) -> None:
        impossible = (
            ("(w:mon)@in-week=2nd", "at most 1 matching date per week"),
            ("(w:mon)@in-month=6th", "at most 5 matching dates per month"),
            ("(m:15)@in-month=2nd", "at most 1 matching date per month"),
            ("(y:d100)@in-year=2nd", "at most 1 matching date per year"),
            ("(y:w-1 + w:fri)@in-year=3rd", "at most 2 matching dates per year"),
        )
        for expression, expected in impossible:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(expected, str(raised.exception))

        mixed = core.validate_anchor_expr_strict("(w:mon)@in-month=first,6th")
        self.assertEqual(mixed[0][0].get("positions"), (1, 6))

    def test_semantic_advice_explains_dead_redundant_and_cross_calendar_positions(self) -> None:
        mixed = position_selection.selection_advice_for_dnf(
            core.validate_anchor_expr_strict("(w:mon)@in-month=first,6th")
        )
        self.assertTrue(any("6th can never contribute" in item for item in mixed))

        redundant = position_selection.selection_advice_for_dnf(
            core.validate_anchor_expr_strict("(m:15)@in-month=first")
        )
        self.assertTrue(any("is redundant here" in item for item in redundant))

        boundary_expression = "(y:w-1)@in-year=8th"
        boundary = position_selection.selection_advice_for_dnf(
            core.validate_anchor_expr_strict(boundary_expression)
        )
        self.assertTrue(
            any("ISO-week candidates by calendar year" in item for item in boundary)
        )
        scheduled, _metadata = core.next_after_expr(
            core.validate_anchor_expr_strict(boundary_expression),
            date(2026, 1, 1),
            default_seed=date(2026, 1, 1),
            seed_base="semantic-advice",
        )
        self.assertEqual(scheduled, date(2027, 12, 31))

    def test_period_bounds_use_exact_calendar_and_iso_week_boundaries(self) -> None:
        cases = (
            ("week", date(2024, 12, 31), date(2024, 12, 30), date(2025, 1, 5)),
            ("month", date(2024, 2, 15), date(2024, 2, 1), date(2024, 2, 29)),
            ("month", date(2023, 2, 15), date(2023, 2, 1), date(2023, 2, 28)),
            ("quarter", date(2024, 5, 20), date(2024, 4, 1), date(2024, 6, 30)),
            ("quarter", date(2024, 12, 31), date(2024, 10, 1), date(2024, 12, 31)),
            ("year", date(2024, 2, 29), date(2024, 1, 1), date(2024, 12, 31)),
        )
        for scope, value, start, end in cases:
            with self.subTest(scope=scope, value=value):
                self.assertEqual(
                    position_selection.period_bounds(scope, value), (start, end)
                )
        with self.assertRaisesRegex(TypeError, "must be a date"):
            position_selection.period_bounds("month", "2024-02-01")

    def test_internal_evaluator_selects_signed_positions_and_deduplicates(self) -> None:
        inner = core.validate_anchor_expr_strict("w:tue | w:thu")
        node = {
            "kind": "select", "scope": "month", "positions": [1, -1, 1],
            "expr": inner, "mods": {},
        }
        normalized = position_selection.normalize_selection_node(node)
        self.assertEqual(normalized["positions"], (1, -1))
        selected = position_selection.selected_candidates_in_period(
            node,
            date(2026, 7, 15),
            matches_on=core.atom_matches_on,
            default_seed=date(2026, 1, 1),
            seed_base="chain-a",
        )
        self.assertEqual(selected, (date(2026, 7, 2), date(2026, 7, 30)))
        self.assertEqual(
            position_selection.select_positions(
                [date(2026, 7, 2), date(2026, 7, 2), date(2026, 7, 30)],
                (1, -1, 4),
            ),
            (date(2026, 7, 2), date(2026, 7, 30)),
        )

    def test_internal_evaluator_rejects_malformed_nodes(self) -> None:
        atom = core.validate_anchor_expr_strict("w:mon")[0][0]
        invalid_nodes = (
            {},
            {"kind": "select", "scope": "month", "positions": [], "expr": [[atom]]},
            {"kind": "select", "scope": "month", "positions": [0], "expr": [[atom]]},
            {"kind": "select", "scope": "week", "positions": [8], "expr": [[atom]]},
            {"kind": "select", "scope": "month", "positions": [1], "expr": []},
            {"kind": "select", "scope": "month", "positions": [1], "expr": [[]]},
            {"kind": "select", "scope": "month", "positions": [1], "expr": [[{"kind": "select"}]]},
        )
        for node in invalid_nodes:
            with self.subTest(node=node), self.assertRaises(ValueError):
                position_selection.normalize_selection_node(node)
        with self.assertRaisesRegex(ValueError, "non-zero integers"):
            position_selection.select_positions([date(2026, 1, 1)], (0,))

    def test_next_selected_date_jumps_empty_periods_with_a_bound(self) -> None:
        node = {
            "kind": "select", "scope": "month", "positions": [-1],
            "expr": core.validate_anchor_expr_strict("w:tue | w:thu"), "mods": {},
        }
        self.assertEqual(
            position_selection.next_selected_date(
                node,
                date(2026, 7, 30),
                matches_on=core.atom_matches_on,
                default_seed=date(2026, 1, 1),
                seed_base="chain-a",
            ),
            date(2026, 8, 27),
        )

        calls = {"count": 0}

        def never_matches(_atom, _value, _seed, seed_base=None):
            calls["count"] += 1
            return False

        self.assertIsNone(
            position_selection.next_selected_date(
                node,
                date(2026, 7, 1),
                matches_on=never_matches,
                default_seed=date(2026, 1, 1),
                max_periods=3,
            )
        )
        self.assertEqual(calls["count"], 184)

    def test_candidate_cache_identity_includes_expression_seed_and_calendar(self) -> None:
        position_selection.clear_candidate_cache()
        calls = {"count": 0}

        def counting_match(atom, value, seed, seed_base=None):
            calls["count"] += 1
            return core.atom_matches_on(atom, value, seed, seed_base=seed_base)

        node = {
            "kind": "select", "scope": "month", "positions": [-1],
            "expr": core.validate_anchor_expr_strict("w:tue | w:thu"), "mods": {},
        }
        kwargs = {
            "matches_on": counting_match,
            "default_seed": date(2026, 1, 1),
            "seed_base": "chain-a",
            "calendar_fingerprint": "calendar-a",
        }
        try:
            first = position_selection.selected_candidates_in_period(
                node, date(2026, 7, 1), **kwargs
            )
            first_call_count = calls["count"]
            second = position_selection.selected_candidates_in_period(
                node, date(2026, 7, 20), **kwargs
            )
            self.assertEqual(first, second)
            self.assertEqual(first, (date(2026, 7, 30),))
            self.assertEqual(calls["count"], first_call_count)

            reversed_node = dict(node, expr=list(reversed(node["expr"])))
            position_selection.selected_candidates_in_period(
                reversed_node, date(2026, 7, 1), **kwargs
            )
            self.assertEqual(calls["count"], first_call_count)

            position_selection.selected_candidates_in_period(
                node,
                date(2026, 7, 1),
                **dict(kwargs, calendar_fingerprint="calendar-b"),
            )
            self.assertEqual(calls["count"], first_call_count * 2)
            position_selection.selected_candidates_in_period(
                node, date(2026, 7, 1), **dict(kwargs, seed_base="chain-b")
            )
            self.assertEqual(calls["count"], first_call_count * 3)
            info = position_selection.candidate_cache_info()
            self.assertGreaterEqual(info.hits, 2)
            self.assertEqual(info.misses, 3)
        finally:
            position_selection.clear_candidate_cache()


if __name__ == "__main__":
    unittest.main()
