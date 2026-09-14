"""Long recurrence intervals retain their requested cadence."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

import nautical_core as core


class LongIntervalContractTests(unittest.TestCase):
    def test_large_weekly_and_monthly_intervals_are_not_clamped(self) -> None:
        weekly = core.validate_anchor_expr_strict("w/1000:mon")
        self.assertEqual(weekly[0][0]["ival"], 1000)
        result, _ = core.next_after_expr(
            weekly, date(2026, 1, 5), default_seed=date(2026, 1, 1)
        )
        self.assertEqual(result, date(2045, 2, 27))

        monthly = core.validate_anchor_expr_strict("m/500:1")
        result, _ = core.next_after_expr(
            monthly, date(2026, 2, 1), default_seed=date(2026, 1, 1)
        )
        self.assertEqual(result, date(2067, 9, 1))

    def test_valid_long_period_intersection_is_accepted_and_bad_yearly_weekday_rejected(self) -> None:
        valid = core.validate_anchor_expr_strict("w/200:mon + y:01-01")
        self.assertEqual(valid[0][0]["ival"], 200)
        with self.assertRaisesRegex(core.ParseError, "weekday selectors belong in 'w:'"):
            core.validate_anchor_expr_strict("y/500:mon + y:01-01")

    def test_yearly_random_constraints_keep_month_and_weekday_boundaries(self) -> None:
        for expression, allowed_months, weekday in (
            ("y:07-rand", {7}, None),
            ("y:rand-07", {7}, None),
            ("y:rand + w:sat", set(range(1, 13)), 5),
        ):
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                current = date(2026, 1, 1)
                selected: list[date] = []
                for _ in range(5):
                    occurrence, _ = core.next_after_expr(
                        dnf,
                        current,
                        default_seed=date(2026, 1, 1),
                        seed_base="yearly-rand-migration",
                    )
                    self.assertIsNotNone(occurrence)
                    self.assertGreater(occurrence, current)
                    self.assertIn(occurrence.month, allowed_months)
                    if weekday is not None:
                        self.assertEqual(occurrence.weekday(), weekday)
                    selected.append(occurrence)
                    current = occurrence + timedelta(days=1)
                self.assertEqual(len(selected), 5)

    def test_yearly_random_respects_sibling_month_filter_and_advances_once_per_year(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf_cached("y:rand + y:apr,jul,oct")
        current = date(2025, 1, 1)
        selected: list[date] = []
        for _ in range(8):
            occurrence, metadata = core.next_after_expr(
                dnf,
                current,
                seed_base="yearly-rand-month-filter",
            )
            self.assertIsNotNone(occurrence)
            self.assertGreater(occurrence, current)
            self.assertEqual((metadata or {}).get("basis"), "rand")
            selected.append(occurrence)
            current = occurrence

        self.assertTrue(all(value.month in {4, 7, 10} for value in selected), selected)
        self.assertEqual(len({value.year for value in selected}), len(selected))

    def test_constrained_yearly_random_is_chain_scoped_and_not_a_forced_cycle(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf_cached("y:rand + y:apr,jul,oct")
        start = date(2025, 12, 31)

        def sequence(seed: str, count: int = 8) -> list[date]:
            current = start
            selected = []
            for _ in range(count):
                current, _metadata = core.next_after_expr(
                    dnf, current, default_seed=start, seed_base=seed
                )
                selected.append(current)
            return selected

        first = sequence("yearly-chain-a")
        self.assertEqual(first, sequence("yearly-chain-a"))
        self.assertNotEqual(first, sequence("yearly-chain-b"))

        three_month_sequences = {
            tuple(day.month for day in sequence(f"yearly-cycle-check-{index}", 3))
            for index in range(24)
        }
        self.assertTrue(
            any(set(months) != {4, 7, 10} for months in three_month_sequences),
            three_month_sequences,
        )

        counts = {4: 0, 7: 0, 10: 0}
        for index in range(600):
            counts[sequence(f"yearly-distribution-{index}", 1)[0].month] += 1
        self.assertTrue(all(140 <= count <= 260 for count in counts.values()), counts)


if __name__ == "__main__":
    unittest.main()
