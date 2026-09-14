from __future__ import annotations

from datetime import date
import unittest

import nautical_core as core
from nautical_core import anchor_omit


def _next_dates(expression: str, start: date, count: int, *, seed_base: str) -> list[date]:
    dnf = core.validate_anchor_expr_strict(expression)
    cursor = start
    dates = []
    for _ in range(count):
        cursor, _meta = core.next_after_expr(
            dnf,
            cursor,
            default_seed=start,
            seed_base=seed_base,
        )
        if cursor is None:
            raise AssertionError(f"{expression!r} stopped before {count} occurrences")
        dates.append(cursor)
    return dates


class CountedRandomContractTests(unittest.TestCase):
    def test_counted_random_selects_unique_dates_within_each_period(self) -> None:
        weekly = _next_dates("w:2rand", date(2026, 1, 4), 2, seed_base="counted-rand-test")
        self.assertEqual(len(set(weekly)), 2)
        self.assertEqual(len({item.isocalendar()[:2] for item in weekly}), 1)

        monthly = _next_dates("m:3rand", date(2025, 12, 31), 3, seed_base="counted-rand-test")
        self.assertEqual(len(set(monthly)), 3)
        self.assertEqual({(item.year, item.month) for item in monthly}, {(2026, 1)})

        yearly = _next_dates("y:2rand", date(2025, 12, 31), 2, seed_base="counted-rand-test")
        self.assertEqual(len(set(yearly)), 2)
        self.assertEqual({item.year for item in yearly}, {2026})

    def test_counted_random_is_chain_scoped_and_respects_candidate_filters(self) -> None:
        start = date(2025, 12, 31)
        expression = "m:2rand + w:mon,sat"
        first = _next_dates(expression, start, 8, seed_base="chain-a")
        replay = _next_dates(expression, start, 8, seed_base="chain-a")
        other = _next_dates(expression, start, 8, seed_base="chain-b")

        self.assertEqual(first, replay)
        self.assertNotEqual(first, other)
        self.assertTrue(all(item.weekday() in (0, 5) for item in first))
        month_counts: dict[tuple[int, int], int] = {}
        for item in first:
            key = item.year, item.month
            month_counts[key] = month_counts.get(key, 0) + 1
        self.assertTrue(all(count == 2 for count in month_counts.values()))

    def test_omitted_counted_random_candidate_is_redrawn(self) -> None:
        start = date(2025, 12, 31)
        dnf = core.validate_anchor_expr_strict("m:3rand")
        baseline = _next_dates("m:3rand", start, 3, seed_base="omit-redraw")
        omitted = baseline[0]
        omit_state = {"dnf": None, "dates": frozenset({omitted}), "descriptions": {}}

        cursor = start
        result = []
        for _ in range(3):
            cursor, _meta = anchor_omit.next_after_expr_with_omit(
                dnf,
                cursor,
                default_seed=start,
                seed_base="omit-redraw",
                omit_dnf=omit_state,
                core=core,
            )
            self.assertIsNotNone(cursor)
            result.append(cursor)

        self.assertNotIn(omitted, result)
        self.assertEqual(len(set(result)), 3)
        self.assertEqual({(item.year, item.month) for item in result}, {(2026, 1)})

    def test_counted_random_composes_with_interval_time_and_acf(self) -> None:
        expression = "(m/2:2rand + w:mon..fri)@t=09:00"
        dnf = core.validate_anchor_expr_strict(expression)
        dates = _next_dates(expression, date(2025, 12, 31), 4, seed_base="counted-rand-test")
        self.assertTrue(all(item.weekday() < 5 for item in dates))
        self.assertEqual(
            [(item.year, item.month) for item in dates],
            [(2026, 2), (2026, 2), (2026, 4), (2026, 4)],
        )
        self.assertTrue(
            all(atom["mods"]["t"] == (9, 0) for term in dnf for atom in term)
        )
        canonical = core.acf_to_original_format(core.build_acf(expression))
        self.assertTrue(core.dnf_has_counted_random(core.validate_anchor_expr_strict(canonical)))

        yearly = _next_dates("y/2:2rand", date(2025, 12, 31), 4, seed_base="counted-rand-test")
        self.assertEqual([item.year for item in yearly], [2027, 2027, 2029, 2029])


if __name__ == "__main__":
    unittest.main()
