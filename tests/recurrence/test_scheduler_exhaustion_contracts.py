"""Direct scheduler contracts for exhaustion and sparse intersections."""

from __future__ import annotations

from datetime import date
import unittest

import nautical_core as core
from nautical_core import scheduler_expr


class SchedulerExhaustionContractTests(unittest.TestCase):
    def test_sparse_and_rule_finds_its_real_distant_match(self) -> None:
        seed = date(2026, 1, 1)
        dnf = core.validate_anchor_expr_strict("w/100:mon + y:01-01")

        result, _metadata = core.next_after_expr(
            dnf, seed, default_seed=seed
        )

        self.assertEqual(result, date(3151, 1, 1))
        self.assertTrue(all(core.factor_matches_on(atom, result, seed) for atom in dnf[0]))

    def test_exhausted_or_branch_does_not_hide_a_valid_alternative(self) -> None:
        reference = date(2026, 1, 1)
        exhausted = core.OccurrenceSearchExhausted(
            "test branch", reference=reference, limit=2
        )

        def next_for_and(term, _reference, _seed, seed_base=None):
            if term[0].get("spec") == "exhausted":
                raise exhausted
            return date(2026, 1, 5)

        result = scheduler_expr.next_for_or(
            [[{"typ": "w", "spec": "exhausted"}], [{"typ": "w", "spec": "valid"}]],
            reference,
            reference,
            next_for_and=next_for_and,
        )

        self.assertEqual(result, date(2026, 1, 5))

    def test_periodic_sparse_intersection_uses_gregorian_cycle(self) -> None:
        seed = date(2026, 1, 1)
        dnf = core.validate_anchor_expr_strict("w/20:mon + m:1")

        result, _metadata = core.next_after_expr(
            dnf, seed, default_seed=seed
        )

        self.assertEqual(result, date(2134, 2, 1))

    def test_date_and_search_limits_are_typed_exhaustion(self) -> None:
        dnf = core.validate_anchor_expr_strict("w:mon")
        with self.assertRaises(core.OccurrenceSearchExhausted) as boundary:
            core.next_after_expr(dnf, date.max, default_seed=date.max)
        self.assertEqual(boundary.exception.scope, "simple weekly scheduling")
        self.assertEqual(boundary.exception.kind, boundary.exception.DATE_LIMIT)
        self.assertTrue(boundary.exception.is_date_limit)

        search_limited = core.OccurrenceSearchExhausted(
            "bounded test", reference=date(2026, 1, 1), limit=1
        )
        self.assertEqual(search_limited.kind, search_limited.SEARCH_LIMIT)

        huge = core.validate_anchor_expr_strict("w/1000000000:mon")
        with self.assertRaises(core.OccurrenceSearchExhausted) as extreme:
            core.next_after_expr(huge, date(2026, 1, 1), default_seed=date(2026, 1, 1))
        self.assertTrue(extreme.exception.is_date_limit)


if __name__ == "__main__":
    unittest.main()
