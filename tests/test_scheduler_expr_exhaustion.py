from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from nautical_core.scheduler_expr import next_after_expr
from nautical_core.scheduler_models import OccurrenceSearchExhausted


class SchedulerExpressionExhaustionTests(unittest.TestCase):
    def test_exhausted_ordinary_term_does_not_abort_sibling(self) -> None:
        calls = iter([
            OccurrenceSearchExhausted("ordinary", kind=OccurrenceSearchExhausted.SEARCH_LIMIT),
            (date(2026, 1, 3), {"term": 1}),
        ])
        with patch("nautical_core.scheduler_expr._next_after_expr_term_candidate", side_effect=calls):
            result, meta = next_after_expr(
                [[{"typ": "d", "spec": "1"}], [{"typ": "d", "spec": "2"}]],
                date(2026, 1, 1), None,
                active_mod_keys=set(),
                expand_weekly_cached=lambda _: [],
                term_rand_info=lambda _: (None, None),
                atype=lambda *_: None,
                next_for_and=lambda *_args, **_kwargs: None,
                months_since=lambda *_: 0,
                term_candidates_in_month=lambda *_: [],
                random_identity=lambda _: "test",
                random_pick_indices=lambda *_args, **_kwargs: [],
                atom_matches_on=lambda *_args, **_kwargs: True,
                next_after_term=lambda *_args, **_kwargs: None,
            )
        self.assertEqual(result, date(2026, 1, 3))
        self.assertEqual(meta, {"term": 1})


if __name__ == "__main__":
    unittest.main()
