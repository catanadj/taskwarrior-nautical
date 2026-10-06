from __future__ import annotations

import re
import unittest
from unittest.mock import patch

import nautical_core.cached_expansion as cached_expansion


class CachedExpansionContractTests(unittest.TestCase):
    def test_monthly_range_does_not_hide_unexpected_integer_parser_errors(self) -> None:
        with patch.object(
            cached_expansion,
            "int",
            side_effect=RuntimeError("integer parser invariant failed"),
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "integer parser invariant failed"):
                cached_expansion.month_tokens_for_atom_values(
                    2026,
                    5,
                    "1..3",
                    expand_monthly_aliases=lambda value: value,
                    days_in_month=lambda _year, _month: 31,
                    bd_re=re.compile(r"^bd(-?\d+)$"),
                    nth_weekday_re=re.compile(r"^([\w-]+)-([a-z]+)$"),
                    weekday_map={},
                    re_mod=re,
                )


if __name__ == "__main__":
    unittest.main()
