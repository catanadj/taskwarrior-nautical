"""Contracts for recurrence token normalization."""

from __future__ import annotations

import unittest
from nautical_core.tokenutil import normalize_weekday


class TokenUtilContracts(unittest.TestCase):
    def test_invalid_weekday_number_remains_a_non_match(self) -> None:
        self.assertIsNone(normalize_weekday("not-a-weekday"))

    def test_unexpected_weekday_conversion_failure_propagates(self) -> None:
        class BrokenWeekdayToken(str):
            def strip(self) -> BrokenWeekdayToken:
                return self

            def lower(self) -> BrokenWeekdayToken:
                return self

            def __int__(self) -> int:
                raise RuntimeError("weekday token conversion failed")

        with self.assertRaisesRegex(RuntimeError, "weekday token conversion failed"):
            normalize_weekday(BrokenWeekdayToken("8"))


if __name__ == "__main__":
    unittest.main()
