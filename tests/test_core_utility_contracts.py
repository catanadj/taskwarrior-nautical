from __future__ import annotations

from datetime import date
import unittest

from nautical_core.common import coerce_int, short_uuid
from nautical_core.scheduler_api import _weeks_between


class CoreUtilityContractTests(unittest.TestCase):
    def test_week_count_uses_iso_week_boundaries(self) -> None:
        self.assertEqual(_weeks_between(date(2024, 12, 31), date(2025, 1, 1)), 0)
        self.assertEqual(_weeks_between(date(2024, 12, 29), date(2024, 12, 30)), 1)

    def test_short_uuid_handles_invalid_and_short_values(self) -> None:
        self.assertEqual(short_uuid(None), "")
        self.assertEqual(short_uuid(1234), "")
        self.assertEqual(short_uuid("abcd"), "abcd")

    def test_integer_coercion_rejects_values_above_supported_bounds(self) -> None:
        too_large = 2**63
        self.assertEqual(coerce_int(too_large, default=7), 7)
        self.assertEqual(coerce_int(float(too_large), default=7), 7)


if __name__ == "__main__":
    unittest.main()
